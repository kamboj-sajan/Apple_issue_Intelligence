"""API response validation against the SYNTHETIC demo dataset (data_source must be 'demo')."""

from __future__ import annotations

import pytest

for _mod in ("fastapi", "sqlalchemy", "psycopg", "pgvector", "pydantic_settings"):
    pytest.importorskip(_mod, reason="API tests need the api extras: pip install -e '.[api]'")

from sqlalchemy import inspect, text  # noqa: E402


def test_health_and_demo_label(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["meta"]["data_source"] == "demo"


def test_overview(client):
    d = client.get("/overview").json()["data"]
    assert d["total_questions"] == 126 and d["anomaly_signals"] == 1


def test_questions_pagination_and_filters(client):
    r = client.get("/questions", params={"limit": 5, "offset": 0, "sort": "oldest"}).json()
    assert r["total"] == 126 and len(r["items"]) == 5 and r["limit"] == 5
    assert r["items"][0]["created_at"] <= r["items"][-1]["created_at"]
    page2 = client.get("/questions", params={"limit": 5, "offset": 5, "sort": "oldest"}).json()
    assert {i["question_id"] for i in r["items"]}.isdisjoint({i["question_id"] for i in page2["items"]})
    wifi = client.get("/questions", params={"tag": "wifi", "limit": 100}).json()
    assert wifi["total"] > 0 and all("wifi" in i["tags"] for i in wifi["items"])
    dev = client.get("/questions", params={"device": "iphone", "limit": 100}).json()
    assert all("iphone" in i["devices"] for i in dev["items"])
    os_ = client.get("/questions", params={"os": "macos:14", "limit": 100}).json()
    assert all("macos:14" in i["os_versions"] for i in os_["items"])
    dated = client.get("/questions", params={"start_date": "2024-03-01", "end_date": "2024-03-31", "limit": 100}).json()
    assert all("2024-03" in i["created_at"] for i in dated["items"])


@pytest.mark.parametrize(
    "params",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"sort": "bogus"},
        {"start_date": "not-a-date"},
        {"start_date": "2024-05-01", "end_date": "2024-01-01"},
    ],
)
def test_questions_validation_errors_are_structured(client, params):
    r = client.get("/questions", params=params)
    assert r.status_code == 422
    body = r.json()["error"]
    assert body["code"] == "validation_error" and body["message"]


def test_question_detail_and_404(client):
    d = client.get("/questions/1005").json()["data"]
    assert d["question_id"] == 1005 and d["attribution"]["source"].startswith("Ask Different")
    assert {e["entity_type"] for e in d["entities"]} >= {"device", "os"}
    r = client.get("/questions/999999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_anomalies_list_filters_and_detail(client):
    r = client.get("/anomalies", params={"tag": "wifi"}).json()
    assert r["total"] == 1
    assert client.get("/anomalies", params={"tag": "battery"}).json()["total"] == 0
    assert client.get("/anomalies", params={"min_score": 100}).json()["total"] == 0
    a = client.get("/anomalies/demo0000wifi0025").json()["data"]
    assert a["observed"] > a["expected"] and a["series"] and a["evidence_questions"]
    assert {"triggered_window", "baseline_period", "observed_vs_expected"} <= set(a["explanation"])
    assert "DEMO DATA" in a["limitations"]
    assert client.get("/anomalies/missing").status_code == 404


def test_trending_and_series(client):
    t = client.get("/issues/trending", params={"weeks": 8}).json()["data"]
    assert t["categories"] and t["recent_signals"] and "not confirmed defects" in t["note"]
    s = client.get("/issues/series", params={"group_type": "tag", "group_value": "wifi"}).json()["data"]
    assert len(s) == 30
    assert client.get("/issues/series", params={"group_type": "tag", "group_value": "nope"}).status_code == 404


def test_graph_neighbors_limits_and_filters(client):
    d = client.get("/graph/neighbors/tag:wifi", params={"limit": 5}).json()["data"]
    assert d["center"]["node_id"] == "tag:wifi" and len(d["edges"]) <= 5 and d["truncated"]
    d2 = client.get("/graph/neighbors/tag:wifi", params={"depth": 2, "max_nodes": 10}).json()["data"]
    assert len(d2["nodes"]) <= 10
    q = client.get("/graph/neighbors/tag:wifi", params={"node_types": "device"}).json()["data"]
    assert q["edges"] == []  # tag:wifi only connects to questions
    assert client.get("/graph/neighbors/tag:wifi", params={"node_types": "planet"}).status_code == 422
    assert client.get("/graph/neighbors/tag:nothing").status_code == 404


def test_related_questions_methods_and_explanations(client):
    d = client.get("/related-questions/1100", params={"k": 3}).json()["data"]
    assert [m["method"] for m in d["methods"]] == ["bm25", "embedding", "graphsage"]
    for m in d["methods"]:
        assert m["score_description"] and 0 < len(m["items"]) <= 3
        for item in m["items"]:
            assert item["created_at"] < d["question"]["created_at"]  # earlier questions only
            assert set(item["explanation"]) >= {"shared_tags", "shared_entities", "lexical_overlap", "embedding_cosine"}
    assert client.get("/related-questions/1100", params={"methods": "foo"}).status_code == 422
    assert client.get("/related-questions/1100", params={"k": 11}).status_code == 422


def test_metrics_drift_and_jobs_validation(client):
    m = client.get("/metrics/model").json()
    assert m["meta"]["data_source"] == "demo" and m["leakage_controls"]
    dr = client.get("/drift", params={"alerts_only": True}).json()["data"]
    assert dr["series"] and all(s["alert"] for s in dr["series"])
    assert client.get("/drift", params={"start_month": "2024/01"}).status_code == 422
    bad = client.post("/jobs/ingest", json={"mode": "everything"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "validation_error"
    assert client.post("/jobs/analyze", json={"steps": ["rm -rf"]}).status_code == 422
    assert client.get("/jobs/does-not-exist").status_code == 404


def test_job_lifecycle_runs_without_blocking(client, monkeypatch):
    """A job executes as a subprocess in a background thread; the request returns immediately."""
    import time

    from backend.app.services import jobs as J

    monkeypatch.setattr(J, "analyze_steps", lambda steps, load: [["ml.cli", "--help"]])
    r = client.post("/jobs/analyze", json={"steps": ["split"], "load_database": False})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    second = client.post("/jobs/analyze", json={"steps": ["split"], "load_database": False})
    assert second.status_code in (202, 409)  # 409 if the first is still running
    for _ in range(100):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert status["status"] == "succeeded", status
    assert "usage" in status["log_tail"]


def test_migrations_create_expected_schema(engine):
    insp = inspect(engine)
    tables = set(insp.get_table_names())
    assert {"questions", "anomalies", "graph_edges", "related_questions", "drift_metrics", "jobs"} <= tables
    with engine.connect() as c:
        assert c.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
        assert c.execute(text("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")).scalar() == 1
    idx = {i["name"] for i in insp.get_indexes("questions")}
    assert {"ix_questions_tags", "ix_questions_embedding_hnsw"} <= idx


def test_migration_downgrade_and_upgrade(test_db_url):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine

    from backend.app.core.config import REPO_ROOT
    from backend.app.services.demo import load_demo

    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    import argparse

    cfg.cmd_opts = argparse.Namespace(x=[f"url={test_db_url}"])
    command.downgrade(cfg, "base")
    eng = create_engine(test_db_url)
    assert "questions" not in inspect(eng).get_table_names()
    command.upgrade(cfg, "head")
    load_demo(eng)  # restore for any later tests
    assert "questions" in inspect(eng).get_table_names()
    eng.dispose()


def test_latest_analysis_handles_mixed_timestamp_payloads():
    from datetime import UTC, datetime

    from backend.app.services.queries import latest_analysis

    metrics = {
        "anomaly": {"summary": {"created_at": "2026-10-29T10:00:00+00:00"}},
        "ppi_reproduction": {"created_at": "2026-10-29T11:30:00+00:00"},
        "drift": {"summary": {"created_at": datetime(2026, 10, 29, 9, tzinfo=UTC)}},
        "splits": {"loaded_at": datetime(2026, 10, 30, tzinfo=UTC)},  # load time is not an analysis time
    }
    assert latest_analysis(metrics) == datetime(2026, 10, 29, 11, 30, tzinfo=UTC)
    assert latest_analysis({}) is None
