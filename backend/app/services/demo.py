"""Small SYNTHETIC demo dataset, clearly labelled as demo data.

Used when the Stack Exchange API is unavailable (``aii-db load-demo``) and by the API tests. Every
response produced from it carries ``meta.data_source = "demo"``. Titles and numbers are invented and
do not describe real questions; links point to ``example.invalid``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from backend.app.services.loader import write_frames

DEMO_SNAPSHOT = "demo-synthetic"
T0 = datetime(2024, 1, 1, tzinfo=UTC)  # a Monday
TOPICS = [
    ("wifi", "network-connectivity", "macbook-pro", "macos:14", "Wi-Fi drops after waking from sleep"),
    ("battery", "battery-power", "iphone", "ios:17", "Battery drains quickly overnight"),
    ("time-machine", "backup-sync", "imac", "macos:13", "Time Machine backup fails to complete"),
    ("bluetooth", "bluetooth-peripherals", "macbook-air", "macos:14", "Bluetooth keyboard disconnects randomly"),
]


def _questions(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    qid = 1000
    for week in range(30):
        n = 4 + (6 if week == 25 else 0)  # synthetic spike for the wifi topic in week 25
        for i in range(n):
            topic = 0 if (week == 25 and i >= 4) else int(rng.integers(0, len(TOPICS)))
            tag, cat, dev, os_v, title = TOPICS[topic]
            emb = np.zeros(384, dtype=np.float32)
            emb[topic * 10 : topic * 10 + 10] = 1.0
            emb += rng.normal(0, 0.05, 384).astype(np.float32)
            emb /= np.linalg.norm(emb)
            rows.append(
                {
                    "question_id": qid,
                    "title": f"[DEMO] {title} (#{qid})",
                    "link": f"https://example.invalid/demo/questions/{qid}",
                    "body_excerpt": f"SYNTHETIC DEMO QUESTION about {tag}.",
                    "created_at": T0 + timedelta(weeks=week, hours=int(rng.integers(0, 160))),
                    "last_activity_at": None,
                    "tags": [tag, "macos" if os_v.startswith("macos") else "ios"],
                    "devices": [dev],
                    "os_versions": [os_v],
                    "error_codes": [],
                    "categories": [cat],
                    "primary_category": cat,
                    "score": int(rng.integers(-1, 10)),
                    "view_count": int(rng.integers(5, 500)),
                    "answer_count": int(rng.integers(0, 3)),
                    "is_closed_duplicate": False,
                    "duplicate_of": [],
                    "author_display_name": "demo-user",
                    "author_link": "https://example.invalid/users/demo",
                    "content_license": "demo",
                    "split": "train" if week < 20 else ("val" if week < 25 else "test"),
                    "in_research_window": True,
                    "embedding": emb,
                }
            )
            qid += 1
    return pd.DataFrame(rows)


def build_demo_frames() -> tuple[dict, dict, dict]:
    rng = np.random.default_rng(42)
    q = _questions(rng)
    q.loc[q.index[-1], ["is_closed_duplicate"]] = True
    q.at[q.index[-1], "duplicate_of"] = [int(q["question_id"].iloc[-2])]

    ent = pd.concat(
        [
            pd.DataFrame(
                {
                    "question_id": q["question_id"],
                    "entity_type": "device",
                    "value": q["devices"].str[0],
                    "surface": q["devices"].str[0],
                    "field": "title",
                }
            ),
            pd.DataFrame(
                {
                    "question_id": q["question_id"],
                    "entity_type": "os",
                    "value": q["os_versions"].str[0],
                    "surface": q["os_versions"].str[0],
                    "field": "body",
                }
            ),
            pd.DataFrame(
                {
                    "question_id": q["question_id"],
                    "entity_type": "category",
                    "value": q["primary_category"],
                    "surface": "demo rule",
                    "field": "rules",
                }
            ),
        ],
        ignore_index=True,
    )

    weeks = [T0 + timedelta(weeks=w) for w in range(30)]
    week_of = q["created_at"].map(lambda t: weeks[min((t - T0).days // 7, 29)])
    totals = week_of.value_counts().reindex(weeks, fill_value=0)
    series_rows = []
    for tag, *_ in TOPICS:
        counts = week_of[q["tags"].map(lambda ts, t=tag: t in ts)].value_counts().reindex(weeks, fill_value=0)
        share = counts.rolling(8, min_periods=4).sum().shift(1) / totals.rolling(8, min_periods=4).sum().shift(1)
        for w in weeks:
            series_rows.append(
                {
                    "group_type": "tag",
                    "group_value": tag,
                    "week_start": w,
                    "observed": int(counts[w]),
                    "expected_negbin": float(share[w] * totals[w]) if pd.notna(share[w]) else None,
                    "expected_rolling": float(share[w] * totals[w]) if pd.notna(share[w]) else None,
                    "site_total": int(totals[w]),
                }
            )
    series = pd.DataFrame(series_rows)
    spike = series[(series["group_value"] == "wifi") & (series["week_start"] == weeks[25])].iloc[0]
    anomalies = pd.DataFrame(
        [
            {
                "anomaly_id": "demo0000wifi0025",
                "group_type": "tag",
                "group_value": "wifi",
                "week_start": weeks[25],
                "observed": int(spike["observed"]),
                "expected": float(spike["expected_negbin"]),
                "abs_change": float(spike["observed"] - spike["expected_negbin"]),
                "rel_change": float(spike["observed"] / spike["expected_negbin"] - 1),
                "p_value": 1e-4,
                "q_value": 4e-4,
                "nb_score": 4.2,
                "rolling_z": 3.9,
                "rolling_agrees": True,
                "overdispersion_alpha": 0.0,
                "signal_type": "spike",
                "rank_score": 5.1,
                "site_total": int(totals[weeks[25]]),
                "baseline_start": weeks[0],
                "baseline_end": weeks[24],
                "partial_week": False,
                "low_site_volume": False,
                "evidence_question_ids": q.loc[
                    (week_of == weeks[25]) & q["tags"].map(lambda t: "wifi" in t), "question_id"
                ].tolist()[:5],
                "co_occurring": ["network-connectivity (6)", "macos:14 (6)"],
                "limitations": "DEMO DATA. Synthetic spike injected for demonstration; "
                "not an analysis of real questions.",
            }
        ]
    )
    weekly = pd.DataFrame(
        {"week_start": weeks, "site_total": totals.to_numpy(), "partial_week": False, "low_site_volume": False}
    )

    nodes = [
        {"node_id": f"question:{i}", "node_type": "question", "label": t, "known_at": c, "degree": 0}
        for i, t, c in zip(q["question_id"], q["title"], q["created_at"], strict=True)
    ]
    edges = []
    for r in q.itertuples():
        edges.append(
            {
                "src": f"question:{r.question_id}",
                "dst": f"tag:{r.tags[0]}",
                "edge_type": "QUESTION_HAS_TAG",
                "weight": 1.0,
                "known_at": r.created_at,
                "evidence": "tag",
            }
        )
        edges.append(
            {
                "src": f"question:{r.question_id}",
                "dst": f"device:{r.devices[0]}",
                "edge_type": "QUESTION_MENTIONS_DEVICE",
                "weight": 1.0,
                "known_at": r.created_at,
                "evidence": "title: demo",
            }
        )
    for tag, _, dev, *_ in TOPICS:
        nodes.append({"node_id": f"tag:{tag}", "node_type": "tag", "label": tag, "known_at": T0, "degree": 0})
        nodes.append({"node_id": f"device:{dev}", "node_type": "device", "label": dev, "known_at": T0, "degree": 0})
    last, prev = int(q["question_id"].iloc[-1]), int(q["question_id"].iloc[-2])
    edges.append(
        {
            "src": f"question:{last}",
            "dst": f"question:{prev}",
            "edge_type": "QUESTION_DUPLICATE_OF",
            "weight": 1.0,
            "known_at": q["created_at"].iloc[-1],
            "evidence": "duplicate",
        }
    )
    nodes_df = pd.DataFrame(nodes).drop_duplicates("node_id")
    edges_df = pd.DataFrame(edges).drop_duplicates(["src", "dst", "edge_type"])
    deg = pd.concat([edges_df["src"], edges_df["dst"]]).value_counts()
    nodes_df["degree"] = nodes_df["node_id"].map(deg).fillna(0).astype(int)

    E = np.vstack(q["embedding"].to_numpy())
    related = []
    for i in range(1, len(q)):
        sims = E[:i] @ E[i]
        for method, scale in (("bm25", 10.0), ("embedding", 1.0), ("graphsage", 5.0)):
            for rank, j in enumerate(np.argsort(-sims)[:5], start=1):
                related.append(
                    {
                        "question_id": int(q["question_id"].iloc[i]),
                        "method": method,
                        "rank": rank,
                        "related_id": int(q["question_id"].iloc[j]),
                        "score": float(sims[j] * scale),
                    }
                )
    idf = pd.DataFrame(
        {"term": ["wifi", "battery", "backup", "bluetooth", "drops", "sleep"], "idf": [3.1, 2.9, 3.3, 3.0, 4.0, 2.5]}
    )

    months = [f"2024-{m:02d}" for m in range(4, 8)]
    drift = pd.DataFrame(
        [
            {
                "month": m,
                "metric": met,
                "status": "ok",
                "value": 0.01,
                "z": z,
                "p_value": 0.3 if z < 3 else 0.01,
                "threshold": 3.0,
                "alert": z > 3,
                "calibration": m < "2024-06",
                "n": 50,
                "n_current": 60,
                "reference": "demo",
            }
            for m, z in zip(months, [0.5, 1.0, 1.2, 4.5], strict=True)
            for met in ("mmd_rbf", "tags")
        ]
    )
    terms = pd.DataFrame([{"month": "2024-07", "term": "demo-term", "docs": 6}])
    metrics = {"splits": {"demo": True, "note": "Synthetic demo dataset; no model was evaluated."}}
    frames = {
        "questions": q,
        "question_entities": ent,
        "anomalies": anomalies,
        "anomaly_series": series,
        "weekly_totals": weekly,
        "graph_nodes": nodes_df,
        "graph_edges": edges_df,
        "related_questions": pd.DataFrame(related),
        "term_idf": idf,
        "drift_metrics": drift,
        "drift_new_terms": terms,
    }
    snapshot_row = {
        "snapshot_id": DEMO_SNAPSHOT,
        "is_demo": True,
        "loaded_at": datetime.now(UTC),
        "pipeline_version": "demo",
        "created_range_start": q["created_at"].min(),
        "created_range_end": q["created_at"].max(),
        "n_questions": len(q),
        "manifest": '{"demo": true, "quality": {}}',
        "collection": '{"demo": true}',
    }
    return frames, snapshot_row, metrics


def load_demo(engine: Engine) -> dict:
    frames, snapshot_row, metrics = build_demo_frames()
    counts = write_frames(engine, frames, snapshot_row, metrics)
    return {"snapshot_id": DEMO_SNAPSHOT, "is_demo": True, "rows": counts}
