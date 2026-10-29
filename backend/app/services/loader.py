"""Load one snapshot's pipeline artifacts into PostgreSQL (replacing the previous contents).

Bulk inserts use PostgreSQL ``COPY``. Everything happens in one transaction, so the API sees either
the old snapshot or the new one, never a mix.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pgvector.psycopg import register_vector
from sqlalchemy import Engine

from backend.app.db.models import Base

log = logging.getLogger(__name__)

DATA_TABLES = [t.name for t in Base.metadata.sorted_tables if t.name not in ("jobs", "alembic_version")]
BODY_EXCERPT_CHARS = 600


def _clean(v: Any) -> Any:
    """Convert pandas/numpy values to plain Python values psycopg can adapt."""
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        return None if pd.isna(v) else v.to_pydatetime()
    if isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.ndim == 1 and v.size > 50:
        return v  # embedding vectors (pgvector adapter)
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if v is pd.NaT or v is pd.NA:
        return None
    return v


def copy_rows(cur, table: str, columns: list[str], rows: Iterable[Iterable[Any]]) -> int:
    n = 0
    with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN") as cp:
        for row in rows:
            cp.write_row([_clean(v) for v in row])
            n += 1
    return n


def copy_frame(cur, table: str, df: pd.DataFrame, columns: list[str] | None = None) -> int:
    columns = columns or list(df.columns)
    return copy_rows(cur, table, columns, df[columns].itertuples(index=False, name=None))


def write_frames(engine: Engine, frames: dict[str, pd.DataFrame], snapshot_row: dict, metrics: dict[str, dict]) -> dict:
    """Replace all data tables with ``frames`` (table name -> DataFrame) inside one transaction."""
    raw = engine.raw_connection()
    counts = {}
    try:
        conn = raw.driver_connection
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE {', '.join(DATA_TABLES)}")
            copy_rows(cur, "dataset_snapshots", list(snapshot_row), [list(snapshot_row.values())])
            for table, df in frames.items():
                if df is not None and len(df):
                    counts[table] = copy_frame(cur, table, df)
            now = datetime.now(UTC)
            copy_rows(
                cur,
                "model_metrics",
                ["kind", "created_at", "payload"],
                [(k, now, json.dumps(v, default=str)) for k, v in metrics.items()],
            )
        conn.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    return counts


# ------------------------------------------------------------------------------- real snapshot


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _entity_lists(entities: pd.DataFrame, entity_type: str, drop_models: bool = False) -> pd.Series:
    e = entities[entities["entity_type"] == entity_type]
    if drop_models:
        e = e[~e["value"].str.contains(":")]
    return e.groupby("question_id")["value"].apply(lambda s: sorted(set(s)))


def build_frames(data_dir: Path, with_related: bool = True, with_graphsage: bool = True) -> tuple[dict, dict, dict]:
    processed, artifacts = data_dir / "processed", data_dir / "artifacts"
    manifest = json.loads((processed / "manifest.json").read_text())
    snap = manifest["snapshot_id"]
    q = (
        pd.read_parquet(processed / "questions.parquet")
        .sort_values(["created_at", "question_id"])
        .reset_index(drop=True)
    )
    ent = pd.read_parquet(processed / "question_entities.parquet")
    splits = pd.read_parquet(processed / "splits.parquet").set_index("question_id")

    emb = emb_ids = None
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    edir = artifacts / "embeddings" / snap
    meta = _json(edir / "meta.json")
    if meta:
        model_name = meta["model"]
        slug = model_name.split("/")[-1]
        emb_all = np.load(edir / f"{slug}.npy")
        emb_ids = pd.read_parquet(edir / f"{slug}.ids.parquet")["question_id"].to_numpy()
        emb = emb_all[pd.Series(np.arange(len(emb_ids)), index=emb_ids).loc[q["question_id"]].to_numpy()]

    def lists(series: pd.Series) -> list:
        return [v if isinstance(v, list) else [] for v in q["question_id"].map(series)]

    questions = pd.DataFrame(
        {
            "question_id": q["question_id"],
            "title": q["title"],
            "link": q["link"],
            "body_excerpt": q["body_text"].str.slice(0, BODY_EXCERPT_CHARS),
            "created_at": q["created_at"],
            "last_activity_at": q["last_activity_at"],
            "tags": q["tags"].map(list),
            "devices": lists(_entity_lists(ent, "device", drop_models=True)),
            "os_versions": lists(_entity_lists(ent, "os")),
            "error_codes": lists(_entity_lists(ent, "error")),
            "categories": q["categories"].map(list),
            "primary_category": q["primary_category"],
            "score": q["score"],
            "view_count": q["view_count"],
            "answer_count": q["answer_count"],
            "is_closed_duplicate": q["is_closed_duplicate"],
            "duplicate_of": q["duplicate_of"].map(list),
            "author_display_name": q["author_display_name"],
            "author_link": q["author_link"],
            "content_license": q["content_license"],
            "split": q["question_id"].map(splits["split"]),
            "in_research_window": q["in_research_window"],
        }
    )
    if emb is not None:
        questions["embedding"] = list(emb)

    frames: dict[str, pd.DataFrame | None] = {
        "questions": questions,
        "question_entities": ent[["question_id", "entity_type", "value", "surface", "field"]],
    }

    adir = artifacts / "anomalies" / snap
    sig = _read(adir / "signals.parquet")
    if sig is not None:
        frames["anomalies"] = sig[
            [
                "anomaly_id",
                "group_type",
                "group_value",
                "week_start",
                "observed",
                "expected",
                "abs_change",
                "rel_change",
                "p_value",
                "q_value",
                "nb_score",
                "rolling_z",
                "rolling_agrees",
                "overdispersion_alpha",
                "signal_type",
                "rank_score",
                "site_total",
                "baseline_start",
                "baseline_end",
                "partial_week",
                "low_site_volume",
                "evidence_question_ids",
                "co_occurring",
                "limitations",
            ]
        ].assign(
            evidence_question_ids=lambda d: d["evidence_question_ids"].map(list),
            co_occurring=lambda d: d["co_occurring"].map(list),
        )
    frames["anomaly_series"] = _read(adir / "series.parquet")
    frames["weekly_totals"] = _read(adir / "weekly_totals.parquet")

    gdir = artifacts / "graph" / snap
    nodes, edges = _read(gdir / "nodes.parquet"), _read(gdir / "edges.parquet")
    if nodes is not None and edges is not None:
        deg = pd.concat([edges["src"], edges["dst"]]).value_counts()
        frames["graph_nodes"] = nodes[["node_id", "node_type", "label", "known_at"]].assign(
            degree=nodes["node_id"].map(deg).fillna(0).astype(int)
        )
        frames["graph_edges"] = edges[["src", "dst", "edge_type", "weight", "known_at", "evidence"]]

    if with_related:
        from backend.app.services.related_precompute import load_graphsage, precompute

        gs = (
            load_graphsage(artifacts, snap, q, emb, emb_ids, model_name)
            if (with_graphsage and emb is not None)
            else None
        )
        related, idf = precompute(q, emb, graphsage=gs)
        frames["related_questions"], frames["term_idf"] = related, idf

    ddir = artifacts / "drift" / snap
    dm = _read(ddir / "drift_metrics.parquet")
    if dm is not None:
        frames["drift_metrics"] = dm[
            [
                "month",
                "metric",
                "status",
                "value",
                "z",
                "p_value",
                "threshold",
                "alert",
                "calibration",
                "n",
                "n_current",
                "reference",
            ]
        ].astype({"n": "Int64", "n_current": "Int64"})  # nullable ints (NaN for insufficient months)
    mv = _read(ddir / "cluster_movements.parquet")
    if mv is not None and len(mv):
        frames["drift_cluster_movements"] = mv[
            ["month", "cluster", "count", "share", "reference_share", "share_ratio", "direction", "top_terms"]
        ].assign(top_terms=lambda d: d["top_terms"].map(list))
    nt = _read(ddir / "new_terms.csv")
    if nt is not None and len(nt):
        frames["drift_new_terms"] = nt[["month", "term", "docs"]].drop_duplicates(["month", "term"])

    metrics = _collect_metrics(data_dir, snap, adir, ddir)
    ckpt = _json(data_dir / "state" / "ingest_apple.json") or {}  # checkpoint written by `aii ingest` (site=apple)
    snapshot_row = {
        "snapshot_id": snap,
        "is_demo": False,
        "loaded_at": datetime.now(UTC),
        "pipeline_version": manifest.get("pipeline_version"),
        "created_range_start": pd.Timestamp(q.loc[q["in_research_window"], "created_at"].min()),
        "created_range_end": pd.Timestamp(q.loc[q["in_research_window"], "created_at"].max()),
        "n_questions": int(q["in_research_window"].sum()),
        "manifest": json.dumps(manifest, default=str),
        "collection": json.dumps(
            {"last_runs": ckpt.get("runs", [])[-5:], "update_watermark": ckpt.get("update", {}).get("watermark")},
            default=str,
        ),
    }
    return frames, snapshot_row, metrics


def _collect_metrics(data_dir: Path, snap: str, adir: Path, ddir: Path) -> dict[str, dict]:
    art = data_dir / "artifacts"
    out: dict[str, dict] = {}
    lp = art / "link_prediction" / snap
    if (lp / "report.json").exists():
        out["link_prediction"] = {
            "report": _json(lp / "report.json"),
            "ranking": _read(lp / "ranking_summary.csv").to_dict("records"),
            "classification": _read(lp / "classification_summary.csv").to_dict("records"),
            "paired": _read(lp / "paired_mrr_differences.csv").to_dict("records"),
        }
    if (art / "retrieval" / snap / "manifest.json").exists():
        out["retrieval"] = {
            "manifest": _json(art / "retrieval" / snap / "manifest.json"),
            "summary": _read(art / "retrieval" / snap / "summary.csv").to_dict("records"),
        }
    if (art / "benchmarks" / "ppi_results.json").exists():
        ppi = _json(art / "benchmarks" / "ppi_results.json")
        out["ppi_reproduction"] = {
            k: ppi[k] for k in ("paper", "paper_micro_f1", "dataset", "summary", "environment", "created_at")
        }
    if (adir / "summary.json").exists():
        ev = adir / "evaluation"
        out["anomaly"] = {
            "summary": _json(adir / "summary.json"),
            "injection": _read(ev / "injection_summary.csv").to_dict("records")
            if (ev / "injection_summary.csv").exists()
            else [],
            "null_calibration": _read(ev / "null_calibration.csv").to_dict("records")
            if (ev / "null_calibration.csv").exists()
            else [],
        }
    if (ddir / "summary.json").exists():
        evals = {}
        for name in ("eval_no_drift", "eval_injected"):
            df = _read(ddir / f"{name}.csv")
            if df is not None:
                evals[name] = (
                    df.groupby("metric")["alert"].mean().round(3).to_dict()
                    if name == "eval_no_drift"
                    else df.groupby(["fraction", "metric"])["alert"]
                    .mean()
                    .round(3)
                    .unstack()
                    .reset_index()
                    .to_dict("records")
                )
        out["drift"] = {
            "summary": _json(ddir / "summary.json"),
            "novelty_rate": _read(ddir / "novelty_rate.parquet").to_dict("records"),
            "new_cluster_candidates": _json(ddir / "new_cluster_candidates.json") or [],
            "evaluation": evals,
        }
    processed = data_dir / "processed"
    out["splits"] = {
        "split_manifest": _json(processed / "split_manifest.json"),
        "clusters": {k: v for k, v in (_json(processed / "clusters_meta.json") or {}).items() if k != "sizes"},
    }
    return out


def load_snapshot(engine: Engine, data_dir: Path, with_related: bool = True, with_graphsage: bool = True) -> dict:
    frames, snapshot_row, metrics = build_frames(data_dir, with_related, with_graphsage)
    counts = write_frames(engine, frames, snapshot_row, metrics)
    return {"snapshot_id": snapshot_row["snapshot_id"], "is_demo": False, "rows": counts, "metrics": sorted(metrics)}
