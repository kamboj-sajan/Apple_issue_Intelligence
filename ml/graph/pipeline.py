"""Build, validate, analyse and snapshot the knowledge graph for one data snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml.data.raw_store import atomic_write_text
from ml.graph import analysis as A
from ml.graph.build import GraphConfig, build_graph_tables
from ml.graph.schema import validate


def run_graph(
    processed_dir: Path,
    out_dir: Path,
    emb,
    emb_ids,
    signals: pd.DataFrame | None,
    cfg: GraphConfig,
    seed: int = 0,
) -> dict[str, Any]:
    questions = pd.read_parquet(processed_dir / "questions.parquet")
    entities = pd.read_parquet(processed_dir / "question_entities.parquet")
    links = pd.read_parquet(processed_dir / "question_links.parquet")
    splits = pd.read_parquet(processed_dir / "splits.parquet")
    snapshot = json.loads((processed_dir / "manifest.json").read_text())["snapshot_id"]

    nodes, edges = build_graph_tables(questions, entities, links, emb, emb_ids, signals, splits, cfg)
    checks = validate(nodes, edges)
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(out_dir / "nodes.parquet", index=False)
    edges.to_parquet(out_dir / "edges.parquet", index=False)

    stats = A.graph_statistics(nodes, edges)
    A.degree_distribution(nodes, edges).to_csv(out_dir / "degree_distribution.csv", index=False)

    qq = A.question_projection(nodes, edges)
    comp = A.component_summary(qq)
    labels = A.communities(qq, seed=seed)
    labels.rename_axis("node_id").reset_index().to_parquet(out_dir / "communities.parquet", index=False)
    desc = A.describe_communities(labels, nodes, edges)
    desc.to_parquet(out_dir / "community_summary.parquet", index=False)
    bridges = {p: A.bridging_communities(labels, edges, p) for p in ("tag", "os", "device")}
    for p, b in bridges.items():
        b.to_csv(out_dir / f"bridges_{p}.csv", index=False)

    last = questions["created_at"].max().tz_convert("UTC")
    years = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), last, freq="YS").tolist()
    as_of = [*years, last + pd.Timedelta(seconds=1)]
    snaps = A.snapshot_series(nodes, edges, as_of)
    snaps.to_csv(out_dir / "snapshots.csv", index=False)

    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot_id": snapshot,
        "config": asdict(cfg),
        "validation": checks,
        "statistics": stats,
        "question_projection_components": comp,
        "n_communities": int(labels[labels >= 0].nunique()),
        "questions_in_small_communities": int((labels < 0).sum()),
        "top_bridges": {p: b.head(5).to_dict("records") for p, b in bridges.items()},
        "snapshots_as_of": [str(t) for t in as_of],
        "louvain_seed": seed,
    }
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, default=str))
    return summary
