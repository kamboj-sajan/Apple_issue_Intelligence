"""Command-line entry point: ``aii <command>`` or ``python -m ml.cli <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from ml.config import Config, load_config
from ml.data.ingest import Checkpoint, Ingestor, parse_date
from ml.data.raw_store import RawStore
from ml.data.stackexchange_client import StackExchangeClient


def _site_root(cfg: Config):
    return cfg.paths.raw / "stackexchange" / cfg.source["site"]


def _ingestor(cfg: Config, max_requests: int | None) -> Ingestor:
    ing = cfg.ingestion
    client = StackExchangeClient(
        site=cfg.source["site"],
        api_filter=cfg.source["filter"],
        key=cfg.api_key,
        api_base=cfg.source["api_base"],
        min_interval=ing["min_interval_seconds"],
        max_retries=ing["max_retries"],
        timeout=ing["timeout_seconds"],
    )
    return Ingestor(
        client,
        RawStore(_site_root(cfg) / "questions"),
        Checkpoint(cfg.paths.state / f"ingest_{cfg.source['site']}.json"),
        page_size=ing["page_size"],
        quota_reserve=ing["quota_reserve"],
        max_requests=max_requests,
    )


def cmd_ingest(cfg: Config, args: argparse.Namespace) -> dict:
    start = parse_date(args.start or cfg.ingestion["start"])
    end = parse_date(args.end or cfg.ingestion.get("end"))
    return _ingestor(cfg, args.max_requests).ingest_by_creation(start, end).__dict__


def cmd_update(cfg: Config, args: argparse.Namespace) -> dict:
    start = parse_date(args.start or cfg.ingestion["start"])
    return _ingestor(cfg, args.max_requests).ingest_updates(start, parse_date(args.since)).__dict__


def cmd_preprocess(cfg: Config, args: argparse.Namespace) -> dict:
    from ml.preprocessing.normalize import run_preprocessing

    return run_preprocessing(_site_root(cfg) / "questions", cfg.paths.processed, cfg.entities_config_path)


def cmd_split(cfg: Config, args: argparse.Namespace) -> dict:
    from dataclasses import asdict

    from ml.preprocessing.splits import run_splits

    s = cfg.splits
    return asdict(
        run_splits(
            cfg.paths.processed,
            train_fraction=s["train_fraction"],
            val_fraction=s["val_fraction"],
            val_start=s.get("val_start"),
            test_start=s.get("test_start"),
        )
    )


def cmd_fetch_linked(cfg: Config, args: argparse.Namespace) -> dict:
    import pandas as pd

    links = pd.read_parquet(cfg.paths.processed / "question_links.parquet")
    missing = links.loc[~links["target_in_corpus"], "target_id"].unique().tolist()
    return _ingestor(cfg, args.max_requests).ingest_ids(missing).__dict__


def _snapshot(cfg: Config) -> str:
    return json.loads((cfg.paths.processed / "manifest.json").read_text())["snapshot_id"]


def cmd_anomalies(cfg: Config, args: argparse.Namespace) -> dict:
    from ml.anomaly_detection.pipeline import AnomalyConfig, run_anomaly_detection

    acfg = AnomalyConfig.from_dict(cfg.raw.get("anomaly"))
    return run_anomaly_detection(cfg.paths.processed, cfg.paths.artifacts / "anomalies" / _snapshot(cfg), acfg)


def cmd_anomaly_eval(cfg: Config, args: argparse.Namespace) -> dict:
    import pandas as pd

    from ml.anomaly_detection.aggregate import build_memberships, weekly_counts
    from ml.anomaly_detection.evaluation import (
        InjectionConfig,
        injection_study,
        null_calibration,
        summarize_injections,
    )
    from ml.anomaly_detection.pipeline import AnomalyConfig

    acfg = AnomalyConfig.from_dict(cfg.raw.get("anomaly"))
    q = pd.read_parquet(cfg.paths.processed / "questions.parquet")
    e = pd.read_parquet(cfg.paths.processed / "question_entities.parquet")
    counts, totals, _ = weekly_counts(q, build_memberships(q, e, None, acfg.min_tagpair_support), acfg.min_group_total)
    inj = InjectionConfig(n_trials=args.trials, seed=args.seed)
    res = injection_study(counts, totals, acfg, inj)
    summary = summarize_injections(res)
    cal = null_calibration(counts, totals, acfg, seed=args.seed)
    out = cfg.paths.artifacts / "anomalies" / _snapshot(cfg) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / "injection_trials.parquet", index=False)
    summary.to_csv(out / "injection_summary.csv", index=False)
    cal.to_csv(out / "null_calibration.csv", index=False)
    return {"injection": summary.round(3).to_dict("records"), "null_calibration": cal.round(4).to_dict("records")}


def _retrieval_index(cfg: Config, with_embeddings: bool = True):
    import pandas as pd

    from ml.retrieval.embeddings import load_or_compute
    from ml.retrieval.related import RetrievalIndex

    q = pd.read_parquet(cfg.paths.processed / "questions.parquet")
    e = pd.read_parquet(cfg.paths.processed / "question_entities.parquet")
    splits = pd.read_parquet(cfg.paths.processed / "splits.parquet")
    test_start = pd.Timestamp(json.loads((cfg.paths.processed / "split_manifest.json").read_text())["test_start"])
    emb = ids = None
    model = cfg.raw.get("retrieval", {}).get("embedding_model")
    if with_embeddings:
        emb, ids = load_or_compute(q, cfg.paths.artifacts, _snapshot(cfg), model)
    return RetrievalIndex(q, e, emb, ids, bm25_fit_before=test_start), splits, model


def cmd_embed(cfg: Config, args: argparse.Namespace) -> dict:
    import pandas as pd

    from ml.retrieval.embeddings import load_or_compute

    q = pd.read_parquet(cfg.paths.processed / "questions.parquet")
    model = cfg.raw.get("retrieval", {}).get("embedding_model")
    emb, _ = load_or_compute(q, cfg.paths.artifacts, _snapshot(cfg), model)
    return {"model": model, "n": int(emb.shape[0]), "dim": int(emb.shape[1])}


def cmd_clusters(cfg: Config, args: argparse.Namespace) -> dict:
    import pandas as pd

    from ml.preprocessing.clusters import run_clusters
    from ml.retrieval.embeddings import load_or_compute

    q = pd.read_parquet(cfg.paths.processed / "questions.parquet")
    model = cfg.raw.get("retrieval", {}).get("embedding_model")
    emb, ids = load_or_compute(q, cfg.paths.artifacts, _snapshot(cfg), model)
    meta = run_clusters(cfg.paths.processed, emb, ids, k=args.k, seed=args.seed)
    return {k: meta[k] for k in ("k", "n_fit", "top_terms")}


def cmd_retrieval_eval(cfg: Config, args: argparse.Namespace) -> dict:
    import pandas as pd

    from ml.retrieval.related import run_retrieval_eval

    index, splits, model = _retrieval_index(cfg, with_embeddings=not args.no_embeddings)
    links = pd.read_parquet(cfg.paths.processed / "question_links.parquet")
    out = cfg.paths.artifacts / "retrieval" / _snapshot(cfg)
    res = run_retrieval_eval(index, links, splits, out, {"snapshot_id": _snapshot(cfg), "embedding_model": model})
    t = res["table"]
    keep = t["metric"].isin(["mrr", "hits@1", "hits@10", "recall@10", "precision@10"])
    return {"manifest": res["manifest"], "results": t[keep].round(4).to_dict("records")}


def cmd_status(cfg: Config, args: argparse.Namespace) -> dict:
    ckpt = Checkpoint(cfg.paths.state / f"ingest_{cfg.source['site']}.json").state
    chunks = ckpt["chunks"]
    out = {
        "data_dir": str(cfg.paths.data_dir),
        "chunks_complete": sorted(k for k, v in chunks.items() if v["status"] == "complete"),
        "chunks_partial": {k: v["next_page"] for k, v in chunks.items() if v["status"] == "partial"},
        "update_watermark": ckpt["update"].get("watermark"),
        "last_run": ckpt["runs"][-1] if ckpt["runs"] else None,
    }
    manifest = cfg.paths.processed / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        out["processed"] = {k: m[k] for k in ("snapshot_id", "created_at", "created_at_range")}
        out["processed"]["n_questions"] = m["quality"]["n_questions"]
    return out


COMMANDS = {
    "ingest": (cmd_ingest, "Fetch questions by creation-date chunks (resumable)."),
    "update": (cmd_update, "Fetch questions with activity since the last watermark."),
    "preprocess": (cmd_preprocess, "Rebuild normalised tables from raw pages."),
    "split": (cmd_split, "Write the time-based train/val/test split."),
    "fetch-linked": (cmd_fetch_linked, "Fetch linked/duplicate-target questions missing from the corpus."),
    "anomalies": (cmd_anomalies, "Weekly anomaly detection with FDR control and signal ranking."),
    "anomaly-eval": (cmd_anomaly_eval, "Injection study and null calibration of the detectors."),
    "embed": (cmd_embed, "Compute (or load cached) sentence-transformer embeddings."),
    "clusters": (cmd_clusters, "Fit semantic clusters on train embeddings (used as issue groups)."),
    "retrieval-eval": (cmd_retrieval_eval, "Evaluate BM25 and embedding related-question retrieval."),
    "status": (cmd_status, "Show ingestion checkpoint and processed snapshot."),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aii", description=__doc__)
    p.add_argument("--config", help="path to pipeline.yaml")
    p.add_argument("--data-dir", help="override data directory")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        sp = sub.add_parser(name, help=help_text)
        if name in ("ingest", "update"):
            sp.add_argument("--start", help="research window start (ISO date, UTC)")
        if name in ("ingest", "update", "fetch-linked"):
            sp.add_argument("--max-requests", type=int, help="stop after this many API requests")
        if name == "ingest":
            sp.add_argument("--end", help="exclusive end (ISO date, UTC); default now")
        if name == "update":
            sp.add_argument("--since", help="override activity watermark (ISO datetime)")
        if name == "anomaly-eval":
            sp.add_argument("--trials", type=int, default=400)
            sp.add_argument("--seed", type=int, default=13)
        if name == "clusters":
            sp.add_argument("--k", type=int, default=40)
            sp.add_argument("--seed", type=int, default=0)
        if name == "retrieval-eval":
            sp.add_argument("--no-embeddings", action="store_true", help="BM25 only")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config, args.data_dir)
    result = COMMANDS[args.command][0](cfg, args)
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
