"""GraphSAGE vs BM25 / embedding / Jaccard for related-question prediction on Ask Different.

Protocol (docs/research_design.md section 5, docs/link_prediction.md):

1. Positives: duplicate-closure and body-link pairs, oriented query (later) -> target (earlier).
2. Train: pairs whose query is in the train split and whose label is known before the validation
   boundary. Message passing uses feature edges known before that boundary; label edges are never
   message-passing edges.
3. Model selection: small grid on validation MRR (weekly as-of graphs), seed 0, early stopping.
4. Test: the selected configuration retrained with 3 seeds; each test query is embedded with the
   graph exactly as of its creation time. All methods rank the same candidates (every earlier
   question) and score the same positive/negative pairs.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.evaluation.link_eval import classification_metrics, evaluate_methods, paired_mrr_difference
from ml.evaluation.ranking import summarize
from ml.models.graphsage_link import (
    Scorer,
    TrainConfig,
    build_training_pairs,
    sample_negatives,
    train_model,
)
from ml.models.link_data import NODE_TYPES, build_link_graph, entity_text, label_pairs
from ml.retrieval.embeddings import encode
from ml.retrieval.graph_heuristic import JaccardHeuristic
from ml.retrieval.related import RetrievalIndex, build_eval_set

log = logging.getLogger(__name__)

# Validation grid. Each hypothesis for "why might GraphSAGE trail the embedding baseline" has a
# variant; ``text_only_*`` entries (no message passing) are ablation controls and can never be
# selected as the GraphSAGE model.
GRID = [
    {"name": "sage2_mlp_bce", "decoder": "mlp", "layers": 2},
    {"name": "text_only_mlp_bce", "decoder": "mlp", "layers": 0},
    {"name": "sage2_mlp_listwise", "decoder": "mlp", "layers": 2, "loss": "listwise", "n_random": 32},
    {"name": "sage2_mlp_bce_residual", "decoder": "mlp", "layers": 2, "text_residual": True},
    {
        "name": "sage2_dot_listwise_residual",
        "decoder": "dot",
        "layers": 2,
        "loss": "listwise",
        "n_random": 32,
        "text_residual": True,
    },
    {
        "name": "sage1_dot_listwise_residual",
        "decoder": "dot",
        "layers": 1,
        "loss": "listwise",
        "n_random": 32,
        "text_residual": True,
    },
    {
        "name": "text_only_dot_listwise_residual",
        "decoder": "dot",
        "layers": 0,
        "loss": "listwise",
        "n_random": 32,
        "text_residual": True,
    },
]


GRID_KEYS = {"decoder", "layers", "loss", "n_random", "text_residual", "hidden", "lr", "dropout"}


def _params(g: dict) -> dict:
    return {k: v for k, v in g.items() if k != "name"}


def _entity_embeddings(nodes: pd.DataFrame, cache: Path, model_name: str) -> dict[str, np.ndarray]:
    ent = nodes.loc[nodes["node_type"].isin(NODE_TYPES) & (nodes["node_type"] != "question"), "node_id"].tolist()
    npy, ids_path = cache / "entity_nodes.npy", cache / "entity_nodes.ids.json"
    if npy.exists() and ids_path.exists() and json.loads(ids_path.read_text()) == ent:
        return dict(zip(ent, np.load(npy), strict=True))
    emb = encode([entity_text(n) for n in ent], model_name)
    np.save(npy, emb)
    ids_path.write_text(json.dumps(ent))
    return dict(zip(ent, emb, strict=True))


def _env() -> dict:
    import sentence_transformers
    import sklearn
    import torch_geometric

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _mlflow(tracking_dir: Path):
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        mlflow.set_experiment("ask-different-link-prediction")
        return mlflow
    except Exception as exc:  # experiment tracking is optional; results are also written as files
        log.warning("mlflow unavailable (%s); continuing with file outputs only", exc)
        return None


def run(
    processed: Path,
    artifacts: Path,
    snapshot: str,
    model_name: str,
    out_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    base: TrainConfig | None = None,
    grid: list[dict] | None = None,
) -> dict:
    t_start = time.time()
    base = base or TrainConfig()
    grid = GRID if grid is None else grid
    torch.set_num_threads(max(1, torch.get_num_threads()))
    q = pd.read_parquet(processed / "questions.parquet")
    ent = pd.read_parquet(processed / "question_entities.parquet")
    links = pd.read_parquet(processed / "question_links.parquet")
    splits = pd.read_parquet(processed / "splits.parquet")
    sm = json.loads((processed / "split_manifest.json").read_text())
    val_start, test_start = pd.Timestamp(sm["val_start"]), pd.Timestamp(sm["test_start"])
    gdir = artifacts / "graph" / snapshot
    nodes, edges = pd.read_parquet(gdir / "nodes.parquet"), pd.read_parquet(gdir / "edges.parquet")
    edir = artifacts / "embeddings" / snapshot
    slug = model_name.split("/")[-1]
    emb = np.load(edir / f"{slug}.npy")
    emb_ids = pd.read_parquet(edir / f"{slug}.ids.parquet")["question_id"].to_numpy()

    graph = build_link_graph(nodes, edges, emb, emb_ids, _entity_embeddings(nodes, edir, model_name))
    idx_val = RetrievalIndex(q, ent, emb, emb_ids, bm25_fit_before=val_start)  # train negatives / validation
    idx_test = RetrievalIndex(q, ent, emb, emb_ids, bm25_fit_before=test_start)  # test
    assert (idx_val.q["question_id"].to_numpy() == idx_test.q["question_id"].to_numpy()).all()

    pairs = label_pairs(links, q, splits)
    train_pairs = pairs[(pairs["split"] == "train") & (pairs["known_at"] < val_start)]
    val_set = build_eval_set(idx_val, links, splits, "val")
    test_set = build_eval_set(idx_test, links, splits, "test")
    cutoff_ns = int(val_start.value) - 1

    counts = {
        "graph_nodes": int(len(graph.node_ids)),
        "graph_feature_edges_directed": int(len(graph.edge_src)),
        "train_graph_edges_directed": int((graph.edge_time <= cutoff_ns).sum()),
        "label_pairs_total": int(len(pairs)),
        "train_positive_pairs": int(len(train_pairs)),
        "train_positive_duplicate": int((train_pairs["link_type"] == "duplicate").sum()),
        "val_queries": int(len(val_set.queries)),
        "test_queries": int(len(test_set.queries)),
        "test_queries_with_duplicate_label": int(
            test_set.queries["relevant_types"].map(lambda t: "duplicate" in t).sum()
        ),
        "test_positive_pairs": int(test_set.queries["relevant_pos"].map(len).sum()),
    }
    log.info("counts %s", counts)
    mlflow = _mlflow(artifacts.parent / "mlruns")
    ckpt_dir = artifacts / "models" / snapshot
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def fit(cfg: TrainConfig, tag: str):
        rows = build_training_pairs(idx_val, train_pairs, pairs, cfg, seed=cfg.seed)
        t0 = time.time()
        run_ctx = mlflow.start_run(run_name=tag) if mlflow else None
        try:
            if mlflow:
                mlflow.log_params(
                    {
                        **{k: v for k, v in asdict(cfg).items() if k != "extra"},
                        "snapshot": snapshot,
                        "embedding_model": model_name,
                        **{f"n_{k}": v for k, v in counts.items()},
                    }
                )

            def on_eval(epoch, loss, val_mrr):
                if mlflow:
                    mlflow.log_metrics({"train_loss": loss, "val_mrr_weekly": val_mrr}, step=epoch)

            model, hist = train_model(graph, idx_val, rows, cutoff_ns, val_set, cfg, on_eval)
            best_val = max(h.get("val_mrr", -1) for h in hist)
            if mlflow:
                mlflow.log_metric("best_val_mrr_weekly", best_val)
        finally:
            if run_ctx:
                mlflow.end_run()
        return model, hist, best_val, rows, time.time() - t0

    # 1. model selection on validation (seed 0)
    selection = []
    for g in grid:
        cfg = replace(base, **_params(g), seed=0)
        _, hist, best_val, rows, secs = fit(cfg, f"select-{g['name']}")
        selection.append(
            {
                **g,
                "best_val_mrr_weekly": best_val,
                "epochs_run": len(hist),
                "seconds": round(secs, 1),
                "train_rows": len(rows),
                "train_rows_by_kind": rows["kind"].value_counts().to_dict(),
            }
        )
        log.info("selection %s", selection[-1])
    eligible = [s for s in selection if s.get("layers", 2) >= 1]
    best_g = max(eligible, key=lambda s: s["best_val_mrr_weekly"])
    best_cfg = replace(base, **_params({k: v for k, v in best_g.items() if k in {*GRID_KEYS, "name"}}))

    # 2. shared test negatives (same pairs for every method)
    rng = np.random.default_rng(12345)
    cat = idx_test.q["primary_category"].fillna("").to_numpy()
    related: dict[int, set[int]] = {}
    for a, b in zip(pairs["query_id"], pairs["target_id"], strict=True):
        if a in idx_test.pos.index and b in idx_test.pos.index:
            pa, pb = int(idx_test.pos[a]), int(idx_test.pos[b])
            related.setdefault(pa, set()).add(pb)
            related.setdefault(pb, set()).add(pa)
    negatives = {}
    for r in test_set.queries.itertuples(index=False):
        rand, hard = sample_negatives(
            idx_test, r.query_pos, related.get(r.query_pos, set()) | {r.query_pos}, cat, rng, n_random=5, n_hard=5
        )
        negatives[r.query_pos] = rand + hard

    # 3. final models (3 seeds), strict per-query as-of embeddings
    jac = JaccardHeuristic(edges, idx_test.q["question_id"].to_numpy())
    scorers = {
        "bm25": lambda p: idx_test.scores("bm25", p)[:p],
        "embedding": lambda p: idx_test.scores("embedding", p)[:p],
        "jaccard_entities": jac.scores,
    }
    final_runs = []
    for s in seeds:
        cfg = replace(best_cfg, seed=s)
        model, hist, best_val, _, secs = fit(cfg, f"final-{cfg.decoder}-L{cfg.layers}-seed{s}")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(cfg),
                "snapshot": snapshot,
                "embedding_model": model_name,
                "val_start": str(val_start),
            },
            ckpt_dir / f"graphsage_seed{s}.pt",
        )
        scorer = Scorer(model, graph, idx_test)
        scorers[f"graphsage_seed{s}"] = scorer.scores  # default: as of the query's creation time
        final_runs.append(
            {
                "seed": s,
                "best_val_mrr_weekly": best_val,
                "epochs_run": len(hist),
                "seconds": round(secs, 1),
                "history": hist,
            }
        )

    # Ablation on test: the selected configuration without message passing (text features only).
    abl_cfg = replace(best_cfg, layers=0, seed=0)
    abl_model, _, abl_val, _, _ = fit(abl_cfg, "final-ablation-no-message-passing")
    scorers["no_message_passing_seed0"] = Scorer(abl_model, graph, idx_test).scores

    t0 = time.time()
    per_query, pair_scores = evaluate_methods(test_set, scorers, negatives)
    eval_secs = time.time() - t0
    seeds_cols = [f"graphsage_seed{s}" for s in seeds]
    mean_rr = per_query[per_query["method"].isin(seeds_cols)].groupby("query_id").mean(numeric_only=True).reset_index()
    per_query = pd.concat(
        [
            per_query,
            mean_rr.assign(
                method="graphsage_mean_over_seeds", has_duplicate_label=mean_rr["has_duplicate_label"].astype(bool)
            ),
        ]
    )

    ranking = pd.concat(
        [
            summarize(per_query).assign(stratum="all_queries"),
            summarize(per_query[per_query["has_duplicate_label"]]).assign(stratum="duplicate_label_queries"),
        ]
    )
    cls = pd.concat(
        [
            classification_metrics(pair_scores).assign(stratum="all_pairs"),
            classification_metrics(pair_scores[pair_scores["has_duplicate_label"]]).assign(
                stratum="duplicate_label_queries"
            ),
        ]
    )
    paired = []
    for stratum, pq in (
        ("all_queries", per_query),
        ("duplicate_label_queries", per_query[per_query["has_duplicate_label"]]),
    ):
        for b in ("embedding", "bm25", "jaccard_entities"):
            for a in [*seeds_cols, "graphsage_mean_over_seeds"]:
                paired.append({"stratum": stratum, **paired_mrr_difference(pq, a, b)})
    seed_table = (
        per_query[per_query["method"].isin(seeds_cols)]
        .groupby("method")[["rr", "hits@1", "hits@10", "recall@10"]]
        .mean()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_parquet(out_dir / "per_query.parquet", index=False)
    pair_scores.to_parquet(out_dir / "pair_scores.parquet", index=False)
    ranking.to_csv(out_dir / "ranking_summary.csv", index=False)
    cls.to_csv(out_dir / "classification_summary.csv", index=False)
    pd.DataFrame(paired).to_csv(out_dir / "paired_mrr_differences.csv", index=False)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot,
        "embedding_model": model_name,
        "val_start": str(val_start),
        "test_start": str(test_start),
        "counts": counts,
        "negatives_per_test_query": "5 random earlier + 5 hard (BM25 top-50, different primary category, cosine < 0.8)",
        "selection": selection,
        "selected_config": asdict(best_cfg),
        "final_runs": final_runs,
        "graphsage_seed_mrr": {
            "mean": float(seed_table["rr"].mean()),
            "std": float(seed_table["rr"].std(ddof=1)) if len(seeds) > 1 else 0.0,
            "per_seed": seed_table["rr"].to_dict(),
        },
        "evaluation_seconds": round(eval_secs, 1),
        "total_seconds": round(time.time() - t_start, 1),
        "environment": _env(),
        "seeds": list(seeds),
        "ablation_no_message_passing_val_mrr_weekly": abl_val,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    _plot(ranking, out_dir)
    return report


def _plot(ranking: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["bm25", "jaccard_entities", "embedding", "graphsage_mean_over_seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, stratum in zip(axes, ("all_queries", "duplicate_label_queries"), strict=True):
        d = ranking[(ranking["stratum"] == stratum) & (ranking["metric"] == "mrr")].set_index("method").loc[methods]
        ax.bar(
            range(len(methods)),
            d["mean"],
            yerr=[d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]],
            color=["#8da0cb", "#a6d854", "#66c2a5", "#fc8d62"],
            capsize=4,
        )
        ax.set_xticks(range(len(methods)), ["BM25", "Jaccard", "Embedding", "GraphSAGE"], rotation=0)
        ax.set_title(stratum.replace("_", " "))
        ax.set_ylabel("MRR (bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(out_dir / "mrr_comparison.png", dpi=150)
    plt.close(fig)
