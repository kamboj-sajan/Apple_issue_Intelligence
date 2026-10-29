"""Unified evaluation of related-question methods on identical queries, candidates and pairs.

* Ranking: for each query, every method scores the same candidate set (all earlier questions);
  metrics come from exact ranks (``ml.evaluation.ranking``).
* Classification: per query, its positives plus the same sampled negatives (random + hard) are
  scored by every method; ROC-AUC and average precision are pooled over pairs. Some "negatives" may
  be unlabelled true positives, so these are lower bounds.
* Uncertainty: bootstrap over queries; paired bootstrap for MRR differences between methods.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml.evaluation.ranking import query_metrics, ranks_of_relevant
from ml.retrieval.related import EvalSet

ScoreFn = Callable[[int], np.ndarray]  # query_pos -> scores over candidates [0, query_pos)


def evaluate_methods(
    es: EvalSet, scorers: dict[str, ScoreFn], negatives: dict[int, list[int]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_query, pairs = [], []
    for r in es.queries.itertuples(index=False):
        negs = [n for n in negatives.get(r.query_pos, []) if n not in r.relevant_pos]
        dup = "duplicate" in r.relevant_types
        for name, fn in scorers.items():
            s = np.asarray(fn(r.query_pos), dtype=float)
            ranks = ranks_of_relevant(s, r.relevant_pos)
            per_query.append(
                {
                    "method": name,
                    "query_id": r.query_id,
                    "query_pos": r.query_pos,
                    "has_duplicate_label": dup,
                    **query_metrics(ranks),
                }
            )
            for c in r.relevant_pos:
                pairs.append((name, r.query_id, c, 1, s[c], dup))
            for c in negs:
                pairs.append((name, r.query_id, c, 0, s[c], dup))
    return pd.DataFrame(per_query), pd.DataFrame(
        pairs, columns=["method", "query_id", "cand_pos", "label", "score", "has_duplicate_label"]
    )


def classification_metrics(pairs: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for method, g in pairs.groupby("method"):
        qids = g["query_id"].unique()
        by_q = {q: d for q, d in g.groupby("query_id")}
        auc, ap = roc_auc_score(g["label"], g["score"]), average_precision_score(g["label"], g["score"])
        boots_auc, boots_ap = [], []
        for _ in range(n_boot):
            sample = pd.concat([by_q[q] for q in rng.choice(qids, len(qids))])
            if sample["label"].nunique() == 2:
                boots_auc.append(roc_auc_score(sample["label"], sample["score"]))
                boots_ap.append(average_precision_score(sample["label"], sample["score"]))
        for metric, val, b in (("roc_auc", auc, boots_auc), ("average_precision", ap, boots_ap)):
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": val,
                    "ci_low": np.quantile(b, 0.025),
                    "ci_high": np.quantile(b, 0.975),
                    "n_pairs": len(g),
                    "positive_rate": g["label"].mean(),
                }
            )
    return pd.DataFrame(rows)


def paired_mrr_difference(per_query: pd.DataFrame, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap CI of mean(rr_a - rr_b) over the same queries."""
    pa = per_query[per_query["method"] == a].set_index("query_id")["rr"]
    pb = per_query[per_query["method"] == b].set_index("query_id")["rr"]
    d = (pa - pb.loc[pa.index]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {
        "a": a,
        "b": b,
        "mean_diff": float(d.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
        "n_queries": int(len(d)),
        "p_a_better": float((boots > 0).mean()),
    }
