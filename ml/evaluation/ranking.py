"""Ranking metrics computed from full candidate score vectors.

Ranks are exact (not truncated at K): the rank of a relevant item is
``1 + #(candidates scoring higher) + #(other candidates tied) / 2`` (ties split evenly, so a method
that returns constant scores is not rewarded).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def ranks_of_relevant(scores: np.ndarray, relevant_idx: Sequence[int]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    out = []
    for i in relevant_idx:
        s = scores[i]
        greater = np.sum(scores > s)
        ties = np.sum(scores == s) - 1
        out.append(1 + greater + ties / 2)
    return np.asarray(out, dtype=float)


def query_metrics(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Metrics for one query given the ranks of all its relevant candidates."""
    m: dict[str, float] = {"rr": 1.0 / ranks.min(), "n_relevant": float(len(ranks)), "best_rank": float(ranks.min())}
    for k in ks:
        hit = ranks <= k
        m[f"hits@{k}"] = float(hit.any())
        m[f"recall@{k}"] = float(hit.mean())
        m[f"precision@{k}"] = float(hit.sum() / k)
    return m


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(per_query: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean and bootstrap 95% CI of each metric column, per method."""
    metric_cols = [c for c in per_query.columns if c == "rr" or "@" in c]
    rows = []
    for method, g in per_query.groupby("method"):
        for c in metric_cols:
            lo, hi = bootstrap_ci(g[c].to_numpy(), n_boot, seed)
            rows.append(
                {
                    "method": method,
                    "metric": "mrr" if c == "rr" else c,
                    "mean": g[c].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_queries": len(g),
                }
            )
    return pd.DataFrame(rows)
