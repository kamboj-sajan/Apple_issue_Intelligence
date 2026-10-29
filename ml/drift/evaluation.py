"""Controlled evaluation of drift alerts with the calibrated thresholds.

* **No drift** (false-alarm rate): a calibration month is split at random into two halves, which
  come from the same distribution by construction.
* **Injected drift** (power): the current sample of a month has a fraction ``f`` of its questions
  replaced by questions from a single semantic cluster, simulating a topic surge of known size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.drift.pipeline import DriftConfig, compare_windows


def _alerts(res: dict, thresholds: dict, cfg: DriftConfig) -> dict[str, bool]:
    return {m: bool(r["z"] > thresholds.get(m, np.inf) and r["p_value"] < cfg.alert_p) for m, r in res.items()}


def no_drift_false_alarms(
    feats: pd.DataFrame,
    emb: np.ndarray,
    thresholds: dict,
    calibration_end: pd.Period,
    cfg: DriftConfig,
    n_months: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    months = [m for m in sorted(feats["month"].unique()) if m < calibration_end]
    sizes = feats.groupby("month").size()
    months = [m for m in months if sizes[m] >= 2 * cfg.min_sample]
    rows = []
    for m in rng.choice(months, min(n_months, len(months)), replace=False):
        idx = rng.permutation(np.nonzero((feats["month"] == m).to_numpy())[0])
        half = len(idx) // 2
        res = compare_windows(idx[:half], idx[half : 2 * half], feats, emb, cfg, rng)
        rows += [
            {"month": str(m), "metric": k, "alert": a, "z": res[k]["z"]}
            for k, a in _alerts(res, thresholds, cfg).items()
        ]
    return pd.DataFrame(rows)


def injected_drift_power(
    feats: pd.DataFrame,
    emb: np.ndarray,
    thresholds: dict,
    cfg: DriftConfig,
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    trials: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 2)
    months = sorted(feats["month"].unique())
    cluster = feats["clusters"].map(lambda v: v[0] if v else None)
    rows = []
    for f in fractions:
        for t in range(trials):
            i = int(rng.integers(cfg.reference_windows, len(months)))
            ref_idx = np.nonzero(feats["month"].isin(months[i - cfg.reference_windows : i]).to_numpy())[0]
            cur_idx = np.nonzero((feats["month"] == months[i]).to_numpy())[0]
            n = min(len(ref_idx), len(cur_idx), cfg.max_sample)
            if n < cfg.min_sample:
                continue
            cur = rng.choice(cur_idx, n, replace=False)
            target = rng.choice(cluster.dropna().unique())
            pool = np.setdiff1d(np.nonzero((cluster == target).to_numpy())[0], np.concatenate([cur, ref_idx]))
            k = int(round(f * n))
            if len(pool) < k:
                continue
            cur = np.concatenate([cur[: n - k], rng.choice(pool, k, replace=False)])
            res = compare_windows(rng.choice(ref_idx, n, replace=False), cur, feats, emb, cfg, rng)
            rows += [
                {
                    "fraction": f,
                    "trial": t,
                    "month": str(months[i]),
                    "cluster": target,
                    "metric": m,
                    "alert": a,
                    "z": res[m]["z"],
                }
                for m, a in _alerts(res, thresholds, cfg).items()
            ]
    return pd.DataFrame(rows)
