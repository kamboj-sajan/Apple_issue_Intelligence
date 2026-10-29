"""Evaluation of the detectors.

1. **Injection study** (power): known spikes (1 week) and sustained shifts (``shift_weeks``) are
   injected into *real* group series. The injected extra questions are also added to the site
   total. A spike counts as detected if the injected week is flagged; a shift if any of its first
   ``follow_up`` weeks is flagged. Negbin flags use the same per-week BH family as production (the
   injected row replaces the original row among all groups tested that week).
2. **Null calibration** (false-positive rate): series are simulated from each group's own fitted
   share with negative-binomial noise and *no* anomaly; we report the fraction of weeks with
   uncorrected p < level. A calibrated detector should be close to ``level``.

Everything is seeded; results are reproducible for a given snapshot and config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.anomaly_detection import detectors as D
from ml.anomaly_detection.pipeline import AnomalyConfig, classify_signal


@dataclass
class InjectionConfig:
    n_trials: int = 300
    spike_factors: tuple[float, ...] = (2.0, 3.0, 5.0)
    shift_factors: tuple[float, ...] = (1.5, 2.0, 3.0)
    shift_weeks: int = 8
    min_group_mean: float = 2.0  # mean weekly count needed for a group to be eligible
    seed: int = 13


def _row(values: np.ndarray, columns) -> pd.DataFrame:
    return pd.DataFrame(values[None, :], index=pd.MultiIndex.from_tuples([("x", "x")]), columns=columns)


def _flags_for_row(
    y: np.ndarray, n: np.ndarray, other_p: np.ndarray, columns, cfg: AnomalyConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Negbin flags (BH with the other groups' p-values per week) and rolling flags for one series."""
    counts, totals = _row(y, columns), pd.Series(n, index=columns)
    B = D.negbin(counts, totals, cfg.detector)
    A = D.rolling_robust(counts, totals, cfg.detector)
    p = B["p_value"].to_numpy()[0]
    mu = B["expected"].to_numpy()[0]
    q = np.full_like(p, np.nan)
    for t in np.nonzero(np.isfinite(p))[0]:
        fam = np.concatenate([[p[t]], other_p[:, t]])
        q[t] = D.benjamini_hochberg(fam)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (y - mu) / mu
    flag_b = (q <= cfg.fdr_q) & (y >= cfg.min_observed) & (rel >= cfg.min_rel_change)
    flag_a = (A["score"].to_numpy()[0] >= cfg.rolling_z) & (y >= cfg.min_observed)
    return flag_b, flag_a


def injection_study(counts: pd.DataFrame, totals: pd.Series, cfg: AnomalyConfig, inj: InjectionConfig) -> pd.DataFrame:
    rng = np.random.default_rng(inj.seed)
    y_all = counts.to_numpy(dtype=float)
    n0 = totals.to_numpy(dtype=float)
    T = y_all.shape[1]
    L = cfg.detector.lookback_weeks
    base_p = D.negbin(counts, totals, cfg.detector)["p_value"].to_numpy()
    eligible = np.nonzero(y_all.mean(axis=1) >= inj.min_group_mean)[0]
    if len(eligible) == 0:
        raise ValueError("no eligible groups for injection")
    lo, hi = L + cfg.detector.min_history_weeks, T - inj.shift_weeks - cfg.follow_up_weeks
    if hi <= lo:
        raise ValueError("series too short for injection study")

    rows = []
    for trial in range(inj.n_trials):
        g = int(rng.choice(eligible))
        t = int(rng.integers(lo, hi))
        kind = "spike" if trial % 2 == 0 else "shift"
        factor = float(rng.choice(inj.spike_factors if kind == "spike" else inj.shift_factors))
        y, n = y_all[g].copy(), n0.copy()
        base = max(y[t - L : t].mean(), 1.0)
        span = range(t, t + 1) if kind == "spike" else range(t, t + inj.shift_weeks)
        for s in span:
            extra = rng.poisson((factor - 1.0) * base)
            y[s] += extra
            n[s] += extra
        other = np.delete(base_p, g, axis=0)
        flag_b, flag_a = _flags_for_row(y, n, other, counts.columns, cfg)
        window = [t] if kind == "spike" else list(range(t, t + cfg.follow_up_weeks + 1))
        hit_b = [s for s in window if flag_b[s]]
        hit_a = [s for s in window if flag_a[s]]
        cps = D.describe_change_points(y, n, D.pelt_poisson(y, n, min_size=cfg.cp_min_size))
        cp_hit = any(abs(c["index"] - t) <= 2 and c["rel_change"] > 0 for c in cps)
        predicted = classify_signal(hit_b[0], T, cps, cfg) if hit_b else None
        rows.append(
            {
                "trial": trial,
                "kind": kind,
                "factor": factor,
                "group_type": counts.index[g][0],
                "group_value": counts.index[g][1],
                "week_index": t,
                "baseline_mean": base,
                "negbin_detected": bool(hit_b),
                "negbin_delay": (hit_b[0] - t) if hit_b else np.nan,
                "rolling_detected": bool(hit_a),
                "rolling_delay": (hit_a[0] - t) if hit_a else np.nan,
                "changepoint_detected": cp_hit,
                "predicted_type": predicted,
                "type_correct": (predicted == ("spike" if kind == "spike" else "sustained")) if predicted else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_injections(res: pd.DataFrame) -> pd.DataFrame:
    agg = res.groupby(["kind", "factor"]).agg(
        trials=("trial", "size"),
        negbin_recall=("negbin_detected", "mean"),
        rolling_recall=("rolling_detected", "mean"),
        changepoint_recall=("changepoint_detected", "mean"),
        negbin_median_delay=("negbin_delay", "median"),
        type_accuracy=("type_correct", "mean"),
    )
    return agg.reset_index()


def null_calibration(
    counts: pd.DataFrame, totals: pd.Series, cfg: AnomalyConfig, level: float = 0.05, n_series: int = 200, seed: int = 7
) -> pd.DataFrame:
    """Simulate anomaly-free series from each group's overall share with NB noise; report the
    fraction of tested weeks with p < level per detector, by expected-count band."""
    rng = np.random.default_rng(seed)
    y_all = counts.to_numpy(dtype=float)
    n = totals.to_numpy(dtype=float)
    eligible = np.nonzero(y_all.sum(axis=1) >= 50)[0]
    gs = rng.choice(eligible, size=min(n_series, len(eligible)), replace=False)
    alpha_hat = np.nanmedian(D.negbin(counts.iloc[gs], totals, cfg.detector)["alpha"].to_numpy())
    alpha_hat = float(alpha_hat) if np.isfinite(alpha_hat) else 0.0
    sims = []
    for g in gs:
        share = y_all[g].sum() / n.sum()
        mu = share * n
        if alpha_hat > 1e-8:
            r = 1.0 / alpha_hat
            sims.append(rng.negative_binomial(r, r / (r + mu)))
        else:
            sims.append(rng.poisson(mu))
    sim = pd.DataFrame(np.array(sims, dtype=float), columns=counts.columns)
    B = D.negbin(sim, totals, cfg.detector)
    A = D.rolling_robust(sim, totals, cfg.detector)
    exp = B["expected"].to_numpy()
    bands = pd.cut(exp.ravel(), [0, 5, 20, np.inf], labels=["2-5", "5-20", ">20"])
    out = []
    for name, p in (("negbin", B["p_value"].to_numpy()), ("rolling_robust", A["p_value"].to_numpy())):
        pv = p.ravel()
        df = pd.DataFrame({"band": bands, "p": pv}).dropna()
        for band, grp in df.groupby("band", observed=True):
            out.append(
                {"method": name, "expected_band": band, "tests": len(grp), "fpr": float((grp["p"] < level).mean())}
            )
        out.append({"method": name, "expected_band": "all", "tests": len(df), "fpr": float((df["p"] < level).mean())})
    return pd.DataFrame(out).assign(level=level, simulated_alpha=alpha_hat)
