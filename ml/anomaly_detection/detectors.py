"""Three anomaly detectors over weekly issue-group counts.

All detectors model a group's count ``y_t`` relative to the site-wide weekly total ``N_t`` so that
the steep decline of Ask Different's overall volume (and site-wide seasonality, e.g. release
weeks) is not mistaken for group-specific change. Baselines use strictly *past* weeks.

A. ``rolling_robust``: rolling median / MAD of the group's share ``y/N`` over the lookback window.
   The scale is floored by the binomial standard error so sparse groups do not produce huge
   z-scores from a zero MAD. p-values are a normal approximation (screening only).
B. ``negbin``: count model ``y_t ~ NegBin(mu_t = p_hat * N_t, alpha)`` where ``p_hat`` is the pooled
   share over the lookback window and ``alpha`` the method-of-moments overdispersion from the same
   window (alpha=0 reduces to Poisson). One-sided upper-tail p-values are exact under the model.
C. ``pelt_poisson``: offline PELT change-point detection with a Poisson-with-offset segment cost;
   detects persistent level shifts in share. Uses the whole series (retrospective), so it is used
   to *characterise* signals (spike vs sustained), not to raise alerts on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

MAD_SCALE = 1.4826


@dataclass
class DetectorConfig:
    lookback_weeks: int = 26
    min_history_weeks: int = 12  # weeks with N>0 inside the lookback window
    min_expected: float = 2.0  # below this the week is "insufficient data"
    chunk_size: int = 512


def _history_weeks(totals: np.ndarray, lookback: int) -> np.ndarray:
    """Number of weeks with N>0 in the window [t-lookback, t) for each t."""
    has = (totals > 0).astype(float)
    c = np.concatenate([[0.0], np.cumsum(has)])
    t = np.arange(len(totals))
    lo = np.maximum(t - lookback, 0)
    return c[t] - c[lo]


def _rolling_past_sum(x: np.ndarray, lookback: int) -> np.ndarray:
    """Sum of x over [t-lookback, t) along the last axis (exclusive of t)."""
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    t = np.arange(x.shape[-1])
    lo = np.maximum(t - lookback, 0)
    return c[..., t] - c[..., lo]


def rolling_robust(counts: pd.DataFrame, totals: pd.Series, cfg: DetectorConfig) -> dict[str, pd.DataFrame]:
    y = counts.to_numpy(dtype=float)
    n = totals.to_numpy(dtype=float)
    share = np.where(n > 0, y / np.where(n > 0, n, 1), np.nan)
    L = cfg.lookback_weeks
    G, T = y.shape
    med = np.full((G, T), np.nan)
    mad = np.full((G, T), np.nan)
    if T > L:
        for s in range(0, G, cfg.chunk_size):
            win = sliding_window_view(share[s : s + cfg.chunk_size], L, axis=1)[:, :-1, :]  # windows ending t-1
            m = np.nanmedian(win, axis=2)
            med[s : s + cfg.chunk_size, L:] = m
            mad[s : s + cfg.chunk_size, L:] = np.nanmedian(np.abs(win - m[..., None]), axis=2)
    hist = _history_weeks(n, L)
    n_safe = np.where(n > 0, n, np.nan)
    binom_se = np.sqrt(np.clip(med * (1 - med), 0, None) / n_safe)
    scale = np.fmax(np.fmax(MAD_SCALE * mad, binom_se), 1.0 / n_safe)
    expected = med * n
    z = (share - med) / scale
    valid = (hist >= cfg.min_history_weeks)[None, :] & (expected >= cfg.min_expected) & np.isfinite(z)
    p = np.where(valid, stats.norm.sf(z), np.nan)
    frame = lambda a: pd.DataFrame(a, index=counts.index, columns=counts.columns)  # noqa: E731
    return {"expected": frame(expected), "score": frame(np.where(valid, z, np.nan)), "p_value": frame(p)}


def negbin(counts: pd.DataFrame, totals: pd.Series, cfg: DetectorConfig) -> dict[str, pd.DataFrame]:
    y = counts.to_numpy(dtype=float)
    n = totals.to_numpy(dtype=float)
    L = cfg.lookback_weeks
    sy = _rolling_past_sum(y, L)
    sn = _rolling_past_sum(n[None, :], L)
    sy2 = _rolling_past_sum(y**2, L)
    syn = _rolling_past_sum(y * n[None, :], L)
    sn2 = _rolling_past_sum((n**2)[None, :], L)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_hat = np.where(sn > 0, sy / sn, np.nan)
        # Method-of-moments overdispersion: alpha = sum[(y - mu)^2 - y] / sum[mu^2], mu_s = p_hat * N_s
        ss_resid = sy2 - 2 * p_hat * syn + p_hat**2 * sn2
        alpha = np.clip((ss_resid - sy) / (p_hat**2 * sn2), 0.0, 10.0)
        mu = p_hat * n[None, :]
    hist = _history_weeks(n, L)
    valid = (hist >= cfg.min_history_weeks)[None, :] & (mu >= cfg.min_expected) & np.isfinite(mu)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)

    p = np.full(y.shape, np.nan)
    pois = valid & (alpha < 1e-8)
    p[pois] = stats.poisson.sf(y[pois] - 1, mu[pois])
    nb = valid & ~pois
    r = 1.0 / alpha[nb]
    p[nb] = stats.nbinom.sf(y[nb] - 1, r, r / (r + mu[nb]))
    # Standardised residual under the model, used as the detector's score.
    sd = np.sqrt(mu + alpha * mu**2)
    score = np.where(valid, (y - mu) / np.where(sd > 0, sd, np.nan), np.nan)
    frame = lambda a: pd.DataFrame(a, index=counts.index, columns=counts.columns)  # noqa: E731
    return {
        "expected": frame(np.where(np.isfinite(mu), mu, np.nan)),
        "score": frame(score),
        "p_value": frame(p),
        "alpha": frame(np.where(valid, alpha, np.nan)),
    }


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values; NaNs are ignored (kept as NaN) and not counted as tests."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return q
    pv = p[mask]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    q[mask] = out
    return q


def bh_per_week(p_values: pd.DataFrame) -> pd.DataFrame:
    """Apply BH across all groups tested in the same week (the screening family)."""
    q = np.column_stack([benjamini_hochberg(p_values[c].to_numpy()) for c in p_values.columns])
    return pd.DataFrame(q, index=p_values.index, columns=p_values.columns)


# ---------------------------------------------------------------- change points (PELT)


def _poisson_offset_cost(cy: np.ndarray, cn: np.ndarray, cylogn: np.ndarray, a: int, b: int) -> float:
    """Negative Poisson log-likelihood (dropping log y! terms) of segment [a, b) with rate lam*N."""
    ys, ns = cy[b] - cy[a], cn[b] - cn[a]
    if ys <= 0 or ns <= 0:
        return 0.0
    return -(ys * np.log(ys / ns) + (cylogn[b] - cylogn[a]) - ys)


def pelt_poisson(y: np.ndarray, n: np.ndarray, penalty: float | None = None, min_size: int = 4) -> list[int]:
    """PELT (Killick et al., 2012) with a Poisson-with-offset cost. Returns change indices (segment starts)."""
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    T = len(y)
    if 2 * min_size > T:
        return []
    beta = penalty if penalty is not None else 2.0 * np.log(T)
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cn = np.concatenate([[0.0], np.cumsum(n)])
    cylogn = np.concatenate([[0.0], np.cumsum(np.where(n > 0, y * np.log(np.where(n > 0, n, 1)), 0.0))])
    F = np.full(T + 1, np.inf)
    F[0] = -beta
    last = np.zeros(T + 1, dtype=int)
    candidates = [0]
    for t in range(min_size, T + 1):
        valid = [s for s in candidates if t - s >= min_size]
        if not valid:
            continue
        costs = np.array([F[s] + _poisson_offset_cost(cy, cn, cylogn, s, t) + beta for s in valid])
        i = int(np.argmin(costs))
        F[t], last[t] = costs[i], valid[i]
        # Pruning: drop s whose cost can never again be optimal.
        keep = [s for s, c in zip(valid, costs, strict=True) if c - beta <= F[t]]
        candidates = keep + [s for s in candidates if t - s < min_size] + [t - min_size + 1]
        candidates = sorted(set(candidates))
    cps, t = [], T
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def describe_change_points(y: np.ndarray, n: np.ndarray, cps: list[int]) -> list[dict[str, float | int]]:
    """Rate before/after each change point and a likelihood-ratio statistic for the split."""
    y = np.asarray(y, dtype=float)
    n = np.asarray(n, dtype=float)
    cy = np.concatenate([[0.0], np.cumsum(y)])
    cn = np.concatenate([[0.0], np.cumsum(n)])
    cylogn = np.concatenate([[0.0], np.cumsum(np.where(n > 0, y * np.log(np.where(n > 0, n, 1)), 0.0))])
    bounds = [0, *cps, len(y)]
    out = []
    for i, cp in enumerate(cps):
        a, b = bounds[i], bounds[i + 2]
        rate_before = (cy[cp] - cy[a]) / max(cn[cp] - cn[a], 1e-12)
        rate_after = (cy[b] - cy[cp]) / max(cn[b] - cn[cp], 1e-12)
        lr = 2 * (
            _poisson_offset_cost(cy, cn, cylogn, a, b)
            - _poisson_offset_cost(cy, cn, cylogn, a, cp)
            - _poisson_offset_cost(cy, cn, cylogn, cp, b)
        )
        out.append(
            {
                "index": cp,
                "segment_start": a,
                "segment_end": b,
                "rate_before": rate_before,
                "rate_after": rate_after,
                "rel_change": (rate_after - rate_before) / rate_before if rate_before > 0 else np.inf,
                "lr_stat": lr,
                # chi2(1) is only approximate after the change point was selected by search.
                "p_value_approx": float(stats.chi2.sf(max(lr, 0.0), 1)),
            }
        )
    return out
