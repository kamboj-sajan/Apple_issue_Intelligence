"""Distribution-shift metrics between a reference and a current window.

Embedding metrics
* ``centroid_cosine``: 1 - cos(mean_ref, mean_cur). Cheap and interpretable, but blind to changes
  that keep the mean fixed (e.g. a topic splitting into two).
* ``mmd_rbf``: unbiased squared maximum mean discrepancy with an RBF kernel whose bandwidth is the
  median pairwise distance in the pooled sample. Sensitive to any distributional change, but
  harder to interpret, and its scale depends on sample size, hence the permutation null below.

Categorical metrics
* ``jsd``: Jensen-Shannon divergence (base 2, in [0, 1]) between frequency distributions of tags,
  categories, OS versions, devices or vocabulary. Rare values are pooled into ``__other__`` so small
  samples do not inflate the divergence through unseen long-tail values.

All raw values grow with sampling noise when windows are small. ``permutation_test`` compares the
observed value with values obtained after shuffling window membership, giving a standardised score
and a p-value that account for sample size.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np


def centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    return float(1.0 - ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum((x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2 * x @ y.T, 0.0)


def mmd_rbf(a: np.ndarray, b: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased MMD^2 estimate with an RBF kernel exp(-gamma * ||x - y||^2)."""
    if gamma is None:
        z = np.vstack([a, b])
        d = _sq_dists(z, z)
        med = np.median(d[np.triu_indices_from(d, k=1)])
        gamma = 1.0 / max(med, 1e-12)
    kaa, kbb, kab = np.exp(-gamma * _sq_dists(a, a)), np.exp(-gamma * _sq_dists(b, b)), np.exp(-gamma * _sq_dists(a, b))
    m, n = len(a), len(b)
    return float(
        (kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (n * (n - 1)) - 2 * kab.mean()
    )


def distribution(items: Iterable[Iterable[str]], vocab: list[str]) -> np.ndarray:
    """Normalised frequency over ``vocab`` + one ``__other__`` bucket (multi-valued items allowed)."""
    idx = {v: i for i, v in enumerate(vocab)}
    counts = np.zeros(len(vocab) + 1)
    for values in items:
        for v in values:
            counts[idx.get(v, len(vocab))] += 1
    total = counts.sum()
    return counts / total if total else counts


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint support)."""
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def shared_vocab(ref: Iterable[Iterable[str]], top: int = 200) -> list[str]:
    """Most frequent values in the *reference* window (the current window must not define the support)."""
    c = Counter(v for values in ref for v in values)
    return [v for v, _ in c.most_common(top)]


def permutation_test(
    ref: list, cur: list, stat: Callable[[list, list], float], n_perm: int = 100, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """Observed statistic vs. its distribution under shuffled window labels.

    Works for arrays (embeddings) and lists (multi-valued categorical items). Returns the observed
    value, null mean/sd, a standardised score ``z`` and a one-sided permutation p-value.
    """
    rng = rng or np.random.default_rng(0)
    obs = stat(ref, cur)
    pooled = list(ref) + list(cur) if not isinstance(ref, np.ndarray) else np.vstack([ref, cur])
    n = len(ref)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        if isinstance(pooled, np.ndarray):
            null[i] = stat(pooled[perm[:n]], pooled[perm[n:]])
        else:
            null[i] = stat([pooled[j] for j in perm[:n]], [pooled[j] for j in perm[n:]])
    sd = null.std(ddof=1)
    return {
        "value": float(obs),
        "null_mean": float(null.mean()),
        "null_sd": float(sd),
        "z": float((obs - null.mean()) / sd) if sd > 0 else 0.0,
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
    }
