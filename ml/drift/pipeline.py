"""Time-aware semantic and categorical drift monitoring (separate from volume anomaly detection).

For every month ``w`` with enough data, the *current* sample (questions posted in ``w``) is compared
with a *reference* sample pooled from the ``reference_windows`` preceding months. Both are subsampled
to the same size, so sample size does not change the metrics' scale, and each metric is turned into a
permutation z-score.

Alert rule: z above the metric's threshold, permutation p < ``alert_p``, and both samples at least
``min_sample``. Each metric's threshold is the ``alert_quantile`` of its z-scores over *calibration*
months (all months before the validation boundary). An alert therefore means "more drift than
usual by this project's history", not "any drift at all".

Also produced: cluster-share movements, the share of *novel* questions (far from every cluster centroid
fit on the training period), candidate new clusters among novel questions, and new vocabulary.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering

from ml.data.raw_store import atomic_write_text
from ml.drift.metrics import centroid_cosine, distribution, jsd, mmd_rbf, permutation_test, shared_vocab
from ml.preprocessing.clusters import describe_clusters
from ml.retrieval.bm25 import tokenize

CATEGORICAL = ("tags", "categories", "os", "devices", "clusters", "vocabulary")


@dataclass
class DriftConfig:
    reference_windows: int = 3
    max_sample: int = 300
    min_sample: int = 50
    n_perm: int = 100
    alert_quantile: float = 0.95
    alert_p: float = 0.01
    top_vocab: int = 200
    novelty_quantile: float = 0.05
    novel_min_group: int = 5
    novel_distance: float | None = None  # None: calibrate from training clusters (see cluster_tightness)
    novel_tightness_quantile: float = 0.1
    new_term_min_docs: int = 5
    new_term_lookback: int = 12
    seed: int = 0


def question_features(questions: pd.DataFrame, entities: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    q = questions[questions["in_research_window"]].copy()
    ent = entities[entities["question_id"].isin(q["question_id"])]
    by = lambda t: ent[ent["entity_type"] == t].groupby("question_id")["value"].apply(list)  # noqa: E731
    q["os"] = q["question_id"].map(by("os")).apply(lambda v: v if isinstance(v, list) else [])
    q["devices"] = (
        q["question_id"]
        .map(by("device"))
        .apply(lambda v: [x for x in v if ":" not in x] if isinstance(v, list) else [])
    )
    q["clusters"] = q["question_id"].map(clusters).apply(lambda c: [f"cluster:{int(c)}"] if pd.notna(c) else [])
    q["vocabulary"] = [sorted(set(tokenize(t))) for t in q["title"]]
    q["month"] = q["created_at"].dt.tz_localize(None).dt.to_period("M")
    return q.sort_values("created_at").reset_index(drop=True)


def _categorical_stat(top: int):
    def stat(ref: list, cur: list) -> float:
        vocab = shared_vocab(ref, top)
        return jsd(distribution(ref, vocab), distribution(cur, vocab))

    return stat


def compare_windows(
    ref_idx: np.ndarray, cur_idx: np.ndarray, feats: pd.DataFrame, emb: np.ndarray, cfg: DriftConfig, rng
) -> dict[str, dict[str, float]]:
    n = min(len(ref_idx), len(cur_idx), cfg.max_sample)
    r = rng.choice(ref_idx, n, replace=False)
    c = rng.choice(cur_idx, n, replace=False)
    out = {
        "centroid_cosine": permutation_test(emb[r], emb[c], centroid_cosine, cfg.n_perm, rng),
        "mmd_rbf": permutation_test(emb[r], emb[c], mmd_rbf, cfg.n_perm, rng),
    }
    for col in CATEGORICAL:
        vals = feats[col].to_numpy()
        out[col] = permutation_test(list(vals[r]), list(vals[c]), _categorical_stat(cfg.top_vocab), cfg.n_perm, rng)
    for v in out.values():
        v["n"] = n
    return out


def drift_series(feats: pd.DataFrame, emb: np.ndarray, cfg: DriftConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    months = sorted(feats["month"].unique())
    rows = []
    for i, m in enumerate(months):
        if i < cfg.reference_windows:
            continue
        ref_months = months[i - cfg.reference_windows : i]
        ref_idx = np.nonzero(feats["month"].isin(ref_months).to_numpy())[0]
        cur_idx = np.nonzero((feats["month"] == m).to_numpy())[0]
        base = {
            "month": str(m),
            "n_current": len(cur_idx),
            "n_reference": len(ref_idx),
            "reference": f"{ref_months[0]}..{ref_months[-1]}",
        }
        if min(len(ref_idx), len(cur_idx)) < cfg.min_sample:
            rows.append({**base, "metric": "all", "status": "insufficient_data"})
            continue
        for metric, res in compare_windows(ref_idx, cur_idx, feats, emb, cfg, rng).items():
            rows.append({**base, "metric": metric, "status": "ok", **res})
    return pd.DataFrame(rows)


def calibrate_and_alert(
    series: pd.DataFrame, calibration_end: pd.Period, cfg: DriftConfig
) -> tuple[pd.DataFrame, dict]:
    ok = series[series["status"] == "ok"].copy()
    ok["calibration"] = ok["month"].map(pd.Period) < calibration_end
    thresholds = ok[ok["calibration"]].groupby("metric")["z"].quantile(cfg.alert_quantile).to_dict()
    ok["threshold"] = ok["metric"].map(thresholds)
    ok["alert"] = (ok["z"] > ok["threshold"]) & (ok["p_value"] < cfg.alert_p) & (ok["n"] >= cfg.min_sample)
    out = pd.concat([ok, series[series["status"] != "ok"]], ignore_index=True).sort_values(["month", "metric"])
    # Rows without enough data have no flags; keep the columns boolean (object dtype breaks ``~``).
    for col in ("calibration", "alert"):
        out[col] = out[col].astype("boolean").fillna(False).astype(bool)
    return out, {k: float(v) for k, v in thresholds.items()}


def cluster_movements(feats: pd.DataFrame, cfg: DriftConfig, top_terms: dict, min_count: int = 5) -> pd.DataFrame:
    """Per month, clusters whose share moved most relative to the reference window."""
    months = sorted(feats["month"].unique())
    c = feats.explode("clusters").dropna(subset=["clusters"])
    share = c.groupby(["month", "clusters"]).size().unstack(fill_value=0)
    share = share.reindex(months, fill_value=0)
    totals = feats.groupby("month").size().reindex(months, fill_value=0)
    rows = []
    for i in range(cfg.reference_windows, len(months)):
        cur, ref = share.iloc[i], share.iloc[i - cfg.reference_windows : i].sum()
        n_cur, n_ref = totals.iloc[i], totals.iloc[i - cfg.reference_windows : i].sum()
        if n_cur < cfg.min_sample:
            continue
        for cl in share.columns:
            sc, sr = cur[cl] / n_cur, ref[cl] / max(n_ref, 1)
            if max(cur[cl], ref[cl]) < min_count:
                continue
            ratio = (sc + 1e-3) / (sr + 1e-3)
            rows.append(
                {
                    "month": str(months[i]),
                    "cluster": cl,
                    "count": int(cur[cl]),
                    "share": sc,
                    "reference_share": sr,
                    "share_ratio": ratio,
                    "top_terms": top_terms.get(cl.split(":")[1], [])[:6],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["direction"] = np.where(df["share_ratio"] >= 1, "growing", "shrinking")
    return df.sort_values(["month", "share_ratio"], ascending=[True, False])


def cluster_tightness(emb: np.ndarray, labels: np.ndarray, quantile: float, rng, cap: int = 150) -> float:
    """``quantile`` of the per-cluster mean pairwise cosine distance among training clusters. A group of
    novel questions only counts as a candidate new cluster if it is at least this coherent."""
    per_cluster = []
    for k in np.unique(labels):
        idx = np.nonzero(labels == k)[0]
        idx = rng.choice(idx, min(cap, len(idx)), replace=False)
        if len(idx) < 2:
            continue
        sim = emb[idx] @ emb[idx].T
        per_cluster.append(float(np.mean(1 - sim[np.triu_indices(len(idx), 1)])))
    return float(np.quantile(per_cluster, quantile))


def novelty(
    feats: pd.DataFrame, emb: np.ndarray, centroids: np.ndarray, calibration_mask: np.ndarray, cfg: DriftConfig
):
    """Share of questions far from every training-period centroid, and candidate new clusters."""
    cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    nearest = (emb @ cn.T).max(axis=1)
    threshold = float(np.quantile(nearest[calibration_mask], cfg.novelty_quantile))
    novel = nearest < threshold
    rate = (
        pd.DataFrame({"month": feats["month"].astype(str), "novel": novel})
        .groupby("month")["novel"]
        .agg(["mean", "sum", "size"])
    )
    rate = rate.rename(columns={"mean": "novel_rate", "sum": "n_novel", "size": "n"}).reset_index()

    # Candidate new clusters: group novel questions per quarter (monthly counts are too small).
    quarters = feats["created_at"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    cands = []
    for qtr in sorted(quarters[novel].unique()):
        idx = np.nonzero(novel & (quarters == qtr).to_numpy())[0]
        if len(idx) < cfg.novel_min_group:
            continue
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=cfg.novel_distance
        ).fit_predict(emb[idx])
        sizes = Counter(labels)
        big = [lab for lab, s in sizes.items() if s >= cfg.novel_min_group]
        if not big:
            continue
        sub = labels[np.isin(labels, big)]
        terms = describe_clusters(feats["title"].iloc[idx[np.isin(labels, big)]].reset_index(drop=True), sub, top=6)
        for lab in big:
            members = idx[labels == lab]
            cands.append(
                {
                    "quarter": qtr,
                    "size": int(sizes[lab]),
                    "top_terms": terms.get(int(lab), []),
                    "example_question_ids": feats["question_id"].iloc[members[:5]].astype(int).tolist(),
                }
            )
    return rate, pd.DataFrame(cands), threshold


def new_terms(feats: pd.DataFrame, cfg: DriftConfig) -> pd.DataFrame:
    """Title terms that appear in >= ``new_term_min_docs`` questions in a month but were (almost) absent
    over the preceding ``new_term_lookback`` months."""
    months = sorted(feats["month"].unique())
    docs = {m: Counter(t for v in feats.loc[feats["month"] == m, "vocabulary"] for t in v) for m in months}
    rows = []
    for i, m in enumerate(months):
        if i < cfg.new_term_lookback:
            continue
        past = Counter()
        for pm in months[i - cfg.new_term_lookback : i]:
            past.update(docs[pm])
        for term, n in docs[m].items():
            if n >= cfg.new_term_min_docs and past[term] <= 1 and not term.isdigit():
                rows.append({"month": str(m), "term": term, "docs": n, "docs_previous_12_months": past[term]})
    return pd.DataFrame(rows).sort_values(["month", "docs"], ascending=[True, False]) if rows else pd.DataFrame()


def volume_drift_correlation(alerts: pd.DataFrame, signals: pd.DataFrame | None) -> dict[str, Any]:
    """H5: are semantic drift and volume anomalies the same signal? Spearman over months."""
    if signals is None or signals.empty:
        return {}
    mmd = alerts[(alerts["metric"] == "mmd_rbf") & (alerts["status"] == "ok")].set_index("month")["z"]
    vol = (
        signals.assign(month=signals["week_start"].dt.tz_localize(None).dt.to_period("M").astype(str))
        .groupby("month")
        .size()
    )
    vol = vol.reindex(mmd.index, fill_value=0)
    rho, p = stats.spearmanr(mmd.to_numpy(), vol.to_numpy())
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_months": int(len(mmd)),
        "months_with_volume_signals": int((vol > 0).sum()),
    }


def run_drift(
    processed: Path,
    emb: np.ndarray,
    emb_ids: np.ndarray,
    out_dir: Path,
    cfg: DriftConfig,
    signals: pd.DataFrame | None = None,
) -> dict[str, Any]:
    questions = pd.read_parquet(processed / "questions.parquet")
    entities = pd.read_parquet(processed / "question_entities.parquet")
    clusters = pd.read_parquet(processed / "clusters.parquet").set_index("question_id")["cluster"]
    meta = json.loads((processed / "clusters_meta.json").read_text())
    centroids = np.load(processed / "cluster_centroids.npy")
    sm = json.loads((processed / "split_manifest.json").read_text())
    calibration_end = pd.Timestamp(sm["val_start"]).tz_localize(None).to_period("M")

    feats = question_features(questions, entities, clusters)
    pos = pd.Series(np.arange(len(emb_ids)), index=emb_ids)
    E = emb[pos.loc[feats["question_id"]].to_numpy()]

    series = drift_series(feats, E, cfg)
    alerts, thresholds = calibrate_and_alert(series, calibration_end, cfg)
    moves = cluster_movements(feats, cfg, meta["top_terms"])
    calib_mask = (feats["month"] < calibration_end).to_numpy()
    if cfg.novel_distance is None:
        labels = feats["clusters"].map(lambda v: int(v[0].split(":")[1]) if v else -1).to_numpy()
        m = calib_mask & (labels >= 0)
        cfg = replace(
            cfg,
            novel_distance=cluster_tightness(
                E[m], labels[m], cfg.novel_tightness_quantile, np.random.default_rng(cfg.seed)
            ),
        )
    rate, candidates, nov_threshold = novelty(feats, E, centroids, calib_mask, cfg)
    terms = new_terms(feats, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(out_dir / "drift_metrics.parquet", index=False)
    moves.to_parquet(out_dir / "cluster_movements.parquet", index=False)
    rate.to_parquet(out_dir / "novelty_rate.parquet", index=False)
    candidates.to_json(out_dir / "new_cluster_candidates.json", orient="records", indent=2)
    terms.to_csv(out_dir / "new_terms.csv", index=False)
    ok = alerts[alerts["status"] == "ok"]
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": asdict(cfg),
        "calibration_end": str(calibration_end),
        "thresholds_z": thresholds,
        "months_evaluated": int(ok["month"].nunique()),
        "months_insufficient": int((alerts["status"] == "insufficient_data").sum()),
        "alerts_by_metric": ok.groupby("metric")["alert"].sum().astype(int).to_dict(),
        "alert_rate_calibration": float(ok[ok["calibration"]]["alert"].mean()),
        "alert_rate_after_calibration": float(ok[~ok["calibration"]]["alert"].mean()),
        "novelty_threshold_cosine": nov_threshold,
        "n_new_cluster_candidates": int(len(candidates)),
        "volume_vs_semantic_drift": volume_drift_correlation(alerts, signals),
    }
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, default=str))
    return summary
