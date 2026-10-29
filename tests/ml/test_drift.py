"""Drift metric and pipeline tests on SYNTHETIC data with known shifts."""

import numpy as np
import pandas as pd
import pytest

from ml.drift.evaluation import injected_drift_power, no_drift_false_alarms
from ml.drift.metrics import centroid_cosine, distribution, jsd, mmd_rbf, permutation_test, shared_vocab
from ml.drift.pipeline import DriftConfig, calibrate_and_alert, cluster_movements, drift_series, new_terms


def test_jsd_bounds_and_other_bucket():
    p = distribution([["a"], ["b"]], ["a", "b"])
    assert jsd(p, p) == pytest.approx(0.0)
    assert jsd(distribution([["a"]], ["a", "b"]), distribution([["b"]], ["a", "b"])) == pytest.approx(1.0)
    d = distribution([["a", "zzz"]], ["a"])
    assert d.tolist() == [0.5, 0.5]  # unseen value pooled into __other__
    assert shared_vocab([["x", "y"], ["x"]], top=1) == ["x"]


def test_embedding_metrics_detect_shift_not_noise():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 8)), rng.normal(size=(200, 8))
    shifted = rng.normal(size=(200, 8)) + np.r_[1.5, np.zeros(7)]
    null, shift = abs(mmd_rbf(a, b)), mmd_rbf(a, shifted)
    assert null < 0.02
    assert shift > 10 * null and shift > 0.05  # MMD scale depends on bandwidth/dimension; compare to null
    assert centroid_cosine(a + 3, shifted + 3) > centroid_cosine(a + 3, b + 3)


def test_permutation_test_calibrated_under_null_and_powerful_under_shift():
    rng = np.random.default_rng(1)
    ps = [
        permutation_test(rng.normal(size=(60, 4)), rng.normal(size=(60, 4)), mmd_rbf, 50, rng)["p_value"]
        for _ in range(30)
    ]
    assert 0.2 < np.mean(ps) < 0.8  # roughly uniform under the null
    res = permutation_test(rng.normal(size=(60, 4)), rng.normal(size=(60, 4)) + 1.0, mmd_rbf, 50, rng)
    assert res["p_value"] < 0.05 and res["z"] > 3


def synthetic_feats(months=10, per_month=120, drift_month=8):
    """Monthly questions; from ``drift_month`` on, 40 % come from a new topic. SYNTHETIC."""
    rng = np.random.default_rng(2)
    rows, emb = [], []
    base = rng.normal(size=(3, 16))
    new_topic = rng.normal(size=16) * 2
    for m in range(months):
        for i in range(per_month):
            novel = m >= drift_month and i < 0.4 * per_month
            topic = 3 if novel else int(rng.integers(0, 3))
            v = (new_topic if novel else base[topic]) + 0.3 * rng.normal(size=16)
            emb.append(v / np.linalg.norm(v))
            rows.append(
                {
                    "question_id": len(rows),
                    "month": pd.Period("2022-01", "M") + m,
                    "created_at": pd.Timestamp("2022-01-01", tz="UTC")
                    + pd.DateOffset(months=m)
                    + pd.Timedelta(hours=i),
                    "title": "sequoia upgrade broke wifi"
                    if novel
                    else ["printer jam", "battery drain", "dock icon"][topic],
                    "tags": ["sequoia"] if novel else [["printing"], ["battery"], ["dock"]][topic],
                    "categories": ["update-install"] if novel else [["other"], ["battery-power"], ["other"]][topic],
                    "os": ["macos:15"] if novel else [],
                    "devices": [],
                    "clusters": [f"cluster:{topic}"],
                    "vocabulary": ["sequoia", "upgrade", "wifi"]
                    if novel
                    else [["printer"], ["battery"], ["dock"]][topic],
                }
            )
    return pd.DataFrame(rows), np.array(emb)


CFG = DriftConfig(
    reference_windows=2,
    max_sample=100,
    min_sample=50,
    n_perm=40,
    alert_quantile=0.95,
    alert_p=0.05,
    new_term_lookback=3,
    new_term_min_docs=5,
)


def test_drift_series_alerts_on_topic_surge_only():
    feats, emb = synthetic_feats()
    series = drift_series(feats, emb, CFG)
    alerts, thresholds = calibrate_and_alert(series, pd.Period("2022-07", "M"), CFG)
    assert set(thresholds) >= {"mmd_rbf", "tags", "vocabulary"}
    ok = alerts[alerts["status"] == "ok"]
    drift_month = ok[ok["month"] == "2022-09"]
    assert drift_month.set_index("metric").loc["mmd_rbf", "alert"]
    assert drift_month.set_index("metric").loc["tags", "alert"]
    before = ok[(ok["month"] >= "2022-03") & (ok["month"] <= "2022-08")]
    assert before["alert"].mean() < 0.2


def test_new_terms_and_cluster_movements():
    feats, _ = synthetic_feats()
    terms = new_terms(feats, CFG)
    assert "sequoia" in set(terms.loc[terms["month"] == "2022-09", "term"])
    moves = cluster_movements(feats, CFG, {"3": ["sequoia"]})
    grow = moves[(moves["month"] == "2022-09") & (moves["cluster"] == "cluster:3")]
    assert grow["direction"].iloc[0] == "growing" and grow["share"].iloc[0] == pytest.approx(0.4)


def test_controlled_evaluation_runs():
    feats, emb = synthetic_feats()
    series = drift_series(feats, emb, CFG)
    _, thresholds = calibrate_and_alert(series, pd.Period("2022-07", "M"), CFG)
    fa = no_drift_false_alarms(feats, emb, thresholds, pd.Period("2022-07", "M"), CFG, n_months=3)
    assert fa["alert"].mean() < 0.3
    power = injected_drift_power(feats, emb, thresholds, CFG, fractions=(0.4,), trials=3)
    assert power.groupby("metric")["alert"].mean()["mmd_rbf"] >= 0.6


def test_flags_stay_boolean_with_insufficient_months():
    feats, emb = synthetic_feats()
    feats = feats[~((feats["month"] == pd.Period("2022-05", "M")) & (feats["question_id"] % 4 != 0))]
    emb_sub = emb[feats["question_id"].to_numpy()]
    series = drift_series(feats.reset_index(drop=True), emb_sub, CFG)
    alerts, _ = calibrate_and_alert(series, pd.Period("2022-07", "M"), CFG)
    assert (alerts["status"] == "insufficient_data").any()
    assert alerts["calibration"].dtype == bool and alerts["alert"].dtype == bool
    assert (~alerts["calibration"]).dtype == bool
