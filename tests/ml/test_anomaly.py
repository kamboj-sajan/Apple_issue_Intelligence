"""Anomaly detection tests on SYNTHETIC series with known structure."""

import numpy as np
import pandas as pd
import pytest

from ml.anomaly_detection import detectors as D
from ml.anomaly_detection.aggregate import build_memberships, weekly_counts
from ml.anomaly_detection.evaluation import InjectionConfig, injection_study, null_calibration, summarize_injections
from ml.anomaly_detection.pipeline import AnomalyConfig, classify_signal, detect

WEEKS = pd.date_range("2022-01-03", periods=80, freq="7D", tz="UTC")
CFG = D.DetectorConfig(lookback_weeks=20, min_history_weeks=10, min_expected=2.0)


def frame(rows, names=("a", "b", "c")):
    idx = pd.MultiIndex.from_tuples([("tag", n) for n in names[: len(rows)]])
    return pd.DataFrame(np.array(rows, dtype=float), index=idx, columns=WEEKS)


def declining_totals():
    return pd.Series(np.linspace(400, 150, len(WEEKS)).round(), index=WEEKS)


def test_bh_matches_statsmodels_and_ignores_nan():
    from statsmodels.stats.multitest import multipletests

    p = np.array([0.001, 0.02, 0.04, 0.3, np.nan, 0.011])
    q = D.benjamini_hochberg(p)
    expected = multipletests(p[np.isfinite(p)], method="fdr_bh")[1]
    assert np.allclose(q[np.isfinite(p)], expected)
    assert np.isnan(q[4])


def test_share_model_does_not_flag_site_wide_decline():
    totals = declining_totals()
    rng = np.random.default_rng(0)
    y = rng.binomial(totals.astype(int), 0.05)  # constant share, declining raw counts
    res = D.negbin(frame([y]), totals, CFG)
    q = D.benjamini_hochberg(res["p_value"].to_numpy()[0])
    assert np.nanmin(q) > 0.05


def test_negbin_flags_injected_spike_with_correct_expected():
    totals = pd.Series(300.0, index=WEEKS)
    y = np.full(len(WEEKS), 15.0)
    y[50] = 60
    res = D.negbin(frame([y]), totals, CFG)
    p = res["p_value"].to_numpy()[0]
    assert p[50] < 1e-6
    assert res["expected"].to_numpy()[0, 50] == pytest.approx(15.0)
    assert np.isnan(p[: CFG.min_history_weeks]).all()  # insufficient history -> not scored


def test_sparse_groups_are_not_scored_and_rolling_scale_is_floored():
    totals = pd.Series(300.0, index=WEEKS)
    y = np.zeros(len(WEEKS))
    y[[10, 30, 55]] = 1  # mostly zero -> MAD is 0
    y[60] = 3
    A = D.rolling_robust(frame([y]), totals, CFG)
    B = D.negbin(frame([y]), totals, CFG)
    assert np.isnan(B["p_value"].to_numpy()[0]).all()  # expected < min_expected everywhere
    assert np.isnan(A["score"].to_numpy()[0]).all()


def test_rolling_uses_only_past_weeks():
    totals = pd.Series(200.0, index=WEEKS)
    y = np.full(len(WEEKS), 10.0)
    y[40:] = 40  # level change at week 40
    A = D.rolling_robust(frame([y]), totals, CFG)
    exp = A["expected"].to_numpy()[0]
    assert exp[40] == pytest.approx(10.0)  # the shifted week itself is not in its own baseline
    assert A["score"].to_numpy()[0, 40] > 3


def test_pelt_finds_level_shift_but_not_single_spike():
    rng = np.random.default_rng(1)
    n = np.full(120, 300.0)
    y = rng.poisson(np.where(np.arange(120) < 70, 10, 25)).astype(float)
    cps = D.pelt_poisson(y, n)
    assert any(abs(c - 70) <= 2 for c in cps)
    desc = D.describe_change_points(y, n, cps)
    assert desc[-1]["rate_after"] > desc[-1]["rate_before"] and desc[-1]["p_value_approx"] < 1e-6

    y2 = rng.poisson(10, 120).astype(float)
    y2[60] = 40
    assert D.pelt_poisson(y2, n) == []


def test_pelt_with_offset_ignores_proportional_total_change():
    n = np.where(np.arange(100) < 50, 400.0, 150.0)
    y = n * 0.05
    assert D.pelt_poisson(y, n) == []


def test_classify_signal():
    cfg = AnomalyConfig()
    cps = [{"index": 50, "rel_change": 1.0}]
    assert classify_signal(50, 80, cps, cfg) == "sustained"
    assert classify_signal(30, 80, cps, cfg) == "spike"
    assert classify_signal(78, 80, [], cfg) == "too_recent"
    assert classify_signal(50, 80, [{"index": 50, "rel_change": -0.5}], cfg) == "spike"


def _synthetic_corpus():
    """Questions over 80 weeks; tag 'wifi' spikes in week 60. SYNTHETIC."""
    rng = np.random.default_rng(3)
    rows, qid = [], 0
    for w, week in enumerate(WEEKS):
        for _ in range(60):
            tags = [str(rng.choice(["macos", "iphone", "ipad", "safari"]))]
            if rng.random() < 0.1 or (w == 60 and rng.random() < 0.5):
                tags.append("wifi")
            rows.append(
                {"question_id": qid, "created_at": week + pd.Timedelta(hours=int(rng.integers(0, 160))), "tags": tags}
            )
            qid += 1
    q = pd.DataFrame(rows)
    q["week_start"] = q["created_at"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")
    q["in_research_window"] = True
    q["view_count"] = 1
    ent = pd.DataFrame(columns=["question_id", "entity_type", "value", "surface", "field", "rule"])
    return q, ent


def test_memberships_and_weekly_counts_exclude_context_and_mark_partial_weeks():
    q, ent = _synthetic_corpus()
    q.loc[0, "in_research_window"] = False
    m = build_memberships(q, ent, min_tagpair_support=10)
    assert 0 not in set(m["question_id"])
    counts, totals, partial = weekly_counts(q, m)
    assert totals.sum() == len(q) - 1
    assert counts.columns.equals(totals.index)
    assert ("tagpair", "macos+wifi") in counts.index


def test_detect_end_to_end_ranks_injected_spike_first():
    q, ent = _synthetic_corpus()
    cfg = AnomalyConfig(detector=CFG, min_tagpair_support=10)
    res = detect(q, ent, cfg)
    sig = res["signals"]
    top = sig.iloc[0]
    assert top["group_value"] in {"wifi"} or "wifi" in top["group_value"]
    assert top["week_start"] == WEEKS[60]
    assert top["signal_type"] == "spike"
    assert top["q_value"] <= 0.05 and top["observed"] > top["expected"]
    assert len(top["evidence_question_ids"]) > 0
    assert sig["anomaly_id"].is_unique
    assert "not evidence of a product defect" in top["limitations"]


def test_injection_study_and_null_calibration_run_deterministically():
    rng = np.random.default_rng(5)
    totals = pd.Series(rng.integers(250, 350, len(WEEKS)).astype(float), index=WEEKS)
    rows = [rng.binomial(totals.astype(int), s) for s in (0.05, 0.08, 0.12)]
    counts = frame(rows)
    cfg = AnomalyConfig(detector=CFG)
    inj = InjectionConfig(n_trials=20, seed=1)
    r1 = injection_study(counts, totals, cfg, inj)
    r2 = injection_study(counts, totals, cfg, inj)
    pd.testing.assert_frame_equal(r1, r2)
    summary = summarize_injections(r1)
    big = summary[(summary["kind"] == "spike") & (summary["factor"] == 5.0)]
    assert big.empty or big["negbin_recall"].iloc[0] >= 0.5
    cal = null_calibration(counts, totals, cfg, n_series=3)
    nb_all = cal[(cal["method"] == "negbin") & (cal["expected_band"] == "all")]["fpr"].iloc[0]
    assert nb_all < 0.15
