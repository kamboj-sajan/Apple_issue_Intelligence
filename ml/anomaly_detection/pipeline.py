"""Run all detectors, screen with FDR control, characterise and rank emerging-issue signals.

A *signal* is a (group, week) where the negative-binomial model's BH q-value is below
``fdr_q`` and the observed volume clears minimum-count and minimum-relative-change checks.
The rolling detector's agreement and the change-point characterisation (spike vs sustained)
are attached as supporting evidence. Signals are candidate discussion patterns, never proof of
a defect.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.anomaly_detection import detectors as D
from ml.anomaly_detection.aggregate import build_memberships, weekly_counts
from ml.data.raw_store import atomic_write_text

LIMITATIONS = (
    "Counts reflect questions posted on Ask Different, not affected users. A significant increase is a "
    "candidate discussion pattern; it is not evidence of a product defect or of a causal relationship. "
    "Expected values come from the group's share of site-wide volume over the preceding weeks."
)


@dataclass
class AnomalyConfig:
    detector: D.DetectorConfig = field(default_factory=D.DetectorConfig)
    fdr_q: float = 0.05
    min_observed: int = 5
    min_rel_change: float = 0.5
    min_group_total: int = 20
    min_tagpair_support: int = 30
    rolling_z: float = 3.0
    cp_min_group_total: int = 100
    cp_min_size: int = 4
    cp_min_rel_increase: float = 0.25
    follow_up_weeks: int = 3
    low_volume_week_ratio: float = 0.5
    series_min_group_total: int = 100
    evidence_questions: int = 5

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AnomalyConfig:
        d = dict(d or {})
        det = D.DetectorConfig(**d.pop("detector", {}))
        return cls(detector=det, **d)


def anomaly_id(group_type: str, value: str, week: pd.Timestamp, method: str) -> str:
    return hashlib.sha1(f"{group_type}|{value}|{week.date()}|{method}".encode()).hexdigest()[:16]


def _week_quality(totals: pd.Series, partial: pd.Series, ratio: float) -> pd.DataFrame:
    """Weeks with partial coverage or unusually low site volume (possible collection gaps)."""
    rolling_med = totals.shift(1).rolling(26, min_periods=8).median()
    low = (totals < ratio * rolling_med).fillna(False)
    return pd.DataFrame({"partial_week": partial, "low_site_volume": low})


def classify_signal(t: int, n_weeks: int, cps: list[dict[str, Any]], cfg: AnomalyConfig) -> str:
    """'sustained' if an increasing change point starts within the follow-up window around t;
    'too_recent' if fewer than ``follow_up_weeks`` weeks follow t; otherwise 'spike'."""
    for cp in cps:
        if t - cfg.follow_up_weeks <= cp["index"] <= t + 1 and cp["rel_change"] >= cfg.cp_min_rel_increase:
            return "sustained"
    if t >= n_weeks - cfg.follow_up_weeks:
        return "too_recent"
    return "spike"


def detect(
    questions: pd.DataFrame,
    entities: pd.DataFrame,
    cfg: AnomalyConfig,
    clusters: pd.Series | None = None,
) -> dict[str, pd.DataFrame]:
    memberships = build_memberships(questions, entities, clusters, cfg.min_tagpair_support)
    counts, totals, partial = weekly_counts(questions, memberships, cfg.min_group_total)
    weeks = counts.columns
    quality = _week_quality(totals, partial, cfg.low_volume_week_ratio)

    A = D.rolling_robust(counts, totals, cfg.detector)
    B = D.negbin(counts, totals, cfg.detector)
    A["q_value"] = D.bh_per_week(A["p_value"])
    B["q_value"] = D.bh_per_week(B["p_value"])

    y = counts.to_numpy(dtype=float)
    mu = B["expected"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (y - mu) / mu
    flag_b = (B["q_value"].to_numpy() <= cfg.fdr_q) & (y >= cfg.min_observed) & (rel >= cfg.min_rel_change)
    flag_a = (A["score"].to_numpy() >= cfg.rolling_z) & (y >= cfg.min_observed)

    # Long detection table (every flagged cell, per method) for method comparison.
    det_rows = []
    for name, res, flags in (("negbin", B, flag_b), ("rolling_robust", A, flag_a)):
        gi, ti = np.nonzero(flags)
        exp = res["expected"].to_numpy()[gi, ti]
        det_rows.append(
            pd.DataFrame(
                {
                    "method": name,
                    "group_type": counts.index.get_level_values(0)[gi],
                    "group_value": counts.index.get_level_values(1)[gi],
                    "week_start": weeks[ti],
                    "observed": y[gi, ti].astype(int),
                    "expected": exp,
                    "score": res["score"].to_numpy()[gi, ti],
                    "p_value": res["p_value"].to_numpy()[gi, ti],
                    "q_value": res["q_value"].to_numpy()[gi, ti],
                }
            )
        )
    detections = pd.concat(det_rows, ignore_index=True)

    # Change points for sufficiently large groups and every group with a negbin signal.
    signal_groups = set(np.nonzero(flag_b.any(axis=1))[0])
    big_groups = set(np.nonzero(y.sum(axis=1) >= cfg.cp_min_group_total)[0])
    n = totals.to_numpy(dtype=float)
    cp_rows, cps_by_group = [], {}
    for g in sorted(signal_groups | big_groups):
        idx = D.pelt_poisson(y[g], n, min_size=cfg.cp_min_size)
        desc = D.describe_change_points(y[g], n, idx)
        cps_by_group[g] = desc
        gt, gv = counts.index[g]
        for c in desc:
            cp_rows.append({"group_type": gt, "group_value": gv, "week_start": weeks[c["index"]], **c})
    change_points = pd.DataFrame(cp_rows)

    # Signals (primary: negbin with FDR) with supporting evidence.
    gi, ti = np.nonzero(flag_b)
    sig = pd.DataFrame(
        {
            "group_type": counts.index.get_level_values(0)[gi],
            "group_value": counts.index.get_level_values(1)[gi],
            "week_start": weeks[ti],
            "observed": y[gi, ti].astype(int),
            "expected": mu[gi, ti],
            "site_total": n[ti].astype(int),
            "p_value": B["p_value"].to_numpy()[gi, ti],
            "q_value": B["q_value"].to_numpy()[gi, ti],
            "nb_score": B["score"].to_numpy()[gi, ti],
            "overdispersion_alpha": B["alpha"].to_numpy()[gi, ti],
            "rolling_z": A["score"].to_numpy()[gi, ti],
            "rolling_agrees": flag_a[gi, ti],
            "_g": gi,
            "_t": ti,
        }
    )
    L = cfg.detector.lookback_weeks
    sig["abs_change"] = sig["observed"] - sig["expected"]
    sig["rel_change"] = sig["abs_change"] / sig["expected"]
    sig["baseline_start"] = weeks[np.maximum(ti - L, 0)]
    sig["baseline_end"] = weeks[np.maximum(ti - 1, 0)]
    sig["signal_type"] = [
        classify_signal(t, len(weeks), cps_by_group.get(g, []), cfg) for g, t in zip(gi, ti, strict=True)
    ]
    sig["partial_week"] = quality["partial_week"].to_numpy()[ti]
    sig["low_site_volume"] = quality["low_site_volume"].to_numpy()[ti]
    # Ranking score: strength of evidence x size of the effect (documented, not tuned on outcomes).
    sig["rank_score"] = -np.log10(np.clip(sig["q_value"], 1e-300, 1)) * np.log2(1 + sig["rel_change"].clip(lower=0))
    sig["anomaly_id"] = [
        anomaly_id(a, b, w, "negbin")
        for a, b, w in zip(sig["group_type"], sig["group_value"], sig["week_start"], strict=True)
    ]
    sig = _attach_evidence(sig, questions, memberships, entities, cfg)
    sig["limitations"] = LIMITATIONS
    sig = sig.drop(columns=["_g", "_t"]).sort_values("rank_score", ascending=False).reset_index(drop=True)

    # Series for charts: observed vs expected for large groups and every group with a signal.
    keep = sorted(signal_groups | set(np.nonzero(y.sum(axis=1) >= cfg.series_min_group_total)[0]))
    series = (
        pd.DataFrame(
            {
                "group_type": np.repeat(counts.index.get_level_values(0)[keep], len(weeks)),
                "group_value": np.repeat(counts.index.get_level_values(1)[keep], len(weeks)),
                "week_start": np.tile(weeks, len(keep)),
                "observed": y[keep].ravel().astype(int),
                "expected_negbin": mu[keep].ravel(),
                "expected_rolling": A["expected"].to_numpy()[keep].ravel(),
                "site_total": np.tile(n, len(keep)).astype(int),
            }
        )
        if keep
        else pd.DataFrame()
    )
    weekly_totals = pd.DataFrame({"week_start": weeks, "site_total": n.astype(int)}).join(
        quality.reset_index(drop=True)
    )
    return {
        "signals": sig,
        "detections": detections,
        "change_points": change_points,
        "series": series,
        "weekly_totals": weekly_totals,
        "_n_groups": pd.DataFrame({"n_groups": [len(counts)], "n_weeks": [len(weeks)]}),
    }


def _attach_evidence(
    sig: pd.DataFrame, questions: pd.DataFrame, memberships: pd.DataFrame, entities: pd.DataFrame, cfg: AnomalyConfig
) -> pd.DataFrame:
    """Top questions (by views) and co-occurring entities/tags for each signal's group-week."""
    if sig.empty:
        return sig.assign(evidence_question_ids=[], co_occurring=[])
    qmeta = questions.set_index("question_id")[["week_start", "view_count", "tags"]]
    by_group = memberships.groupby(["group_type", "group_value"])["question_id"].apply(list)
    ent_by_q = (
        entities[entities["entity_type"].isin(["os", "device", "error", "tech", "category"])]
        .groupby("question_id")["value"]
        .apply(set)
    )
    ev_ids, co = [], []
    for gt, gv, wk in zip(sig["group_type"], sig["group_value"], sig["week_start"], strict=True):
        members = qmeta.loc[qmeta.index.intersection(by_group.get((gt, gv), []))]
        week_q = members[members["week_start"] == wk]
        top = week_q.sort_values("view_count", ascending=False, na_position="last").index[: cfg.evidence_questions]
        ev_ids.append([int(i) for i in top])
        bag: dict[str, int] = {}
        for qid in week_q.index:
            for v in ent_by_q.get(qid, set()) | {f"tag:{t}" for t in week_q.at[qid, "tags"]}:
                if v != gv and v != f"tag:{gv}":
                    bag[v] = bag.get(v, 0) + 1
        co.append([f"{k} ({c})" for k, c in sorted(bag.items(), key=lambda kv: -kv[1])[:6]])
    return sig.assign(evidence_question_ids=ev_ids, co_occurring=co)


def run_anomaly_detection(processed_dir: Path, out_dir: Path, cfg: AnomalyConfig) -> dict[str, Any]:
    questions = pd.read_parquet(processed_dir / "questions.parquet")
    entities = pd.read_parquet(processed_dir / "question_entities.parquet")
    clusters = None
    cpath = processed_dir / "clusters.parquet"
    if cpath.exists():
        c = pd.read_parquet(cpath)
        clusters = c.set_index("question_id")["cluster"]
    res = detect(questions, entities, cfg, clusters)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("signals", "detections", "change_points", "series", "weekly_totals"):
        res[name].to_parquet(out_dir / f"{name}.parquet", index=False)
    snapshot = json.loads((processed_dir / "manifest.json").read_text()).get("snapshot_id")
    sig = res["signals"]
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_snapshot_id": snapshot,
        "config": asdict(cfg),
        "n_groups_tested": int(res["_n_groups"]["n_groups"][0]),
        "n_weeks": int(res["_n_groups"]["n_weeks"][0]),
        "n_signals": int(len(sig)),
        "signals_by_type": sig["signal_type"].value_counts().to_dict() if len(sig) else {},
        "signals_by_group_type": sig["group_type"].value_counts().to_dict() if len(sig) else {},
        "n_rolling_flags": int((res["detections"]["method"] == "rolling_robust").sum()),
        "n_change_points": int(len(res["change_points"])),
        "used_clusters": clusters is not None,
    }
    atomic_write_text(out_dir / "summary.json", json.dumps(summary, indent=2, default=str))
    return summary
