"""Reproducible time-based train/validation/test splits.

Boundaries are computed from creation-time quantiles (or given explicitly) and snapped down to a
Monday 00:00 UTC, so that no weekly window straddles two splits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from ml.data.raw_store import atomic_write_text


@dataclass
class SplitManifest:
    val_start: str
    test_start: str
    counts: dict[str, int]
    ranges: dict[str, list[str]]
    question_ids_sha: dict[str, str]
    source_snapshot_id: str | None = None


def snap_to_monday(ts: pd.Timestamp) -> pd.Timestamp:
    ts = ts.tz_convert(UTC) if ts.tzinfo else ts.tz_localize(UTC)
    day = ts.normalize()
    return day - timedelta(days=day.weekday())


def compute_boundaries(
    created_at: pd.Series,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    val_start: str | datetime | None = None,
    test_start: str | datetime | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 - train_fraction:
        raise ValueError("fractions must satisfy 0 < train < train+val < 1")
    s = created_at.sort_values()
    v = pd.Timestamp(val_start, tz=UTC) if val_start else s.quantile(train_fraction)
    t = pd.Timestamp(test_start, tz=UTC) if test_start else s.quantile(train_fraction + val_fraction)
    v, t = snap_to_monday(v), snap_to_monday(t)
    if not v < t:
        raise ValueError(f"degenerate split: val_start={v} test_start={t}; need more data or explicit dates")
    return v, t


def assign_splits(created_at: pd.Series, val_start: pd.Timestamp, test_start: pd.Timestamp) -> pd.Series:
    split = pd.Series("train", index=created_at.index, dtype="object")
    split[created_at >= val_start] = "val"
    split[created_at >= test_start] = "test"
    return split


def _ids_sha(ids: pd.Series) -> str:
    return hashlib.sha256(",".join(map(str, sorted(ids))).encode()).hexdigest()[:16]


def make_splits(
    questions: pd.DataFrame, snapshot: str | None = None, **boundary_kwargs
) -> tuple[pd.DataFrame, SplitManifest]:
    # Boundaries come from research-window questions only; context questions fetched by id (older
    # duplicate targets) are assigned by creation time but must not shift the quantiles.
    if "in_research_window" not in questions:
        questions = questions.assign(in_research_window=True)
    in_window = questions["in_research_window"].astype(bool)
    v, t = compute_boundaries(questions.loc[in_window, "created_at"], **boundary_kwargs)
    df = questions[["question_id", "created_at", "in_research_window"]].copy()
    df["split"] = assign_splits(df["created_at"], v, t)
    win = df[df["in_research_window"]]
    manifest = SplitManifest(
        val_start=v.isoformat(),
        test_start=t.isoformat(),
        counts={s: int((win["split"] == s).sum()) for s in ("train", "val", "test")}
        | {"context_outside_window": int((~df["in_research_window"]).sum())},
        ranges={
            s: [g["created_at"].min().isoformat(), g["created_at"].max().isoformat()] for s, g in win.groupby("split")
        },
        question_ids_sha={s: _ids_sha(g["question_id"]) for s, g in df.groupby("split")},
        source_snapshot_id=snapshot,
    )
    return df[["question_id", "split", "in_research_window"]], manifest


def run_splits(processed_dir: Path, **boundary_kwargs) -> SplitManifest:
    questions = pd.read_parquet(
        processed_dir / "questions.parquet", columns=["question_id", "created_at", "in_research_window"]
    )
    manifest_path = processed_dir / "manifest.json"
    snapshot = json.loads(manifest_path.read_text())["snapshot_id"] if manifest_path.exists() else None
    splits, manifest = make_splits(questions, snapshot, **boundary_kwargs)
    splits.to_parquet(processed_dir / "splits.parquet", index=False)
    atomic_write_text(processed_dir / "split_manifest.json", json.dumps(asdict(manifest), indent=2))
    return manifest
