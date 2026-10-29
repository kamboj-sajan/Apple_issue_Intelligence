import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.splits import compute_boundaries, make_splits, snap_to_monday


@pytest.fixture
def questions():
    rng = np.random.default_rng(0)
    created = pd.Timestamp("2023-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 365, 500)), unit="D")
    return pd.DataFrame({"question_id": np.arange(500), "created_at": created})


def test_snap_to_monday():
    assert snap_to_monday(pd.Timestamp("2024-01-04T15:00Z")) == pd.Timestamp("2024-01-01T00:00Z")
    assert snap_to_monday(pd.Timestamp("2024-01-01T00:00Z")) == pd.Timestamp("2024-01-01T00:00Z")


def test_splits_are_temporal_disjoint_and_week_aligned(questions):
    splits, manifest = make_splits(questions)
    df = questions.merge(splits, on="question_id")
    assert set(df["split"]) == {"train", "val", "test"}
    assert df.groupby("split")["created_at"].max()["train"] < df.groupby("split")["created_at"].min()["val"]
    assert df.groupby("split")["created_at"].max()["val"] < df.groupby("split")["created_at"].min()["test"]
    assert sum(manifest.counts.values()) == len(questions)
    for b in (manifest.val_start, manifest.test_start):
        assert pd.Timestamp(b).weekday() == 0 and pd.Timestamp(b).hour == 0
    week = df["created_at"].dt.tz_localize(None).dt.to_period("W-SUN")
    assert (df.groupby(week)["split"].nunique() == 1).all()  # no week straddles splits


def test_splits_deterministic_and_order_independent(questions):
    _, m1 = make_splits(questions)
    _, m2 = make_splits(questions.sample(frac=1, random_state=1))
    assert m1 == m2


def test_explicit_boundaries_and_validation(questions):
    v, t = compute_boundaries(questions["created_at"], val_start="2023-09-06", test_start="2023-11-01")
    assert v == pd.Timestamp("2023-09-04T00:00Z") and t == pd.Timestamp("2023-10-30T00:00Z")
    with pytest.raises(ValueError):
        compute_boundaries(questions["created_at"], train_fraction=0.9, val_fraction=0.2)


def test_context_questions_do_not_shift_boundaries(questions):
    _, base = make_splits(questions)
    old = pd.DataFrame(
        {
            "question_id": np.arange(1000, 1400),
            "created_at": pd.Timestamp("2015-01-01", tz="UTC") + pd.to_timedelta(np.arange(400), unit="D"),
            "in_research_window": False,
        }
    )
    combined = pd.concat([questions.assign(in_research_window=True), old], ignore_index=True)
    splits, m = make_splits(combined)
    assert (m.val_start, m.test_start) == (base.val_start, base.test_start)
    assert m.counts["context_outside_window"] == 400
    assert m.counts["train"] == base.counts["train"]
    assert (splits.loc[splits["question_id"] >= 1000, "split"] == "train").all()
