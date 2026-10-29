"""Issue-group membership and weekly count matrices.

An *issue group* is a key ``(group_type, value)``. Each research-window question belongs to zero
or more groups:

* ``os``, ``error``, ``tech``: values from the rule-based entity table
* ``device``: device families (model-level values such as ``iphone:15`` are kept too)
* ``category``: every rule-assigned category (not only the primary one)
* ``tag``: Stack Exchange tags
* ``tagpair``: frequent co-occurring tag pairs (minimum support configurable)
* ``cluster``: semantic clusters, when cluster labels are supplied

Counts are always paired with the site-wide weekly total, because Ask Different's volume
declines steeply over time; detectors model a group's *share* of total volume, not raw counts.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

ENTITY_GROUP_TYPES = ("os", "device", "error", "tech", "category")


def build_memberships(
    questions: pd.DataFrame,
    entities: pd.DataFrame,
    clusters: pd.Series | None = None,
    min_tagpair_support: int = 30,
) -> pd.DataFrame:
    """Long table ``question_id, group_type, group_value`` (unique rows)."""
    q = questions
    if "in_research_window" in q:
        q = q[q["in_research_window"]]
    ids = set(q["question_id"])

    ent = entities[entities["question_id"].isin(ids) & entities["entity_type"].isin(ENTITY_GROUP_TYPES)]
    parts = [
        ent.rename(columns={"entity_type": "group_type", "value": "group_value"})[
            ["question_id", "group_type", "group_value"]
        ]
    ]

    tags = q[["question_id", "tags"]].explode("tags").dropna()
    parts.append(tags.rename(columns={"tags": "group_value"}).assign(group_type="tag"))

    pairs = [
        (qid, f"{a}+{b}")
        for qid, ts in zip(q["question_id"], q["tags"], strict=True)
        for a, b in combinations(sorted(set(ts)), 2)
    ]
    if pairs:
        tp = pd.DataFrame(pairs, columns=["question_id", "group_value"]).assign(group_type="tagpair")
        support = tp["group_value"].map(tp["group_value"].value_counts())
        parts.append(tp[support >= min_tagpair_support])

    if clusters is not None:
        c = clusters[clusters.index.isin(ids)]
        parts.append(
            pd.DataFrame({"question_id": c.index, "group_value": "cluster:" + c.astype(str).to_numpy()}).assign(
                group_type="cluster"
            )
        )

    out = pd.concat(parts, ignore_index=True)[["question_id", "group_type", "group_value"]]
    return out.drop_duplicates().reset_index(drop=True)


def weekly_index(questions: pd.DataFrame) -> pd.DatetimeIndex:
    """Contiguous Monday-start weeks covering the research-window questions."""
    weeks = questions["week_start"]
    return pd.date_range(weeks.min(), weeks.max(), freq="7D", tz="UTC")


def weekly_counts(
    questions: pd.DataFrame, memberships: pd.DataFrame, min_total_count: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (counts[group x week], totals[week], partial_week flags[week]).

    ``counts`` has a MultiIndex ``(group_type, group_value)`` and a zero-filled column per week.
    Groups with fewer than ``min_total_count`` questions overall are dropped.
    """
    q = questions[questions["in_research_window"]] if "in_research_window" in questions else questions
    weeks = weekly_index(q)
    totals = q.groupby("week_start").size().reindex(weeks, fill_value=0).astype("int64")

    m = memberships.merge(q[["question_id", "week_start"]], on="question_id")
    counts = (
        m.groupby(["group_type", "group_value", "week_start"]).size().unstack("week_start", fill_value=0)
    ).reindex(columns=weeks, fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_total_count].astype("int64")

    # Edge weeks are partial if the data does not cover all 7 days.
    first, last = q["created_at"].min(), q["created_at"].max()
    partial = pd.Series(False, index=weeks)
    if len(weeks):
        partial.iloc[0] = first > weeks[0] + pd.Timedelta(hours=12)
        partial.iloc[-1] = last < weeks[-1] + pd.Timedelta(days=6, hours=12)
    return counts, totals, partial


def group_share(counts: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    denom = totals.replace(0, np.nan)
    return counts.div(denom, axis=1)
