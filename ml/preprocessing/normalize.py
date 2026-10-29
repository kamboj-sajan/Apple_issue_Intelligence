"""Raw API pages -> normalised, de-duplicated tables.

Outputs (Parquet, in ``data/processed``):

* ``questions``: one row per question id (latest fetched version), UTC timestamps, cleaned text,
  rule-based categories and data-quality flags.
* ``question_links``: duplicate-closure and body-hyperlink edges between questions, each with the
  timestamp at which the link is known to exist (used for leakage-safe graph construction).
* ``question_entities``: long table of extracted entities with evidence.
* ``manifest.json``: snapshot id, counts, missing-value report and config hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ml import PIPELINE_VERSION
from ml.data.raw_store import RawStore, atomic_write_text
from ml.preprocessing.entities import EntityExtractor
from ml.preprocessing.text import clean_html, content_hash, normalize_text

log = logging.getLogger(__name__)

TS_FIELDS = {
    "creation_date": "created_at",
    "last_activity_date": "last_activity_at",
    "last_edit_date": "last_edit_at",
    "closed_date": "closed_at",
}
WINDOW_KINDS = {"creation", "activity"}
INT_FIELDS = ["score", "view_count", "answer_count", "accepted_answer_id"]


def load_raw_items(store: RawStore) -> tuple[pd.DataFrame, int]:
    """Flatten every stored page into one row per item version."""
    rows: list[dict[str, Any]] = []
    n_pages = 0
    for path, record in store.iter_pages():
        n_pages += 1
        for item in record["response"].get("items", []):
            rows.append(
                {
                    **item,
                    "_fetched_at": record["fetched_at"],
                    "_source_file": str(path.relative_to(store.root)),
                }
            )
    return pd.DataFrame(rows), n_pages


def deduplicate_versions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the most recent version of each question (by last activity, then fetch time)."""
    if df.empty:
        return df, 0
    df = df.assign(
        _activity=pd.to_numeric(df.get("last_activity_date"), errors="coerce").fillna(-1),
        _fetched=pd.to_datetime(df["_fetched_at"], utc=True),
    )
    counts = df.groupby("question_id").size().rename("n_versions")
    latest = (
        df.sort_values(["question_id", "_activity", "_fetched"])
        .drop_duplicates("question_id", keep="last")
        .drop(columns=["_activity", "_fetched"])
        .merge(counts, left_on="question_id", right_index=True)
    )
    return latest.reset_index(drop=True), len(df) - len(latest)


def _epoch_to_utc(series: pd.Series | None, n: int) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", utc=True)


def _duplicate_targets(closed_details: Any) -> list[int]:
    if not isinstance(closed_details, dict):
        return []
    return [int(q["question_id"]) for q in closed_details.get("original_questions") or [] if "question_id" in q]


def normalize_questions(
    items: pd.DataFrame, extractor: EntityExtractor
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quality: dict[str, Any] = {"input_versions": int(len(items))}
    if items.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, quality

    valid = items["question_id"].notna() & pd.to_numeric(items.get("creation_date"), errors="coerce").notna()
    quality["dropped_missing_id_or_creation"] = int((~valid).sum())
    items = items[valid].copy()
    items["question_id"] = items["question_id"].astype("int64")

    # Questions seen via creation-window or activity crawls are in the research window; questions
    # fetched only by id (e.g. older duplicate targets) are context for link prediction and must be
    # excluded from volume statistics.
    kinds = items["_source_file"].str.split("/").str[0]
    window_ids = set(items.loc[kinds.isin(WINDOW_KINDS), "question_id"])
    items, collapsed = deduplicate_versions(items)
    quality["collapsed_versions"] = int(collapsed)
    n = len(items)

    q = pd.DataFrame({"question_id": items["question_id"].to_numpy()})
    q["link"] = items.get("link")
    q["title"] = [normalize_text(t) for t in items.get("title", pd.Series([""] * n)).fillna("")]
    cleaned = [
        clean_html(b if isinstance(b, str) else "", int(qid))
        for b, qid in zip(items.get("body", pd.Series([None] * n)), items["question_id"], strict=True)
    ]
    q["body_text"] = [c.text for c in cleaned]
    q["code_text"] = [c.code for c in cleaned]
    q["linked_question_ids"] = [c.question_links for c in cleaned]
    q["tags"] = [list(t) if isinstance(t, list) else [] for t in items.get("tags", pd.Series([None] * n))]
    for src, dst in TS_FIELDS.items():
        q[dst] = _epoch_to_utc(items.get(src), n).to_numpy()
    for col in INT_FIELDS:
        q[col] = pd.to_numeric(items.get(col), errors="coerce").astype("Int64").to_numpy()
    q["is_answered"] = items.get("is_answered", pd.Series([None] * n)).astype("boolean").to_numpy()
    q["has_accepted_answer"] = q["accepted_answer_id"].notna()
    q["closed_reason"] = items.get("closed_reason", pd.Series([None] * n)).to_numpy()
    q["duplicate_of"] = [_duplicate_targets(cd) for cd in items.get("closed_details", pd.Series([None] * n))]
    q["is_closed_duplicate"] = q["duplicate_of"].map(bool)
    q["content_license"] = items.get("content_license", pd.Series([None] * n)).to_numpy()
    owners = items.get("owner", pd.Series([None] * n))
    q["author_display_name"] = [o.get("display_name") if isinstance(o, dict) else None for o in owners]
    q["author_link"] = [o.get("link") if isinstance(o, dict) else None for o in owners]
    q["n_versions"] = items["n_versions"].to_numpy()
    q["fetched_at"] = pd.to_datetime(items["_fetched_at"], utc=True).to_numpy()
    q["source_file"] = items["_source_file"].to_numpy()
    q["in_research_window"] = q["question_id"].isin(window_ids)
    q["week_start"] = q["created_at"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time.dt.tz_localize(UTC)

    # Exact-content duplicates (re-posts), distinct from community duplicate closures.
    q["content_hash"] = [content_hash(t, b) for t, b in zip(q["title"], q["body_text"], strict=True)]
    q = q.sort_values(["created_at", "question_id"]).reset_index(drop=True)
    first = q.groupby("content_hash")["question_id"].transform("first")
    q["is_content_duplicate"] = first != q["question_id"]
    q["content_duplicate_of"] = first.where(q["is_content_duplicate"]).astype("Int64")

    # Entities and categories.
    entity_rows: list[dict[str, Any]] = []
    primaries, cat_lists = [], []
    for row in q.itertuples(index=False):
        fields = {
            "title": row.title,
            "body": row.body_text,
            "code": row.code_text,
            "tags": " ".join(t.replace("-", " ") for t in row.tags),
        }
        matches = extractor.extract(fields)
        cats, primary = extractor.categorize(row.tags, f"{row.title}\n{row.body_text}")
        primaries.append(primary)
        cat_lists.append([c.value for c in cats])
        entity_rows.extend({"question_id": row.question_id, **m.to_dict()} for m in matches + cats)
    q["primary_category"] = primaries
    q["categories"] = cat_lists
    entities = pd.DataFrame(entity_rows, columns=["question_id", "entity_type", "value", "surface", "field", "rule"])

    # Question-to-question links with the earliest time the link is known to exist.
    known = set(q["question_id"])
    link_rows = []
    for row in q.itertuples(index=False):
        for target in row.duplicate_of:
            link_rows.append((row.question_id, target, "duplicate", row.closed_at))
        body_link_time = row.last_edit_at if pd.notna(row.last_edit_at) else row.created_at
        for target in row.linked_question_ids:
            link_rows.append((row.question_id, target, "body_link", body_link_time))
    links = pd.DataFrame(link_rows, columns=["source_id", "target_id", "link_type", "known_at"])
    links["target_in_corpus"] = links["target_id"].isin(known)
    links = links.drop_duplicates(["source_id", "target_id", "link_type"]).reset_index(drop=True)

    quality.update(
        n_questions=int(len(q)),
        missing={
            col: int(q[col].isna().sum())
            for col in [
                "link",
                "view_count",
                "score",
                "answer_count",
                "last_edit_at",
                "closed_at",
                "content_license",
                "author_display_name",
            ]
        },
        empty_title=int((q["title"] == "").sum()),
        empty_body=int((q["body_text"] == "").sum()),
        content_duplicates=int(q["is_content_duplicate"].sum()),
        context_questions_outside_window=int((~q["in_research_window"]).sum()),
        closed_as_duplicate=int(q["is_closed_duplicate"].sum()),
    )
    return q, links, entities, quality


def snapshot_id(questions: pd.DataFrame, extra: str = "") -> str:
    h = hashlib.sha256()
    h.update(f"{PIPELINE_VERSION}|{extra}|".encode())
    for qid, ts in sorted(zip(questions["question_id"], questions["last_activity_at"].astype(str), strict=True)):
        h.update(f"{qid}:{ts};".encode())
    return h.hexdigest()[:12]


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_preprocessing(raw_root: Path, out_dir: Path, entities_config: Path) -> dict[str, Any]:
    store = RawStore(raw_root)
    items, n_pages = load_raw_items(store)
    extractor = EntityExtractor.from_yaml(entities_config)
    questions, links, entities, quality = normalize_questions(items, extractor)
    if questions.empty:
        raise RuntimeError(f"no raw items found under {raw_root}; run ingestion first")

    out_dir.mkdir(parents=True, exist_ok=True)
    questions.to_parquet(out_dir / "questions.parquet", index=False)
    links.to_parquet(out_dir / "question_links.parquet", index=False)
    entities.to_parquet(out_dir / "question_entities.parquet", index=False)

    ent_hash = file_hash(entities_config)
    manifest = {
        "snapshot_id": snapshot_id(questions, ent_hash),
        "pipeline_version": PIPELINE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_root": str(raw_root),
        "raw_pages": n_pages,
        "entities_config_sha": ent_hash,
        "created_at_range": [questions["created_at"].min().isoformat(), questions["created_at"].max().isoformat()],
        "quality": quality,
        "links_by_type": links.groupby("link_type").size().to_dict() if not links.empty else {},
        "links_target_in_corpus": int(links["target_in_corpus"].sum()) if not links.empty else 0,
        "entities_by_type": entities.groupby("entity_type").size().to_dict() if not entities.empty else {},
        "primary_category_counts": questions["primary_category"].value_counts().to_dict(),
    }
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2, default=str))
    return manifest
