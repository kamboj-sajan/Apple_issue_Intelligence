import json

import pandas as pd

from ml.config import REPO_ROOT
from ml.data.raw_store import RawStore
from ml.preprocessing.normalize import load_raw_items, normalize_questions, run_preprocessing


def _store_with_page(tmp_path, page, batch="2024-01"):
    store = RawStore(tmp_path / "raw")
    store.write_page("creation", batch, 1, {"method": "questions"}, page)
    return store


def test_normalize_fixture(tmp_path, synthetic_page, extractor):
    items, n_pages = load_raw_items(_store_with_page(tmp_path, synthetic_page))
    q, links, entities, quality = normalize_questions(items, extractor)
    assert n_pages == 1
    assert quality["dropped_missing_id_or_creation"] == 1
    assert list(q["question_id"]) == [9000001, 9000002, 9000003, 9000004]  # sorted by creation

    # UTC timestamps
    assert str(q["created_at"].dt.tz) == "UTC"
    assert q.loc[0, "created_at"] == pd.Timestamp("2024-01-01T00:00:00Z")
    assert q.loc[0, "week_start"] == pd.Timestamp("2024-01-01T00:00:00Z")  # Monday
    assert q.loc[1, "week_start"] == pd.Timestamp("2024-01-01T00:00:00Z")

    # Missing values become nulls, not zeros or crashes
    row2 = q.set_index("question_id").loc[9000002]
    assert pd.isna(row2["view_count"]) and pd.isna(row2["author_display_name"])
    assert quality["missing"]["view_count"] == 1

    # HTML cleanup
    row1 = q.set_index("question_id").loc[9000001]
    assert row1["title"] == "MacBook Pro won't boot after updating to macOS Sonoma 14.2"
    assert "error -36" in row1["code_text"] and "<p>" not in row1["body_text"]
    assert row2["body_text"] == "Battery drain since iOS 17.1 - Wi-Fi and Bluetooth off."

    # Duplicate closure and body links with known-at times
    dup = links[links["link_type"] == "duplicate"].iloc[0]
    assert (dup["source_id"], dup["target_id"]) == (9000003, 9000001)
    assert dup["known_at"] == pd.Timestamp(1704499200, unit="s", tz="UTC")
    body = links[links["link_type"] == "body_link"].iloc[0]
    assert (body["source_id"], body["target_id"]) == (9000001, 9000002)
    assert body["known_at"] == pd.Timestamp(1704100000, unit="s", tz="UTC")  # last edit, conservative
    assert len(links) == 2 and links["target_in_corpus"].all()

    # Exact-content duplicate (re-post) detection
    row4 = q.set_index("question_id").loc[9000004]
    assert row4["is_content_duplicate"] and row4["content_duplicate_of"] == 9000003

    # Entities with provenance; categories
    e1 = entities[entities["question_id"] == 9000001]
    assert {"macos:14", "macbook-pro", "error:-36", "boot-startup"} <= set(e1["value"])
    assert row1["primary_category"] == "boot-startup"
    assert row2["primary_category"] == "battery-power"


def test_latest_version_wins_across_pages(tmp_path, synthetic_page, extractor):
    store = _store_with_page(tmp_path, synthetic_page)
    newer = json.loads(json.dumps(synthetic_page))
    newer["items"] = [dict(newer["items"][1], last_activity_date=1704300000, score=7)]
    store.write_page("activity", "activity_x", 1, {"method": "questions"}, newer)
    items, _ = load_raw_items(store)
    q, _, _, quality = normalize_questions(items, extractor)
    assert q["question_id"].is_unique
    assert q.set_index("question_id").loc[9000002, "score"] == 7
    assert quality["collapsed_versions"] == 1


def test_run_preprocessing_writes_tables_and_manifest(tmp_path, synthetic_page):
    _store_with_page(tmp_path, synthetic_page)
    out = tmp_path / "processed"
    manifest = run_preprocessing(tmp_path / "raw", out, REPO_ROOT / "configs" / "entities.yaml")
    for name in ("questions", "question_links", "question_entities"):
        assert (out / f"{name}.parquet").exists()
    assert manifest["quality"]["n_questions"] == 4
    assert len(manifest["snapshot_id"]) == 12
    again = run_preprocessing(tmp_path / "raw", out, REPO_ROOT / "configs" / "entities.yaml")
    assert again["snapshot_id"] == manifest["snapshot_id"]  # deterministic
    reloaded = pd.read_parquet(out / "questions.parquet")
    assert reloaded["tags"].iloc[0].tolist() == ["macos-sonoma", "boot", "macbook-pro"]
