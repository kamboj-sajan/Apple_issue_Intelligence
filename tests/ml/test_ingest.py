from datetime import UTC, datetime

from ml.data.ingest import Checkpoint, Ingestor, month_chunks, to_unix
from ml.data.raw_store import RawStore
from ml.data.stackexchange_client import StackExchangeClient
from ml.preprocessing.normalize import deduplicate_versions, load_raw_items
from tests.conftest import FakeClock, FakeResponse, FakeSession, make_question

# Synthetic corpus: 5 questions in Jan 2024, 3 in Feb 2024, 2 in Mar 2024.
CORPUS = (
    [make_question(100 + i, to_unix(datetime(2024, 1, 2 + i, tzinfo=UTC))) for i in range(5)]
    + [make_question(200 + i, to_unix(datetime(2024, 2, 3 + i, tzinfo=UTC))) for i in range(3)]
    + [make_question(300 + i, to_unix(datetime(2024, 3, 4 + i, tzinfo=UTC))) for i in range(2)]
)


def api_handler(corpus, quota_start=300):
    state = {"quota": quota_start}

    def handler(url, params):
        state["quota"] -= 1
        if params.get("sort") == "activity":
            sel = [q for q in corpus if q["last_activity_date"] >= params["min"]]
            sel.sort(key=lambda q: q["last_activity_date"])
        else:
            sel = [q for q in corpus if params["fromdate"] <= q["creation_date"] <= params["todate"]]
            sel.sort(key=lambda q: q["creation_date"])
        page, size = params["page"], params["pagesize"]
        chunk = sel[(page - 1) * size : page * size]
        return FakeResponse({"items": chunk, "has_more": page * size < len(sel), "quota_remaining": state["quota"]})

    return handler


def build(tmp_path, corpus=CORPUS, max_requests=None, now=datetime(2024, 3, 20, tzinfo=UTC), page_size=2):
    session = FakeSession(handler=api_handler(corpus))
    clock = FakeClock()
    client = StackExchangeClient("apple", "F", session=session, sleep=clock.sleep, clock=clock, min_interval=0)
    ing = Ingestor(
        client,
        RawStore(tmp_path / "raw"),
        Checkpoint(tmp_path / "ckpt.json"),
        page_size=page_size,
        max_requests=max_requests,
        now=now,
    )
    return ing, session


def test_month_chunks_clip_and_cover_range():
    chunks = month_chunks(datetime(2023, 12, 15, tzinfo=UTC), datetime(2024, 2, 10, tzinfo=UTC))
    assert [c[0] for c in chunks] == ["2023-12", "2024-01", "2024-02"]
    assert chunks[0][1] == datetime(2023, 12, 15, tzinfo=UTC)
    assert chunks[-1][2] == datetime(2024, 2, 10, tzinfo=UTC)


def test_paginates_and_completes_closed_chunks(tmp_path):
    ing, session = build(tmp_path)
    summary = ing.ingest_by_creation(datetime(2024, 1, 1, tzinfo=UTC))
    assert summary.stopped_reason == "completed"
    assert summary.chunks_completed == ["2024-01", "2024-02"]  # March is still open on 2024-03-20
    assert summary.items == 10
    # Jan has 5 questions at page size 2 -> 3 pages
    jan_calls = [c for c in session.calls if c["params"]["fromdate"] == to_unix(datetime(2024, 1, 1, tzinfo=UTC))]
    assert [c["params"]["page"] for c in jan_calls] == [1, 2, 3]
    assert all(c["params"]["todate"] < to_unix(datetime(2024, 2, 1, tzinfo=UTC)) for c in jan_calls)


def test_resume_after_request_budget_and_idempotent_rerun(tmp_path):
    ing, _ = build(tmp_path, max_requests=2)
    first = ing.ingest_by_creation(datetime(2024, 1, 1, tzinfo=UTC))
    assert first.stopped_reason == "max_requests"
    assert Checkpoint(tmp_path / "ckpt.json").state["chunks"]["2024-01"] == {
        "status": "partial",
        "next_page": 3,
        "items": 4,
    }

    ing2, session2 = build(tmp_path)
    second = ing2.ingest_by_creation(datetime(2024, 1, 1, tzinfo=UTC))
    assert second.stopped_reason == "completed"
    assert session2.calls[0]["params"]["page"] == 3  # resumed, not restarted

    items, _ = load_raw_items(RawStore(tmp_path / "raw"))
    assert sorted(items["question_id"].unique()) == sorted(q["question_id"] for q in CORPUS)

    # Re-run: closed chunks are skipped; only the open month is fetched again.
    ing3, session3 = build(tmp_path)
    ing3.ingest_by_creation(datetime(2024, 1, 1, tzinfo=UTC))
    assert {c["params"]["fromdate"] for c in session3.calls} == {to_unix(datetime(2024, 3, 1, tzinfo=UTC))}
    items, _ = load_raw_items(RawStore(tmp_path / "raw"))
    latest, collapsed = deduplicate_versions(items)
    assert len(latest) == len(CORPUS)  # no duplicate records after normalisation
    assert collapsed == 2  # the two March questions were fetched twice


def test_update_mode_captures_later_activity_and_sets_watermark(tmp_path):
    ing, _ = build(tmp_path)
    ing.ingest_by_creation(datetime(2024, 1, 1, tzinfo=UTC))
    later = to_unix(datetime(2024, 3, 25, tzinfo=UTC))
    corpus = [dict(q) for q in CORPUS]
    corpus[0].update(
        last_activity_date=later,
        closed_reason="Duplicate",
        closed_details={"original_questions": [{"question_id": 101}]},
    )

    session = FakeSession(handler=api_handler(corpus))
    clock = FakeClock()
    client = StackExchangeClient("apple", "F", session=session, sleep=clock.sleep, clock=clock, min_interval=0)
    upd = Ingestor(
        client,
        RawStore(tmp_path / "raw"),
        Checkpoint(tmp_path / "ckpt.json"),
        page_size=2,
        now=datetime(2024, 3, 26, tzinfo=UTC),
    )
    summary = upd.ingest_updates(datetime(2024, 1, 1, tzinfo=UTC), since=datetime(2024, 3, 24, tzinfo=UTC))
    assert summary.items == 1
    assert (
        Checkpoint(tmp_path / "ckpt.json").state["update"]["watermark"]
        == to_unix(datetime(2024, 3, 26, tzinfo=UTC)) - 3600
    )

    items, _ = load_raw_items(RawStore(tmp_path / "raw"))
    latest, _ = deduplicate_versions(items)
    row = latest.set_index("question_id").loc[100]
    assert row["closed_reason"] == "Duplicate"  # newest version wins
