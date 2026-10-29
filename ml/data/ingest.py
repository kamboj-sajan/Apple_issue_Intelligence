"""Checkpointed, resumable ingestion of Ask Different questions.

Two modes:

* ``ingest_by_creation`` walks fixed creation-date chunks (default: calendar months). A chunk that
  lies entirely in the past is fetched once; its progress is checkpointed after every page, so an
  interrupted or quota-limited run resumes at the next page. A chunk that is still "open" (ends in
  the future) is re-fetched on every run into a run-specific batch.
* ``ingest_updates`` fetches questions whose *last activity* is after a watermark, which captures
  later edits, score changes and duplicate closures (needed for link-prediction labels).

Raw pages may therefore contain several versions of a question; normalisation keeps the latest.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.data.raw_store import RawStore, atomic_write_text
from ml.data.stackexchange_client import ApiError, QuotaExhausted, StackExchangeClient

log = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
UPDATE_OVERLAP_SECONDS = 3600


def to_unix(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def parse_date(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    dt = datetime.fromisoformat(str(value))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def month_chunks(start: datetime, end: datetime) -> list[tuple[str, datetime, datetime]]:
    """Calendar-month chunks ``[chunk_start, chunk_end)`` clipped to ``[start, end)``."""
    chunks = []
    cur = datetime(start.year, start.month, 1, tzinfo=UTC)
    while cur < end:
        nxt = datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
        lo, hi = max(cur, start), min(nxt, end)
        if lo < hi:
            chunks.append((cur.strftime("%Y-%m"), lo, hi))
        cur = nxt
    return chunks


@dataclass
class RunSummary:
    run_id: str
    mode: str
    started_at: str
    finished_at: str | None = None
    requests: int = 0
    pages: int = 0
    items: int = 0
    chunks_completed: list[str] = field(default_factory=list)
    stopped_reason: str = "completed"
    quota_remaining: int | None = None


class Checkpoint:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.state: dict[str, Any] = json.loads(self.path.read_text())
        else:
            self.state = {"version": CHECKPOINT_VERSION, "chunks": {}, "update": {}, "runs": []}

    def save(self) -> None:
        atomic_write_text(self.path, json.dumps(self.state, indent=2, sort_keys=True))

    def chunk(self, chunk_id: str) -> dict[str, Any]:
        return self.state["chunks"].setdefault(chunk_id, {"status": "pending", "next_page": 1})


class Ingestor:
    def __init__(
        self,
        client: StackExchangeClient,
        store: RawStore,
        checkpoint: Checkpoint,
        page_size: int = 100,
        quota_reserve: int = 10,
        max_requests: int | None = None,
        now: datetime | None = None,
    ):
        self.client = client
        self.store = store
        self.checkpoint = checkpoint
        self.page_size = page_size
        self.quota_reserve = quota_reserve
        self.max_requests = max_requests
        self.now = now or datetime.now(UTC)

    def _budget_reason(self, summary: RunSummary) -> str | None:
        if self.max_requests is not None and summary.requests >= self.max_requests:
            return "max_requests"
        q = self.client.quota_remaining
        if q is not None and q <= self.quota_reserve:
            return "quota_reserve"
        return None

    def _fetch(self, summary: RunSummary, kind: str, batch: str, page: int, params: dict):
        resp = self.client.get("questions", {**params, "page": page, "pagesize": self.page_size})
        summary.requests += 1
        summary.pages += 1
        summary.items += len(resp.items)
        summary.quota_remaining = resp.quota_remaining
        self.store.write_page(kind, batch, page, {"method": resp.method, **resp.params}, resp.payload)
        return resp

    def _finish(self, summary: RunSummary) -> RunSummary:
        summary.finished_at = datetime.now(UTC).isoformat()
        self.checkpoint.state["runs"].append(summary.__dict__)
        self.checkpoint.save()
        log.info("run %s finished: %s", summary.run_id, summary.__dict__)
        return summary

    def ingest_by_creation(self, start: datetime, end: datetime | None = None) -> RunSummary:
        end = min(end or self.now, self.now)
        summary = RunSummary(uuid.uuid4().hex[:12], "creation", datetime.now(UTC).isoformat())
        try:
            for chunk_id, lo, hi in month_chunks(start, end):
                is_open = hi >= self.now
                state = self.checkpoint.chunk(chunk_id)
                if state["status"] == "complete" and not is_open:
                    continue
                batch = f"{chunk_id}_open_{summary.run_id}" if is_open else chunk_id
                page = 1 if is_open else state["next_page"]
                # API 'todate' is inclusive, so subtract one second from the exclusive bound.
                params = {
                    "fromdate": to_unix(lo),
                    "todate": to_unix(hi) - 1,
                    "sort": "creation",
                    "order": "asc",
                }
                while True:
                    if reason := self._budget_reason(summary):
                        summary.stopped_reason = reason
                        return self._finish(summary)
                    resp = self._fetch(summary, "creation", batch, page, params)
                    page += 1
                    if not is_open:
                        state.update(next_page=page, status="partial")
                        state["items"] = state.get("items", 0) + len(resp.items)
                    if not resp.has_more:
                        break
                    self.checkpoint.save()
                if not is_open:
                    state.update(status="complete", completed_at=to_unix(datetime.now(UTC)))
                    summary.chunks_completed.append(chunk_id)
                self.checkpoint.save()
        except QuotaExhausted as exc:
            summary.stopped_reason = f"quota_exhausted: {exc}"
        except ApiError as exc:
            summary.stopped_reason = f"api_error: {exc}"
        return self._finish(summary)

    def ingest_updates(self, start: datetime, since: datetime | None = None) -> RunSummary:
        """Fetch questions created after ``start`` whose last activity is after the watermark."""
        upd = self.checkpoint.state["update"]
        summary = RunSummary(uuid.uuid4().hex[:12], "activity", datetime.now(UTC).isoformat())
        if upd.get("in_progress"):
            run = upd["in_progress"]  # resume an interrupted update run
        else:
            if since is not None:
                min_ts = to_unix(since)
            elif upd.get("watermark") is not None:
                min_ts = upd["watermark"]
            else:
                done = [c.get("completed_at") for c in self.checkpoint.state["chunks"].values()]
                done = [t for t in done if t]
                if not done:
                    summary.stopped_reason = "nothing_to_update"
                    return self._finish(summary)
                min_ts = min(done) - UPDATE_OVERLAP_SECONDS
            run = {
                "batch": f"activity_{summary.run_id}",
                "min": min_ts,
                "next_page": 1,
                "started_at": to_unix(self.now),
            }
            upd["in_progress"] = run
            self.checkpoint.save()

        params = {"min": run["min"], "fromdate": to_unix(start), "sort": "activity", "order": "asc"}
        try:
            while True:
                if reason := self._budget_reason(summary):
                    summary.stopped_reason = reason
                    return self._finish(summary)
                resp = self._fetch(summary, "activity", run["batch"], run["next_page"], params)
                run["next_page"] += 1
                self.checkpoint.save()
                if not resp.has_more:
                    break
            upd["watermark"] = run["started_at"] - UPDATE_OVERLAP_SECONDS
            upd.pop("in_progress", None)
        except QuotaExhausted as exc:
            summary.stopped_reason = f"quota_exhausted: {exc}"
        except ApiError as exc:
            summary.stopped_reason = f"api_error: {exc}"
        return self._finish(summary)

    def ingest_ids(self, question_ids: list[int], batch_size: int = 100) -> RunSummary:
        """Fetch specific questions (e.g. duplicate targets created before the research window).

        Ids already fetched, or confirmed missing (deleted/inaccessible), are skipped on later runs.
        """
        state = self.checkpoint.state.setdefault("by_id", {"fetched": [], "missing": []})
        done = set(state["fetched"]) | set(state["missing"])
        todo = sorted({int(i) for i in question_ids} - done)
        summary = RunSummary(uuid.uuid4().hex[:12], "by_id", datetime.now(UTC).isoformat())
        try:
            for n, start in enumerate(range(0, len(todo), batch_size), start=1):
                if reason := self._budget_reason(summary):
                    summary.stopped_reason = reason
                    return self._finish(summary)
                ids = todo[start : start + batch_size]
                resp = self.client.get(f"questions/{';'.join(map(str, ids))}", {"pagesize": batch_size})
                summary.requests += 1
                summary.pages += 1
                summary.items += len(resp.items)
                summary.quota_remaining = resp.quota_remaining
                self.store.write_page(
                    "by_id", f"ids_{summary.run_id}", n, {"method": resp.method, **resp.params}, resp.payload
                )
                got = {int(item["question_id"]) for item in resp.items}
                state["fetched"].extend(sorted(got))
                state["missing"].extend(sorted(set(ids) - got))
                self.checkpoint.save()
        except QuotaExhausted as exc:
            summary.stopped_reason = f"quota_exhausted: {exc}"
        except ApiError as exc:
            summary.stopped_reason = f"api_error: {exc}"
        return self._finish(summary)
