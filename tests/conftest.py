"""Shared test helpers. All data here is SYNTHETIC, hand-written to mimic the API's JSON shape."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ml.config import REPO_ROOT
from ml.preprocessing.entities import EntityExtractor

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | None, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Returns queued responses (or a handler's response) and records each call's params."""

    def __init__(self, responses=None, handler=None):
        self.responses = list(responses or [])
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if self.handler:
            return self.handler(url, dict(params or {}))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


def make_question(qid: int, created: int, **overrides) -> dict[str, Any]:
    q = {
        "question_id": qid,
        "title": f"Synthetic question {qid}",
        "body": f"<p>Synthetic body {qid}</p>",
        "tags": ["macos"],
        "creation_date": created,
        "last_activity_date": created,
        "score": 0,
        "view_count": 10,
        "answer_count": 0,
        "is_answered": False,
        "link": f"https://apple.stackexchange.com/questions/{qid}/synthetic",
        "content_license": "CC BY-SA 4.0",
        "owner": {"display_name": "synthetic-user", "link": "https://example.invalid/u/1"},
    }
    q.update(overrides)
    return q


@pytest.fixture
def extractor() -> EntityExtractor:
    return EntityExtractor.from_yaml(REPO_ROOT / "configs" / "entities.yaml")


@pytest.fixture
def synthetic_page() -> dict[str, Any]:
    return json.loads((FIXTURES / "synthetic_questions_page.json").read_text())
