"""Append-only storage of verbatim API pages, kept separate from normalised data."""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class RawStore:
    """Layout: ``<root>/<kind>/<batch>/page_<n>.json.gz``.

    Each file holds ``{"request": {...}, "fetched_at": iso8601, "response": <raw payload>}``.
    Pages are never modified after being written.
    """

    def __init__(self, root: Path):
        self.root = Path(root)

    def page_path(self, kind: str, batch: str, page: int) -> Path:
        return self.root / kind / batch / f"page_{page:04d}.json.gz"

    def write_page(
        self,
        kind: str,
        batch: str,
        page: int,
        request: dict[str, Any],
        response: dict[str, Any],
        fetched_at: datetime | None = None,
    ) -> Path:
        path = self.page_path(kind, batch, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "request": request,
            "fetched_at": (fetched_at or datetime.now(UTC)).isoformat(),
            "response": response,
        }
        tmp = path.with_name(path.name + ".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def iter_pages(self) -> Iterator[tuple[Path, dict[str, Any]]]:
        for path in sorted(self.root.rglob("page_*.json.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                yield path, json.load(fh)
