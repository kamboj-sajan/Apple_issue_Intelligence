from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session

from backend.app.db.session import get_session

SessionDep = Annotated[Session, Depends(get_session)]
Limit = Annotated[int, Query(ge=1, le=100, description="page size (max 100)")]
Offset = Annotated[int, Query(ge=0, le=1_000_000)]


def csv_set(value: str | None) -> set[str] | None:
    return {v.strip() for v in value.split(",") if v.strip()} if value else None


def runner(request: Request):
    return request.app.state.jobs


JobsDep = Annotated[object, Depends(runner)]
