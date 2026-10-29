from __future__ import annotations

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Meta(BaseModel):
    data_source: Literal["real", "demo", "empty"] = Field(
        description="'real' = calculated from collected Ask Different data; 'demo' = small synthetic dataset"
    )
    snapshot_id: str | None = None
    loaded_at: datetime | None = None


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int
    meta: Meta


class Envelope(BaseModel, Generic[T]):
    data: T
    meta: Meta
