"""API settings (environment variables or .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://aii:aii_local_dev@127.0.0.1:5434/aii"
    aii_data_dir: Path = REPO_ROOT / "data"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_page_size: int = 100
    graph_max_nodes: int = 300

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
