"""FastAPI application: ``uvicorn backend.app.main:app``."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core import errors
from backend.app.core.config import get_settings
from backend.app.db.session import get_engine
from backend.app.services.jobs import JobRunner


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner = JobRunner(get_engine())
    runner.mark_interrupted()
    app.state.jobs = runner
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Apple Issue Intelligence API",
        version="0.2.0",
        description=(
            "Emerging-issue signals, related questions, knowledge graph and drift monitoring over public "
            "Ask Different questions. Outputs are candidate discussion patterns, not confirmed defects. "
            "Every response states whether it is real or demo data (`meta.data_source`)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    errors.install(app)
    app.include_router(router)
    return app


app = create_app()
