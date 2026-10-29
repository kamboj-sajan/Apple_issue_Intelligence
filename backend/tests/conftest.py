"""API test fixtures: a dedicated ``<db>_test`` database, migrated with Alembic and loaded with the
SYNTHETIC demo dataset. Tests are skipped when PostgreSQL is unreachable."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import get_settings


def _test_url() -> str:
    url = make_url(get_settings().database_url)
    return url.set(database=f"{url.database}_test").render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def test_db_url() -> str:
    base = make_url(get_settings().database_url)
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as c:
            name = f"{base.database}_test"
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            c.execute(text(f'CREATE DATABASE "{name}"'))
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"PostgreSQL not available: {exc}")
    finally:
        admin.dispose()
    return _test_url()


@pytest.fixture(scope="session")
def engine(test_db_url):
    from backend.app.cli import migrate
    from backend.app.services.demo import load_demo

    migrate(test_db_url)
    eng = create_engine(test_db_url)
    load_demo(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def client(engine):
    from fastapi.testclient import TestClient

    from backend.app.db import session as db_session
    from backend.app.main import create_app

    maker = sessionmaker(bind=engine, expire_on_commit=False)

    def override():
        s = maker()
        try:
            yield s
        finally:
            s.close()

    db_session.get_engine.cache_clear()
    app = create_app()
    app.dependency_overrides[db_session.get_session] = override
    original = db_session.get_engine
    db_session.get_engine = lambda url=None: engine  # job runner uses the test database too
    import backend.app.main as main_module

    main_module.get_engine = db_session.get_engine
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    db_session.get_engine = original
    main_module.get_engine = original
