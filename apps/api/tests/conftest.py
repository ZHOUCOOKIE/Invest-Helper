from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO_API_DIR = Path(__file__).resolve().parents[1]


def _is_test_database_url(url: str) -> bool:
    parsed = urlparse(url)
    db_name = parsed.path.lstrip("/").lower()
    query = parse_qs(parsed.query)
    schema_values = query.get("currentSchema", []) + query.get("search_path", [])
    schema_text = ",".join(schema_values).lower()

    return ("test" in db_name) or ("test" in schema_text)


async def _reset_public_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


def _run_migrations() -> None:
    env = os.environ.copy()
    command = ["uv", "run", "alembic", "upgrade", "head"]
    try:
        subprocess.run(command, cwd=REPO_API_DIR, env=env, check=True)
    except FileNotFoundError:
        subprocess.run(["alembic", "upgrade", "head"], cwd=REPO_API_DIR, env=env, check=True)


def _resolve_test_database_url() -> str:
    explicit_test_url = (os.getenv("DATABASE_URL_TEST") or "").strip()
    default_url = (os.getenv("DATABASE_URL") or "").strip()
    chosen_url = explicit_test_url or default_url
    if not chosen_url:
        pytest.exit("DATABASE_URL_TEST (or DATABASE_URL) is required for tests.", returncode=2)
    if not _is_test_database_url(chosen_url):
        pytest.exit(
            f"Refusing to run tests on non-test database URL: {chosen_url!r}. "
            "Use DATABASE_URL_TEST with a database/schema name containing 'test'.",
            returncode=2,
        )
    return chosen_url


def pytest_sessionstart(session: pytest.Session) -> None:
    database_url = _resolve_test_database_url()
    os.environ["ENV"] = "test"
    os.environ["DEBUG"] = "true"
    os.environ["DATABASE_URL"] = database_url
    os.environ.setdefault("DATABASE_URL_TEST", database_url)

    if os.getenv("TEST_DB_RESET_BEFORE", "1") == "1":
        asyncio.run(_reset_public_schema(database_url))
    _run_migrations()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.getenv("TEST_DB_RESET_AFTER", "1") != "1":
        return

    database_url = _resolve_test_database_url()
    asyncio.run(_reset_public_schema(database_url))
