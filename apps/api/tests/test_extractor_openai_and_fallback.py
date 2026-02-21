from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus
from main import app, reset_reextract_rate_limiter
from models import PostExtraction, RawPost
from settings import get_settings


class FakeResult:
    def __init__(self, items: list[object]):
        self._items = items

    def scalars(self) -> "FakeResult":
        return self

    def all(self) -> list[object]:
        return self._items

    def scalar_one_or_none(self) -> object | None:
        return self._items[0] if self._items else None


class FakeAsyncSession:
    def __init__(self) -> None:
        self._data: dict[type[object], dict[int, object]] = {
            RawPost: {},
            PostExtraction: {},
        }
        self._new: list[object] = []

    def seed(self, obj: object) -> None:
        self._data[type(obj)][getattr(obj, "id")] = obj

    async def get(self, model: type[object], obj_id: int) -> object | None:
        return self._data.get(model, {}).get(obj_id)

    def add(self, obj: object) -> None:
        self._new.append(obj)

    async def flush(self) -> None:
        self._persist_new()

    async def commit(self) -> None:
        self._persist_new()

    async def refresh(self, obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def execute(self, query) -> FakeResult:  # noqa: ANN001
        entity = query.column_descriptions[0]["entity"]
        items = list(self._data[entity].values())

        for criterion in getattr(query, "_where_criteria", ()):
            key = criterion.left.key
            value = criterion.right.value
            items = [item for item in items if getattr(item, key) == value]

        if entity is PostExtraction:
            raw_posts = self._data[RawPost]
            for item in items:
                item.raw_post = raw_posts.get(item.raw_post_id)

        return FakeResult(items)

    def _persist_new(self) -> None:
        now = datetime.now(UTC)
        for obj in self._new:
            model = type(obj)
            bucket = self._data[model]
            if getattr(obj, "id", None) is None:
                next_id = max(bucket.keys(), default=0) + 1
                setattr(obj, "id", next_id)
            if hasattr(obj, "created_at") and getattr(obj, "created_at") is None:
                setattr(obj, "created_at", now)
            bucket[getattr(obj, "id")] = obj
        self._new.clear()


def _seed_raw_post(db: FakeAsyncSession) -> None:
    db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="post-1",
            url="https://x.com/alice/status/post-1",
            content_text="BTC looks constructive if 100k holds this week.",
            posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
            fetched_at=datetime.now(UTC),
            raw_json=None,
        )
    )


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_reextract_rate_limiter()
    yield
    get_settings.cache_clear()
    reset_reextract_rate_limiter()


def test_extract_endpoint_persists_mocked_openai_json_and_get_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)

    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "assets": [{"symbol": "BTC", "name": "Bitcoin", "market": "CRYPTO"}],
            "stance": "bull",
            "horizon": "1w",
            "confidence": 78,
            "summary": "If support holds, short-term upside remains.",
            "source_url": raw_post.url,
            "as_of": "2026-02-21",
            "event_tags": ["support"],
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    create_resp = client.post("/raw-posts/1/extract")
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["status"] == ExtractionStatus.pending.value
    assert created["extractor_name"] == "openai_structured"
    assert created["extracted_json"]["assets"][0]["symbol"] == "BTC"

    read_resp = client.get(f"/extractions/{created['id']}")
    app.dependency_overrides.clear()

    assert read_resp.status_code == 200
    read_body = read_resp.json()
    assert read_body["id"] == created["id"]
    assert read_body["extracted_json"]["stance"] == "bull"
    assert read_body["raw_post"]["id"] == 1


def test_auto_mode_without_api_key_falls_back_to_dummy_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["extractor_name"] == "dummy"
    assert body["model_name"] == "dummy-v2"
    assert body["extracted_json"]["horizon"] == "1w"


def test_reextract_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    first = client.post("/raw-posts/1/extract")
    second = client.post("/raw-posts/1/extract")
    third = client.post("/raw-posts/1/extract")
    fourth = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 201
    assert fourth.status_code == 429


def test_long_content_gets_truncated_with_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_raw_post(fake_db)
    long_content = "BTC " * 1000
    fake_db._data[RawPost][1].content_text = long_content

    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("EXTRACTION_MAX_CONTENT_CHARS", "120")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/1/extract")
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    meta = body["extracted_json"]["meta"]
    assert meta["truncated"] is True
    assert meta["original_length"] == len(long_content)
    assert meta["max_length"] == 120
