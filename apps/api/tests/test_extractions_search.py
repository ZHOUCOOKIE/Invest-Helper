from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, ReviewStatus
from main import app, reset_runtime_counters
from models import PostExtraction, RawPost
from settings import get_settings


class _FakeResult:
    def __init__(self, items: list[object]):
        self._items = items

    def scalars(self) -> "_FakeResult":
        return self

    def all(self) -> list[object]:
        return self._items

    def scalar_one_or_none(self) -> object | None:
        return self._items[0] if self._items else None


class _FakeSession:
    def __init__(self) -> None:
        self.raw_posts: dict[int, RawPost] = {}
        self.extractions: dict[int, PostExtraction] = {}

    async def execute(self, query):  # noqa: ANN001
        entity = query.column_descriptions[0]["entity"]
        if entity is PostExtraction:
            items = sorted(
                self.extractions.values(),
                key=lambda item: (item.created_at or datetime.min.replace(tzinfo=UTC), item.id or 0),
                reverse=True,
            )
            for item in items:
                item.raw_post = self.raw_posts.get(item.raw_post_id)
            return _FakeResult(items)
        if entity is RawPost:
            items = list(self.raw_posts.values())
            return _FakeResult(items)
        return _FakeResult([])

    async def get(self, model, obj_id: int):  # noqa: ANN001
        if model is RawPost:
            return self.raw_posts.get(obj_id)
        return None


def _seed_pending(fake_db: _FakeSession) -> None:
    now = datetime.now(UTC)
    fake_db.raw_posts[1] = RawPost(
        id=1,
        platform="x",
        kol_id=1,
        author_handle="semicon_kol",
        external_id="rp-1",
        url="https://x.com/semicon/status/1",
        content_text="今天继续看好半导体链条。",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.raw_posts[2] = RawPost(
        id=2,
        platform="x",
        kol_id=2,
        author_handle="macro_kol",
        external_id="rp-2",
        url="https://x.com/macro/status/2",
        content_text="美元指数波动，等待数据。",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.extractions[11] = PostExtraction(
        id=11,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "半导体短线延续强势", "reasoning": "库存周期改善"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )
    fake_db.extractions[12] = PostExtraction(
        id=12,
        raw_post_id=2,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "宏观观察", "reasoning": "等待数据落地"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )


def test_extractions_search_filters_by_keyword() -> None:
    get_settings.cache_clear()
    reset_runtime_counters()
    fake_db = _FakeSession()
    _seed_pending(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/extractions?status=pending&q=半导体&limit=20&offset=0")
    clear = client.get("/extractions?status=pending&q=&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    filtered = response.json()
    assert len(filtered) == 1
    assert filtered[0]["raw_post_id"] == 1

    assert clear.status_code == 200
    all_items = clear.json()
    assert len(all_items) == 2


def test_extractions_status_library_filters_islibrary_rows() -> None:
    get_settings.cache_clear()
    reset_runtime_counters()
    fake_db = _FakeSession()
    now = datetime.now(UTC)
    fake_db.raw_posts[1] = RawPost(
        id=1,
        platform="x",
        kol_id=1,
        author_handle="lib_kol",
        external_id="rp-lib-1",
        url="https://x.com/lib/status/1",
        content_text="library post",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.raw_posts[2] = RawPost(
        id=2,
        platform="x",
        kol_id=2,
        author_handle="view_kol",
        external_id="rp-view-2",
        url="https://x.com/view/status/2",
        content_text="view post",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.extractions[31] = PostExtraction(
        id=31,
        raw_post_id=1,
        status=ExtractionStatus.approved,
        extracted_json={
            "as_of": now.date().isoformat(),
            "source_url": "https://x.com/lib/status/1",
            "islibrary": 1,
            "hasview": 0,
            "asset_views": [],
            "library_entry": {"tag": "macro", "summary": "测试"},
        },
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )
    fake_db.extractions[32] = PostExtraction(
        id=32,
        raw_post_id=2,
        status=ExtractionStatus.approved,
        extracted_json={
            "as_of": now.date().isoformat(),
            "source_url": "https://x.com/view/status/2",
            "islibrary": 0,
            "hasview": 1,
            "asset_views": [
                {
                    "symbol": "BTC",
                    "market": "CRYPTO",
                    "stance": "bull",
                    "horizon": "1w",
                    "confidence": 80,
                    "summary": "看多",
                }
            ],
            "library_entry": None,
        },
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/extractions?status=library&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["id"] == 31


def test_extractions_status_all_does_not_filter_by_status() -> None:
    get_settings.cache_clear()
    reset_runtime_counters()
    fake_db = _FakeSession()
    now = datetime.now(UTC)
    fake_db.raw_posts[1] = RawPost(
        id=1,
        platform="x",
        kol_id=1,
        author_handle="kol_1",
        external_id="111",
        url="https://x.com/kol_1/status/111",
        content_text="post 1",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.raw_posts[2] = RawPost(
        id=2,
        platform="x",
        kol_id=2,
        author_handle="kol_2",
        external_id="222",
        url="https://x.com/kol_2/status/222",
        content_text="post 2",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.extractions[31] = PostExtraction(
        id=31,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "pending"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )
    fake_db.extractions[32] = PostExtraction(
        id=32,
        raw_post_id=2,
        status=ExtractionStatus.approved,
        extracted_json={"summary": "approved"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/extractions?status=all&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert {item["status"] for item in payload} == {"pending", "approved"}


def test_extractions_default_latest_only() -> None:
    get_settings.cache_clear()
    reset_runtime_counters()
    fake_db = _FakeSession()
    now = datetime.now(UTC)
    fake_db.raw_posts[1] = RawPost(
        id=1,
        platform="x",
        kol_id=1,
        author_handle="kol_1",
        external_id="111",
        url="https://x.com/kol_1/status/111",
        content_text="post 1",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.extractions[100] = PostExtraction(
        id=100,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "old pending"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )
    fake_db.extractions[101] = PostExtraction(
        id=101,
        raw_post_id=1,
        status=ExtractionStatus.approved,
        extracted_json={"summary": "new approved"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=1,
        auto_policy="threshold",
        auto_applied_kol_view_ids=[1],
        created_at=now + datetime.resolution,
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    latest_only = client.get("/extractions?status=all&limit=20&offset=0")
    pending_latest_only = client.get("/extractions?status=pending&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert latest_only.status_code == 200
    latest_items = latest_only.json()
    assert [item["id"] for item in latest_items] == [101]

    assert pending_latest_only.status_code == 200
    assert pending_latest_only.json() == []


def test_extractions_sorted_by_raw_posted_at_desc() -> None:
    get_settings.cache_clear()
    reset_runtime_counters()
    fake_db = _FakeSession()
    now = datetime.now(UTC)

    fake_db.raw_posts[1] = RawPost(
        id=1,
        platform="x",
        kol_id=1,
        author_handle="kol_1",
        external_id="111",
        url="https://x.com/kol_1/status/111",
        content_text="older post",
        posted_at=now - timedelta(hours=1),
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    fake_db.raw_posts[2] = RawPost(
        id=2,
        platform="x",
        kol_id=2,
        author_handle="kol_2",
        external_id="222",
        url="https://x.com/kol_2/status/222",
        content_text="newer post",
        posted_at=now,
        fetched_at=now,
        review_status=ReviewStatus.unreviewed,
        reviewed_at=None,
        reviewed_by=None,
        raw_json=None,
    )
    # extraction #101 is newer by created_at, but its raw post is older by posted_at
    fake_db.extractions[101] = PostExtraction(
        id=101,
        raw_post_id=1,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "older post newer extraction"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now + datetime.resolution,
    )
    fake_db.extractions[100] = PostExtraction(
        id=100,
        raw_post_id=2,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "newer post older extraction"},
        model_name="test-model",
        extractor_name="openai_structured",
        last_error=None,
        auto_applied_count=0,
        auto_policy=None,
        auto_applied_kol_view_ids=None,
        created_at=now,
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/extractions?status=all&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload] == [100, 101]
