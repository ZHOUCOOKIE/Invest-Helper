from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, Horizon, ReviewStatus, Stance
from main import app, reset_runtime_counters
from models import Asset, Kol, KolView, PostExtraction, RawPost
from services.extraction import OpenAIRequestError
from settings import get_settings




@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    get_settings.cache_clear()
    reset_runtime_counters()


class FakeResult:
    def __init__(self, *, items: list[object] | None = None, scalar_value: object | None = None):
        self._items = items or []
        self._scalar_value = scalar_value

    def scalar(self) -> object | None:
        return self._scalar_value

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
            Asset: {},
            Kol: {},
            KolView: {},
        }
        self._new: list[object] = []

    def seed(self, obj: object) -> None:
        obj_id = getattr(obj, "id")
        self._data[type(obj)][obj_id] = obj

    def add(self, obj: object) -> None:
        self._new.append(obj)

    async def get(self, model: type[object], obj_id: int) -> object | None:
        return self._data.get(model, {}).get(obj_id)

    async def flush(self) -> None:
        self._persist_new()

    async def commit(self) -> None:
        self._persist_new()

    async def refresh(self, obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None

    async def delete(self, obj: object) -> None:
        model = type(obj)
        obj_id = getattr(obj, "id", None)
        if obj_id is None:
            return
        bucket = self._data.get(model)
        if bucket is None:
            return
        bucket.pop(obj_id, None)
        if model is RawPost:
            extraction_bucket = self._data[PostExtraction]
            to_remove = [eid for eid, item in extraction_bucket.items() if item.raw_post_id == obj_id]
            for eid in to_remove:
                extraction_bucket.pop(eid, None)

    async def execute(self, query) -> FakeResult:  # noqa: ANN001
        sql = str(query).lower()
        extractions = self._data[PostExtraction]

        if "count(post_extractions.id)" in sql:
            pending_count = sum(
                1 for item in extractions.values() if item.status == ExtractionStatus.pending
            )
            return FakeResult(scalar_value=pending_count)

        if "avg(kol_views.confidence)" in sql:
            return FakeResult(items=self._build_top_assets_rows())

        if "group by kol_views.horizon, kol_views.stance" in sql:
            return FakeResult(items=self._build_clarity_rows())

        entity = query.column_descriptions[0]["entity"]
        if entity is PostExtraction:
            items = list(extractions.values())
            for criterion in getattr(query, "_where_criteria", ()):
                key = criterion.left.key
                value = criterion.right.value
                items = [item for item in items if getattr(item, key) == value]

            items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
            limit = getattr(query, "_limit_clause", None)
            if limit is not None:
                items = items[: int(limit.value)]

            raw_posts = self._data[RawPost]
            for item in items:
                item.raw_post = raw_posts.get(item.raw_post_id)

            return FakeResult(items=items)

        items = list(self._data.get(entity, {}).values())
        for criterion in getattr(query, "_where_criteria", ()):
            key = criterion.left.key
            value = criterion.right.value
            if key == "symbol" and isinstance(value, str):
                items = [item for item in items if getattr(item, key).upper() == value.upper()]
            else:
                items = [item for item in items if getattr(item, key) == value]
        return FakeResult(items=items)


    def _approved_views(self) -> list[KolView]:
        views_by_id = self._data[KolView]
        approved = []
        for extraction in self._data[PostExtraction].values():
            if extraction.status == ExtractionStatus.approved and extraction.applied_kol_view_id:
                view = views_by_id.get(extraction.applied_kol_view_id)
                if view is not None:
                    approved.append(view)
        return approved

    def _build_top_assets_rows(self) -> list[SimpleNamespace]:
        assets_by_id = self._data[Asset]
        grouped: dict[int, list[KolView]] = {}
        for view in self._approved_views():
            grouped.setdefault(view.asset_id, []).append(view)

        rows: list[SimpleNamespace] = []
        for asset_id, views in grouped.items():
            asset = assets_by_id[asset_id]
            avg_conf = sum(v.confidence for v in views) / len(views)
            rows.append(
                SimpleNamespace(
                    asset_id=asset_id,
                    symbol=asset.symbol,
                    market=asset.market,
                    views_count=len(views),
                    avg_confidence=avg_conf,
                )
            )

        rows.sort(key=lambda item: (item.views_count, item.avg_confidence, -item.asset_id), reverse=True)
        return rows[:20]

    def _build_clarity_rows(self) -> list[SimpleNamespace]:
        counts: dict[tuple[Horizon, Stance], int] = {}
        for view in self._approved_views():
            key = (view.horizon, view.stance)
            counts[key] = counts.get(key, 0) + 1

        rows = [
            SimpleNamespace(horizon=horizon, stance=stance, count=count)
            for (horizon, stance), count in counts.items()
        ]
        return rows

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
            if hasattr(obj, "fetched_at") and getattr(obj, "fetched_at") is None:
                setattr(obj, "fetched_at", now)
            bucket[getattr(obj, "id")] = obj
        self._new.clear()


def test_manual_ingest_creates_raw_post_and_pending_extraction(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post(
        "/ingest/manual",
        json={
            "platform": "x",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/1",
            "content_text": "BTC might break range this week.",
        },
    )
    app.dependency_overrides.clear()

    assert response.status_code == 201
    body = response.json()
    assert body["raw_post"]["id"] == 1
    assert body["extraction"]["id"] == body["extraction_id"]
    assert body["extraction"]["status"] == "pending"


def test_dashboard_returns_top_assets_and_pending_count() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p-1",
            url="https://x.com/alice/status/1",
            content_text="BTC view",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=1,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "pending item"},
            model_name="dummy-v1",
            extractor_name="dummy",
            created_at=now,
        )
    )
    fake_db.seed(
        KolView(
            id=11,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=70,
            summary="bullish",
            source_url="https://x.com/alice/status/1",
            as_of=now.date(),
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=2,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={"summary": "approved item"},
            model_name="dummy-v1",
            extractor_name="dummy",
            applied_kol_view_id=11,
            created_at=now,
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/dashboard?days=7")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["pending_extractions_count"] == 1
    assert isinstance(body["top_assets"], list)
    assert body["top_assets"][0]["symbol"] == "BTC"
    assert body["new_views_24h"] == 1
    assert body["new_views_7d"] == 1
    assert body["assets"][0]["symbol"] == "BTC"
    assert body["assets"][0]["new_views_7d"] == 1
    assert body["assets"][0]["latest_views_by_horizon"][0]["horizon"] == "1w"
    assert body["active_kols_7d"][0]["handle"] == "alice"


def test_asset_views_feed_supports_horizon_and_pagination() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        KolView(
            id=101,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=70,
            summary="week view",
            source_url="https://x.com/alice/status/101",
            as_of=now.date(),
            created_at=now,
        )
    )
    fake_db.seed(
        KolView(
            id=102,
            kol_id=1,
            asset_id=1,
            stance=Stance.bear,
            horizon=Horizon.one_month,
            confidence=65,
            summary="month view",
            source_url="https://x.com/alice/status/102",
            as_of=now.date(),
            created_at=now,
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.get("/assets/1/views/feed?horizon=1w&limit=1&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["asset_id"] == 1
    assert body["horizon"] == "1w"
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["horizon"] == "1w"
    assert body["items"][0]["kol_handle"] == "alice"


def test_x_import_is_idempotent_by_platform_and_external_id(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "10001",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/10001",
            "posted_at": "2026-02-23T10:00:00Z",
            "content_text": "BTC still bid.",
        },
        {
            "external_id": "10002",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/10002",
            "posted_at": "2026-02-23T11:00:00Z",
            "content_text": "ETH catching up.",
        },
    ]

    first = client.post("/ingest/x/import", json=payload)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["received_count"] == 2
    assert first_body["inserted_raw_posts_count"] == 2
    assert len(first_body["inserted_raw_post_ids"]) == 2
    assert first_body["dedup_skipped_count"] == 0
    assert first_body["extract_success_count"] == 0
    assert first_body["extract_failed_count"] == 0
    assert first_body["skipped_already_extracted_count"] == 0
    assert first_body["warnings_count"] == 0
    assert first_body["resolved_author_handle"] == "alice"
    assert first_body["resolved_kol_id"] is not None
    assert first_body["kol_created"] is True
    assert len(fake_db._data[RawPost]) == 2

    second = client.post("/ingest/x/import", json=payload)
    app.dependency_overrides.clear()

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["received_count"] == 2
    assert second_body["inserted_raw_posts_count"] == 0
    assert second_body["inserted_raw_post_ids"] == []
    assert second_body["dedup_skipped_count"] == 2
    assert second_body["extract_success_count"] == 0
    assert second_body["extract_failed_count"] == 0
    assert second_body["skipped_already_extracted_count"] == 0
    assert second_body["resolved_author_handle"] == "alice"
    assert second_body["resolved_kol_id"] == first_body["resolved_kol_id"]
    assert second_body["kol_created"] is False
    assert len(fake_db._data[RawPost]) == 2


def test_x_import_triggers_extraction_for_new_posts_only(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    called_raw_post_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post):  # noqa: ANN001
        called_raw_post_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "mocked"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "20001",
            "author_handle": "bob",
            "url": "https://x.com/bob/status/20001",
            "posted_at": "2026-02-23T09:30:00Z",
            "content_text": "Watching macro risk.",
        }
    ]

    first = client.post("/ingest/x/import?trigger_extraction=true", json=payload)
    assert first.status_code == 200
    assert first.json()["extract_success_count"] == 1
    assert first.json()["skipped_already_extracted_count"] == 0
    assert len(called_raw_post_ids) == 1

    second = client.post("/ingest/x/import?trigger_extraction=true", json=payload)
    app.dependency_overrides.clear()

    assert second.status_code == 200
    assert second.json()["inserted_raw_posts_count"] == 0
    assert second.json()["extract_success_count"] == 0
    assert second.json()["skipped_already_extracted_count"] == 1
    assert len(called_raw_post_ids) == 1


def test_resume_after_partial_success_reupload_only_retries_unfinished(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "resume-1",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-1",
            "posted_at": "2026-02-23T09:00:00Z",
            "content_text": "first",
        },
        {
            "external_id": "resume-2",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-2",
            "posted_at": "2026-02-23T09:01:00Z",
            "content_text": "second",
        },
    ]

    first = client.post("/ingest/x/import", json=payload)
    assert first.status_code == 200
    inserted_ids = first.json()["inserted_raw_post_ids"]
    assert len(inserted_ids) == 2

    fake_db.seed(
        PostExtraction(
            id=100,
            raw_post_id=inserted_ids[0],
            status=ExtractionStatus.approved,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=101,
            raw_post_id=inserted_ids[1],
            status=ExtractionStatus.pending,
            extracted_json={"summary": "failed"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="OpenAIRequestError: status=429 rate_limited",
            created_at=now,
        )
    )

    called_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "retried"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    second = client.post("/ingest/x/import", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["inserted_raw_posts_count"] == 0
    assert second_body["dedup_skipped_count"] == 2
    assert len(second_body["dedup_existing_raw_post_ids"]) == 2

    extract_resp = client.post(
        "/raw-posts/extract-batch",
        json={"raw_post_ids": second_body["dedup_existing_raw_post_ids"], "mode": "pending_or_failed"},
    )
    app.dependency_overrides.clear()

    assert extract_resp.status_code == 200
    body = extract_resp.json()
    assert body["requested_count"] == 2
    assert body["success_count"] == 1
    assert body["failed_count"] == 0
    assert body["skipped_already_extracted_count"] == 1
    assert body["resumed_requested_count"] == 1
    assert body["resumed_success"] == 1
    assert body["resumed_failed"] == 0
    assert body["resumed_skipped"] == 0
    assert called_ids == [inserted_ids[1]]


def test_resume_after_full_success_reupload_skips_all_extraction(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "done-1",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/done-1",
            "posted_at": "2026-02-23T10:00:00Z",
            "content_text": "done one",
        },
        {
            "external_id": "done-2",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/done-2",
            "posted_at": "2026-02-23T10:01:00Z",
            "content_text": "done two",
        },
    ]

    first = client.post("/ingest/x/import", json=payload)
    assert first.status_code == 200
    inserted_ids = first.json()["inserted_raw_post_ids"]
    assert len(inserted_ids) == 2

    fake_db.seed(
        PostExtraction(
            id=200,
            raw_post_id=inserted_ids[0],
            status=ExtractionStatus.approved,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=201,
            raw_post_id=inserted_ids[1],
            status=ExtractionStatus.approved,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )

    calls = {"n": 0}

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "should_not_happen"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    second = client.post("/ingest/x/import", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["inserted_raw_posts_count"] == 0
    assert second_body["dedup_skipped_count"] == 2

    extract_resp = client.post(
        "/raw-posts/extract-batch",
        json={"raw_post_ids": second_body["dedup_existing_raw_post_ids"], "mode": "pending_or_failed"},
    )
    app.dependency_overrides.clear()

    assert extract_resp.status_code == 200
    body = extract_resp.json()
    assert body["requested_count"] == 2
    assert body["success_count"] == 0
    assert body["failed_count"] == 0
    assert body["skipped_count"] == 2
    assert body["skipped_already_extracted_count"] == 2
    assert body["resumed_requested_count"] == 0
    assert body["resumed_success"] == 0
    assert body["resumed_failed"] == 0
    assert body["resumed_skipped"] == 0
    assert calls["n"] == 0


def test_multi_handle_reupload_dedup_and_pending_or_failed_only_resumes_failed(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "mh-re-1",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/mh-re-1",
            "posted_at": "2026-02-23T09:00:00Z",
            "content_text": "alice content",
        },
        {
            "external_id": "mh-re-2",
            "author_handle": "bob",
            "url": "https://x.com/bob/status/mh-re-2",
            "posted_at": "2026-02-23T09:01:00Z",
            "content_text": "bob content",
        },
    ]

    first = client.post("/ingest/x/import", json=payload)
    assert first.status_code == 200
    inserted_ids = first.json()["inserted_raw_post_ids"]
    assert len(inserted_ids) == 2

    fake_db.seed(
        PostExtraction(
            id=400,
            raw_post_id=inserted_ids[0],
            status=ExtractionStatus.approved,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=401,
            raw_post_id=inserted_ids[1],
            status=ExtractionStatus.pending,
            extracted_json={"summary": "failed"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="OpenAIRequestError: status=429 rate_limited",
            created_at=now,
        )
    )

    called_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "retried"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    second = client.post("/ingest/x/import", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["inserted_raw_posts_count"] == 0
    assert second_body["dedup_skipped_count"] == 2
    assert second_body["imported_by_handle"]["alice"]["dedup"] == 1
    assert second_body["imported_by_handle"]["bob"]["dedup"] == 1

    extract_resp = client.post(
        "/raw-posts/extract-batch",
        json={"raw_post_ids": second_body["dedup_existing_raw_post_ids"], "mode": "pending_or_failed"},
    )
    app.dependency_overrides.clear()

    assert extract_resp.status_code == 200
    body = extract_resp.json()
    assert body["requested_count"] == 2
    assert body["success_count"] == 1
    assert body["failed_count"] == 0
    assert body["skipped_already_approved_count"] == 1
    assert body["resumed_requested_count"] == 1
    assert body["resumed_success"] == 1
    assert body["resumed_skipped"] == 0
    assert called_ids == [inserted_ids[1]]


def test_reject_then_reupload_same_file_skips_rejected_and_no_pending_created(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "reject-reupload-1",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/reject-reupload-1",
            "posted_at": "2026-02-23T12:00:00Z",
            "content_text": "reject then reupload",
        }
    ]

    imported = client.post("/ingest/x/import", json=payload)
    assert imported.status_code == 200
    raw_post_id = imported.json()["inserted_raw_post_ids"][0]

    pending = PostExtraction(
        id=901,
        raw_post_id=raw_post_id,
        status=ExtractionStatus.pending,
        extracted_json={"summary": "pending to reject"},
        model_name="dummy-v1",
        extractor_name="dummy",
        created_at=now,
    )
    fake_db.seed(pending)

    rejected = client.post(f"/extractions/{pending.id}/reject", json={"reason": "noise"})
    assert rejected.status_code == 200

    called_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "should_not_create"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    second_import = client.post("/ingest/x/import", json=payload)
    assert second_import.status_code == 200
    dedup_ids = second_import.json()["dedup_existing_raw_post_ids"]

    extract_resp = client.post("/raw-posts/extract-batch", json={"raw_post_ids": dedup_ids, "mode": "pending_or_failed"})
    assert extract_resp.status_code == 200
    body = extract_resp.json()
    assert body["success_count"] == 0
    assert body["failed_count"] == 0
    assert body["skipped_already_rejected_count"] == 1
    assert body["resumed_requested_count"] == 0
    assert body["resumed_success"] == 0
    assert body["resumed_failed"] == 0
    assert body["resumed_skipped"] == 0
    assert called_ids == []

    pending_list = client.get("/extractions?status=pending")
    app.dependency_overrides.clear()
    assert pending_list.status_code == 200
    assert all(item["raw_post"]["id"] != raw_post_id for item in pending_list.json())


def test_rejected_only_force_reextract_can_create_pending(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    raw_post = RawPost(
        id=1,
        platform="x",
        author_handle="alice",
        external_id="force-only-1",
        url="https://x.com/alice/status/force-only-1",
        content_text="rejected content",
        posted_at=now,
        fetched_at=now,
        raw_json=None,
        review_status=ReviewStatus.rejected,
        reviewed_at=now,
        reviewed_by="human-review",
    )
    fake_db.seed(raw_post)
    fake_db.seed(
        PostExtraction(
            id=910,
            raw_post_id=1,
            status=ExtractionStatus.rejected,
            extracted_json={"summary": "rejected"},
            model_name="dummy-v1",
            extractor_name="dummy",
            reviewed_at=now,
            reviewed_by="human-review",
            review_note="noise",
            created_at=now,
        )
    )

    calls: list[bool] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        calls.append(bool(kwargs.get("force_reextract")))
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "forced"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    normal_batch = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "pending_or_failed"})
    assert normal_batch.status_code == 200
    normal_body = normal_batch.json()
    assert normal_body["success_count"] == 0
    assert normal_body["skipped_already_rejected_count"] == 1
    assert calls == []

    forced = client.post("/extractions/910/re-extract")
    app.dependency_overrides.clear()
    assert forced.status_code == 201
    assert forced.json()["status"] == "pending"
    assert calls == [True]


def test_x_import_template_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/ingest/x/import/template")
    assert response.status_code == 200
    body = response.json()
    assert "external_id" in body["required_fields"]
    assert isinstance(body["example"], list)
    assert len(body["example"]) >= 1


def test_extract_batch_pending_only_and_force(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="x-1",
            url="https://x.com/alice/status/1",
            content_text="one",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="alice",
            external_id="x-2",
            url="https://x.com/alice/status/2",
            content_text="two",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=10,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "already"},
            model_name="dummy-v1",
            extractor_name="dummy",
            created_at=now,
        )
    )

    created_for_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        created_for_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "created"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    pending_only = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1, 2], "mode": "pending_only"})
    assert pending_only.status_code == 200
    p_body = pending_only.json()
    assert p_body["success_count"] == 1
    assert p_body["skipped_count"] == 1
    assert p_body["skipped_already_extracted_count"] == 1
    assert p_body["failed_count"] == 0
    assert created_for_ids == [1]

    force = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1, 2], "mode": "force"})
    app.dependency_overrides.clear()
    assert force.status_code == 200
    f_body = force.json()
    assert f_body["success_count"] == 2
    assert f_body["skipped_count"] == 0
    assert f_body["skipped_already_extracted_count"] == 0
    assert f_body["failed_count"] == 0
    assert created_for_ids == [1, 1, 2]


def test_extract_batch_pending_only_skips_when_already_successfully_extracted(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="x-1",
            url="https://x.com/alice/status/1",
            content_text="already extracted",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=10,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={"summary": "approved extraction"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )

    called = {"n": 0}

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called["n"] += 1
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "new"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "pending_only"})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["requested_count"] == 1
    assert body["success_count"] == 0
    assert body["skipped_count"] == 1
    assert body["skipped_already_extracted_count"] == 1
    assert body["failed_count"] == 0
    assert called["n"] == 0


def test_extract_batch_idempotent_for_same_raw_post_pending_reused(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="dup-1",
            url="https://x.com/alice/status/dup-1",
            content_text="dup target",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    monkeypatch.setenv("EXTRACTOR_MODE", "dummy")
    monkeypatch.setenv("AUTO_APPROVE_ENABLED", "false")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "pending_or_failed"})
    second = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "pending_or_failed"})
    app.dependency_overrides.clear()

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["success_count"] == 1

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["success_count"] == 0
    assert second_body["skipped_count"] == 1
    assert second_body["skipped_already_pending_count"] == 1
    assert second_body["skipped_already_success_count"] == 0

    active = [
        item
        for item in fake_db._data[PostExtraction].values()
        if item.raw_post_id == 1 and item.status == ExtractionStatus.pending and not (item.last_error or "").strip()
    ]
    assert len(active) == 1


def test_repeat_upload_resume_only_failed_or_no_extraction_skips_success_and_pending(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    payload = [
        {
            "external_id": "resume-all-1",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-all-1",
            "posted_at": "2026-02-23T09:00:00Z",
            "content_text": "approved",
        },
        {
            "external_id": "resume-all-2",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-all-2",
            "posted_at": "2026-02-23T09:01:00Z",
            "content_text": "pending",
        },
        {
            "external_id": "resume-all-3",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-all-3",
            "posted_at": "2026-02-23T09:02:00Z",
            "content_text": "failed",
        },
        {
            "external_id": "resume-all-4",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/resume-all-4",
            "posted_at": "2026-02-23T09:03:00Z",
            "content_text": "no extraction",
        },
    ]
    imported = client.post("/ingest/x/import", json=payload)
    assert imported.status_code == 200
    raw_ids = imported.json()["inserted_raw_post_ids"]
    assert len(raw_ids) == 4

    fake_db.seed(
        PostExtraction(
            id=300,
            raw_post_id=raw_ids[0],
            status=ExtractionStatus.approved,
            extracted_json={"summary": "approved"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=301,
            raw_post_id=raw_ids[1],
            status=ExtractionStatus.pending,
            extracted_json={"summary": "pending"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=302,
            raw_post_id=raw_ids[2],
            status=ExtractionStatus.pending,
            extracted_json={"summary": "failed"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="OpenAIRequestError: timeout",
            created_at=now,
        )
    )

    called_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "retried"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    second = client.post("/ingest/x/import", json=payload)
    assert second.status_code == 200
    dedup_ids = second.json()["dedup_existing_raw_post_ids"]
    assert len(dedup_ids) == 4

    extract_resp = client.post("/raw-posts/extract-batch", json={"raw_post_ids": dedup_ids, "mode": "pending_or_failed"})
    app.dependency_overrides.clear()

    assert extract_resp.status_code == 200
    body = extract_resp.json()
    assert body["requested_count"] == 4
    assert body["success_count"] == 2
    assert body["skipped_count"] == 2
    assert body["skipped_already_pending_count"] == 1
    assert body["skipped_already_approved_count"] == 1
    assert set(called_ids) == {raw_ids[2], raw_ids[3]}


def test_extract_batch_retries_429_and_records_last_error(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="x-1",
            url="https://x.com/alice/status/1",
            content_text="retry target",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    monkeypatch.setenv("EXTRACT_RETRY_MAX", "2")
    monkeypatch.setenv("EXTRACT_MAX_RPM_DEFAULT", "60")

    calls = {"n": 0}
    sleep_calls: list[float] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        raise OpenAIRequestError(status_code=429, body_preview="rate_limited")

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def fake_acquire(self) -> None:  # noqa: ANN001
        return None

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)
    monkeypatch.setattr("main.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("main._RpmLimiter.acquire", fake_acquire)
    monkeypatch.setattr("main._build_retry_backoff_seconds", lambda **kwargs: 0.25)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/raw-posts/extract-batch", json={"raw_post_ids": [1], "mode": "force"})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["requested_count"] == 1
    assert body["success_count"] == 0
    assert body["failed_count"] == 1
    assert calls["n"] == 3
    assert sleep_calls == [0.25, 0.25]
    saved_extractions = list(fake_db._data[PostExtraction].values())
    assert len(saved_extractions) == 1
    assert saved_extractions[0].last_error is not None
    assert "status=429" in saved_extractions[0].last_error


def test_post_extractions_active_unique_index_declared() -> None:
    index = next((idx for idx in PostExtraction.__table__.indexes if idx.name == "uq_post_extractions_active_raw_post_id"), None)
    assert index is not None
    assert index.unique
    where_clause = index.dialect_options["postgresql"].get("where")
    assert where_clause is not None
    assert "status = 'pending'" in str(where_clause)


def test_x_progress_counts_pending_failed_and_success() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="bob", display_name="Bob", enabled=True, created_at=now))
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p1",
            url="https://x.com/alice/status/1",
            content_text="a",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="alice",
            external_id="p2",
            url="https://x.com/alice/status/2",
            content_text="b",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        RawPost(
            id=3,
            platform="x",
            author_handle="bob",
            external_id="p3",
            url="https://x.com/bob/status/3",
            content_text="c",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=11,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=12,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "failed"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="OpenAIRequestError: timeout",
            created_at=now,
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    global_resp = client.get("/ingest/x/progress")
    handle_resp = client.get("/ingest/x/progress?author_handle=alice")
    app.dependency_overrides.clear()

    assert global_resp.status_code == 200
    global_body = global_resp.json()
    assert global_body["total_raw_posts"] == 3
    assert global_body["extracted_success_count"] == 1
    assert global_body["pending_count"] == 1
    assert global_body["failed_count"] == 1
    assert global_body["no_extraction_count"] == 1
    assert "timeout" in (global_body["latest_error_summary"] or "")

    assert handle_resp.status_code == 200
    handle_body = handle_resp.json()
    assert handle_body["scope"] == "author"
    assert handle_body["author_handle"] == "alice"
    assert handle_body["total_raw_posts"] == 2
    assert handle_body["failed_count"] == 1


def test_retry_failed_retries_only_failed_and_respects_limit(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p1",
            url="https://x.com/alice/status/1",
            content_text="a",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="alice",
            external_id="p2",
            url="https://x.com/alice/status/2",
            content_text="b",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=21,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "err1"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="error-1",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=22,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error=None,
            created_at=now,
        )
    )

    called_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post):  # noqa: ANN001
        called_ids.append(raw_post.id)
        if raw_post.id == 1:
            raise RuntimeError("retry boom")
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "retry"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/ingest/x/retry-failed?author_handle=alice&limit=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["retried_count"] == 1
    assert body["success_count"] == 0
    assert body["failed_count"] == 1
    assert called_ids == [1]
    assert body["failure_reasons"]["RuntimeError"] == 1


def test_import_kol_binding_with_kol_id_and_case_insensitive_match() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Kol(id=7, platform="x", handle="alice_std", display_name="Alice", enabled=True, created_at=now))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post(
        "/ingest/x/import?only_followed=true&allow_unknown_handles=false",
        json=[
            {
                "kol_id": 7,
                "external_id": "k1",
                "author_handle": "alice_any_case",
                "url": "https://x.com/alice/status/k1",
                "posted_at": "2026-02-23T08:00:00Z",
                "content_text": "via kol id",
            },
            {
                "external_id": "k2",
                "author_handle": "ALICE_STD",
                "url": "https://x.com/alice/status/k2",
                "posted_at": "2026-02-23T09:00:00Z",
                "content_text": "via handle match",
            },
            {
                "external_id": "k3",
                "author_handle": "unknown_user",
                "url": "https://x.com/unknown/status/k3",
                "posted_at": "2026-02-23T10:00:00Z",
                "content_text": "no kol match",
            },
        ],
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["inserted_raw_posts_count"] == 2
    assert body["warnings_count"] == 1
    assert body["skipped_not_followed_count"] == 1
    rows = list(fake_db._data[RawPost].values())
    by_external = {item.external_id: item for item in rows}
    assert by_external["k1"].author_handle == "alice_std"
    assert by_external["k2"].author_handle == "alice_std"
    assert "k3" not in by_external


def test_convert_endpoint_export_json_with_date_filter_and_kol_binding() -> None:
    client = TestClient(app)
    payload = {
        "tweets": [
            {
                "tweet_id": "3001",
                "created_at": "2026-02-20T10:00:00Z",
                "full_text": "inside range",
                "screen_name": "raw_handle",
            },
            {
                "tweet_id": "3002",
                "created_at": "2026-02-25T10:00:00Z",
                "full_text": "outside range",
                "screen_name": "raw_handle",
            },
        ]
    }
    response = client.post(
        "/ingest/x/convert?filename=export.json&author_handle=qinbafrank&kol_id=123&start_date=2026-02-19&end_date=2026-02-23",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["converted_rows"] == 2
    assert body["converted_ok"] == 1
    assert body["converted_failed"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["external_id"] == "3001"
    assert body["items"][0]["author_handle"] == "qinbafrank"
    assert body["items"][0]["kol_id"] == 123
    assert body["items"][0]["content_text"] == "inside range"
    assert body["items"][0]["posted_at"] == "2026-02-20T10:00:00Z"
    assert "out of range" in body["errors"][0]["reason"]


def test_convert_endpoint_accepts_standard_x_import_json() -> None:
    client = TestClient(app)
    standard = [
        {
            "external_id": "4001",
            "author_handle": "alice",
            "url": "https://x.com/alice/status/4001",
            "posted_at": "2026-02-20T10:00:00Z",
            "content_text": "already standard",
        }
    ]
    response = client.post(
        "/ingest/x/convert?filename=x_import.json",
        content=json.dumps(standard).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["converted_rows"] == 1
    assert body["converted_ok"] == 1
    assert body["converted_failed"] == 0
    assert len(body["items"]) == 1
    assert body["items"][0]["external_id"] == "4001"
    assert body["items"][0]["url"] == "https://x.com/alice/status/4001"
    assert body["items"][0]["content_text"] == "already standard"


def test_convert_endpoint_reports_row_errors_and_import_only_ok_rows() -> None:
    fake_db = FakeAsyncSession()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    payload = [
        {
            "tweet_id": "ok-1",
            "created_at": "2026-02-23T10:00:00Z",
            "screen_name": "alice",
            "full_text": "ok row",
            "url": "https://x.com/alice/status/ok-1",
        },
        {
            "created_at": "2026-02-23T10:01:00Z",
            "screen_name": "alice",
            "full_text": "missing id",
        },
        {
            "tweet_id": "bad-time",
            "created_at": "not-a-time",
            "screen_name": "alice",
            "full_text": "bad time row",
        },
        "not-an-object",
    ]

    convert_resp = client.post(
        "/ingest/x/convert?filename=broken.json",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert convert_resp.status_code == 200
    body = convert_resp.json()
    assert body["converted_rows"] == 4
    assert body["converted_ok"] == 1
    assert body["converted_failed"] == 3
    assert body["converted_ok"] + body["converted_failed"] == body["converted_rows"]
    assert len(body["errors"]) == 3
    assert any("missing external_id" in item["reason"] for item in body["errors"])
    assert any("invalid posted_at" in item["reason"] for item in body["errors"])
    assert any("row is not an object" in item["reason"] for item in body["errors"])

    import_resp = client.post("/ingest/x/import?only_followed=false&allow_unknown_handles=true", json=body["items"])
    app.dependency_overrides.clear()

    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["received_count"] == 1
    assert imported["inserted_raw_posts_count"] == 1
    assert len(fake_db._data[RawPost]) == 1


def test_admin_delete_pending_requires_confirm_yes() -> None:
    fake_db = FakeAsyncSession()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.delete("/admin/extractions/pending?confirm=NO")
    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "destructive operation" in response.json()["detail"]


def test_admin_delete_pending_forbidden_outside_local_dev(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("DEBUG", "false")

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.delete("/admin/extractions/pending?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 403
    assert "forbidden" in response.json()["detail"]


def test_admin_delete_pending_success_counts_and_author_scope() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    alice_post = RawPost(
        id=1,
        platform="x",
        kol_id=None,
        author_handle="alice",
        external_id="a-1",
        url="https://x.com/alice/status/a-1",
        content_text="alice pending",
        posted_at=now,
        fetched_at=now,
        raw_json=None,
    )
    bob_post = RawPost(
        id=2,
        platform="x",
        kol_id=None,
        author_handle="bob",
        external_id="b-1",
        url="https://x.com/bob/status/b-1",
        content_text="bob pending",
        posted_at=now,
        fetched_at=now,
        raw_json=None,
    )
    fake_db.seed(alice_post)
    fake_db.seed(bob_post)
    fake_db.seed(
        PostExtraction(
            id=11,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "alice"},
            model_name="dummy-v1",
            extractor_name="dummy",
            last_error="OpenAIRequestError: status=429",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=12,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "bob"},
            model_name="dummy-v1",
            extractor_name="dummy",
            created_at=now,
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.delete(
        "/admin/extractions/pending?confirm=YES&author_handle=alice&enable_cascade=true&also_delete_raw_posts=true"
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["deleted_extractions_count"] == 1
    assert body["deleted_raw_posts_count"] == 1
    assert body["scoped_author_handle"] == "alice"
    assert set(fake_db._data[RawPost].keys()) == {2}
    assert set(fake_db._data[PostExtraction].keys()) == {12}


def test_import_without_kol_auto_creates_kol_and_binds_raw_posts() -> None:
    fake_db = FakeAsyncSession()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    payload = [
        {
            "external_id": "auto-1",
            "author_handle": "  Alice_One ",
            "url": "https://x.com/alice_one/status/auto-1",
            "posted_at": "2026-02-23T09:00:00Z",
            "content_text": "hello",
        }
    ]
    response = client.post("/ingest/x/import?only_followed=false&allow_unknown_handles=true", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["resolved_author_handle"] == "alice_one"
    assert body["resolved_kol_id"] is not None
    assert body["kol_created"] is True
    rows = list(fake_db._data[RawPost].values())
    assert len(rows) == 1
    assert rows[0].author_handle == "alice_one"
    assert rows[0].kol_id == body["resolved_kol_id"]
    created_kols = list(fake_db._data[Kol].values())
    assert len(created_kols) == 1
    assert created_kols[0].handle == "alice_one"
    assert created_kols[0].enabled is True


def test_import_without_kol_multiple_handles_auto_creates_kols_and_import_shape() -> None:
    fake_db = FakeAsyncSession()

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    payload = [
        {
            "external_id": "mh-1",
            "author_handle": "Alice",
            "url": "https://x.com/alice/status/mh-1",
            "posted_at": "2026-02-23T09:00:00Z",
            "content_text": "a",
        },
        {
            "external_id": "mh-2",
            "author_handle": "Bob",
            "url": "https://x.com/bob/status/mh-2",
            "posted_at": "2026-02-23T09:01:00Z",
            "content_text": "b",
        },
    ]
    response = client.post("/ingest/x/import?only_followed=false&allow_unknown_handles=true", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["received_count"] == 2
    assert body["inserted_raw_posts_count"] == 2
    assert body["resolved_author_handle"] is None
    assert body["resolved_kol_id"] is None
    assert body["kol_created"] is True
    assert sorted(body["imported_by_handle"].keys()) == ["alice", "bob"]
    assert body["imported_by_handle"]["alice"]["inserted"] == 1
    assert body["imported_by_handle"]["bob"]["inserted"] == 1
    assert sorted(item["handle"] for item in body["created_kols"]) == ["alice", "bob"]
    assert all("kol" in key or "handle" in key or "created" in key or "raw_post" in key or "dedup" in key or "warnings" in key or "extract" in key or "received" in key or "inserted" in key or "imported" in key or "skipped" in key for key in body.keys())
    rows = list(fake_db._data[RawPost].values())
    assert len(rows) == 2
    by_external = {item.external_id: item for item in rows}
    assert by_external["mh-1"].author_handle == "alice"
    assert by_external["mh-2"].author_handle == "bob"
    created_kols = sorted(item.handle for item in fake_db._data[Kol].values())
    assert created_kols == ["alice", "bob"]
    for row in rows:
        assert row.kol_id is not None


def test_convert_without_overrides_multiple_handles_returns_handles_summary() -> None:
    client = TestClient(app)
    payload = {
        "tweets": [
            {
                "tweet_id": "c-1",
                "created_at": "2026-02-20T10:00:00Z",
                "full_text": "a",
                "screen_name": "Alice",
            },
            {
                "tweet_id": "c-2",
                "created_at": "2026-02-20T10:01:00Z",
                "full_text": "b",
                "screen_name": "Bob",
            },
        ]
    }
    response = client.post(
        "/ingest/x/convert?filename=mixed.json",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["converted_ok"] == 2
    assert body["converted_failed"] == 0
    assert sorted(item["author_handle"] for item in body["handles_summary"]) == ["alice", "bob"]
    assert all(item["count"] == 1 for item in body["handles_summary"])
    assert sorted(item["resolved_author_handle"] for item in body["items"]) == ["alice", "bob"]


def test_following_import_creates_and_updates_kols() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Kol(id=1, platform="x", handle="qinbafrank", display_name="Old", enabled=False, created_at=now))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    payload = [
        {"screen_name": "qinbafrank", "name": "Qinba", "following": True},
        {"screen_name": "new_handle", "name": "New One", "following": True},
        {"screen_name": "ignored_one", "name": "Ignored", "following": False},
    ]
    first = client.post(
        "/ingest/x/following/import?filename=twitter-正在关注-test.json",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    second = client.post(
        "/ingest/x/following/import?filename=twitter-正在关注-test.json",
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    app.dependency_overrides.clear()

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["received_rows"] == 3
    assert first_body["following_true_rows"] == 2
    assert first_body["created_kols_count"] == 1
    assert first_body["updated_kols_count"] == 1
    assert sorted(item["handle"] for item in first_body["created_kols"]) == ["new_handle"]
    assert sorted(item["handle"] for item in first_body["updated_kols"]) == ["qinbafrank"]
    stored = {item.handle: item for item in fake_db._data[Kol].values()}
    assert stored["qinbafrank"].enabled is True
    assert stored["new_handle"].enabled is True

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["created_kols_count"] == 0
    assert second_body["updated_kols_count"] == 2


def test_timeline_convert_filters_not_followed() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Kol(id=1, platform="x", handle="qinbafrank", display_name="Q", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="disabled_one", display_name="D", enabled=False, created_at=now))

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    timeline = [
        {
            "id": "t-1",
            "screen_name": "qinbafrank",
            "created_at": "2026-02-21 10:27:47 -05:00",
            "full_text": "followed",
            "url": "https://twitter.com/qinbafrank/status/t-1",
        },
        {
            "id": "t-2",
            "screen_name": "unknown_handle",
            "created_at": "2026-02-21 10:28:47 -05:00",
            "full_text": "unknown",
            "url": "https://twitter.com/unknown_handle/status/t-2",
        },
        {
            "id": "t-3",
            "screen_name": "disabled_one",
            "created_at": "2026-02-21 10:29:47 -05:00",
            "full_text": "disabled",
            "url": "https://twitter.com/disabled_one/status/t-3",
        },
    ]
    response = client.post(
        "/ingest/x/convert?filename=twitter-主页时间线-test.json&only_followed=true&allow_unknown_handles=false",
        content=json.dumps(timeline).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["converted_rows"] == 3
    assert body["converted_ok"] == 1
    assert body["skipped_not_followed_count"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["external_id"] == "t-1"
    assert body["items"][0]["author_handle"] == "qinbafrank"
    assert all(item["reason"] == "not_followed" for item in body["skipped_not_followed_samples"])


def test_timeline_import_only_followed_never_triggers_extraction_for_unknown(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    fake_db.seed(Kol(id=1, platform="x", handle="qinbafrank", display_name="Q", enabled=True, created_at=now))
    called_raw_post_ids: list[int] = []

    async def fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_raw_post_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "ok"},
            model_name="dummy-v1",
            extractor_name="dummy",
        )
        db.add(extraction)
        await db.flush()
        return extraction

    monkeypatch.setattr("main.create_pending_extraction", fake_create_pending_extraction)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    timeline = [
        {
            "id": "i-1",
            "screen_name": "qinbafrank",
            "created_at": "2026-02-21 10:27:47 -05:00",
            "full_text": "followed",
            "url": "https://twitter.com/qinbafrank/status/i-1",
        },
        {
            "id": "i-2",
            "screen_name": "ghost_user",
            "created_at": "2026-02-21 10:28:47 -05:00",
            "full_text": "unknown",
            "url": "https://twitter.com/ghost_user/status/i-2",
        },
    ]
    convert_resp = client.post(
        "/ingest/x/convert?filename=twitter-主页时间线-test.json&only_followed=true&allow_unknown_handles=false",
        content=json.dumps(timeline).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert convert_resp.status_code == 200
    rows = convert_resp.json()["items"]

    import_resp = client.post(
        "/ingest/x/import?trigger_extraction=true&only_followed=true&allow_unknown_handles=false",
        json=rows,
    )
    app.dependency_overrides.clear()

    assert import_resp.status_code == 200
    body = import_resp.json()
    assert body["inserted_raw_posts_count"] == 1
    assert body["skipped_not_followed_count"] == 0
    stored_posts = list(fake_db._data[RawPost].values())
    assert len(stored_posts) == 1
    assert stored_posts[0].author_handle == "qinbafrank"
    assert called_raw_post_ids == [stored_posts[0].id]
