from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, Horizon, Stance
from main import app, reset_runtime_counters
from models import Asset, AssetAlias, DailyDigest, Kol, KolView, PostExtraction, ProfileKolWeight, RawPost
from settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    get_settings.cache_clear()
    reset_runtime_counters()


class FakeResult:
    def __init__(self, items: list[object] | None = None):
        self._items = items or []

    def scalars(self) -> "FakeResult":
        return self

    def all(self) -> list[object]:
        return self._items


class FakeAsyncSession:
    def __init__(self) -> None:
        self._data: dict[type[object], dict[int, object]] = {
            Asset: {},
            AssetAlias: {},
            Kol: {},
            KolView: {},
            RawPost: {},
            PostExtraction: {},
            DailyDigest: {},
            ProfileKolWeight: {},
        }
        self._new: list[object] = []
        self._to_delete: list[object] = []

    def seed(self, obj: object) -> None:
        self._data[type(obj)][getattr(obj, "id")] = obj

    def add(self, obj: object) -> None:
        self._new.append(obj)

    async def get(self, model: type[object], obj_id: int) -> object | None:
        return self._data.get(model, {}).get(obj_id)

    async def execute(self, query):  # noqa: ANN001
        entity = query.column_descriptions[0]["entity"]
        items = list(self._data.get(entity, {}).values())
        return FakeResult(items=items)

    async def delete(self, obj: object) -> None:
        self._to_delete.append(obj)

    async def flush(self) -> None:
        self._persist_new()
        self._apply_delete()

    async def commit(self) -> None:
        self._persist_new()
        self._apply_delete()

    async def refresh(self, obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None

    def _persist_new(self) -> None:
        now = datetime.now(UTC)
        for obj in self._new:
            model = type(obj)
            bucket = self._data[model]
            if getattr(obj, "id", None) is None:
                setattr(obj, "id", max(bucket.keys(), default=0) + 1)
            if hasattr(obj, "created_at") and getattr(obj, "created_at") is None:
                setattr(obj, "created_at", now)
            if hasattr(obj, "fetched_at") and getattr(obj, "fetched_at") is None:
                setattr(obj, "fetched_at", now)
            if hasattr(obj, "generated_at") and getattr(obj, "generated_at") is None:
                setattr(obj, "generated_at", now)
            bucket[getattr(obj, "id")] = obj
        self._new.clear()

    def _apply_delete(self) -> None:
        for obj in self._to_delete:
            model = type(obj)
            obj_id = getattr(obj, "id", None)
            bucket = self._data.get(model, {})
            if obj_id not in bucket:
                continue

            if model is RawPost:
                extraction_bucket = self._data[PostExtraction]
                for extraction_id, extraction in list(extraction_bucket.items()):
                    if extraction.raw_post_id == obj_id:
                        del extraction_bucket[extraction_id]

            if model is Kol:
                raw_bucket = self._data[RawPost]
                for raw_post in raw_bucket.values():
                    if raw_post.kol_id == obj_id:
                        raw_post.kol_id = None

                view_bucket = self._data[KolView]
                for view_id, view in list(view_bucket.items()):
                    if view.kol_id == obj_id:
                        del view_bucket[view_id]

                profile_weight_bucket = self._data[ProfileKolWeight]
                for weight_id, item in list(profile_weight_bucket.items()):
                    if item.kol_id == obj_id:
                        del profile_weight_bucket[weight_id]

            if model is Asset:
                alias_bucket = self._data[AssetAlias]
                for alias_id, alias in list(alias_bucket.items()):
                    if alias.asset_id == obj_id:
                        del alias_bucket[alias_id]

                view_bucket = self._data[KolView]
                for view_id, view in list(view_bucket.items()):
                    if view.asset_id == obj_id:
                        del view_bucket[view_id]

            if model is KolView:
                extraction_bucket = self._data[PostExtraction]
                for extraction in extraction_bucket.values():
                    if extraction.applied_kol_view_id == obj_id:
                        extraction.applied_kol_view_id = None

            del bucket[obj_id]

        self._to_delete.clear()


def _client_with_db(fake_db: FakeAsyncSession) -> TestClient:
    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_admin_delete_digest_by_date_and_profile() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    target_date = date(2026, 2, 24)

    fake_db.seed(DailyDigest(id=1, profile_id=1, digest_date=target_date, version=1, days=7, content={}, generated_at=now))
    fake_db.seed(DailyDigest(id=2, profile_id=1, digest_date=date(2026, 2, 23), version=1, days=7, content={}, generated_at=now))
    fake_db.seed(DailyDigest(id=3, profile_id=2, digest_date=target_date, version=1, days=7, content={}, generated_at=now))

    client = _client_with_db(fake_db)
    response = client.delete("/admin/digests?confirm=YES&digest_date=2026-02-24&profile_id=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["counts"]["daily_digests"] == 1
    assert set(fake_db._data[DailyDigest].keys()) == {2, 3}


def test_admin_delete_kol_default_derived_only_keeps_raw_posts() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="bob", display_name="Bob", enabled=True, created_at=now))

    fake_db.seed(RawPost(id=11, platform="x", kol_id=1, author_handle="alice", external_id="a-1", url="https://x.com/alice/1", content_text="a", posted_at=now, fetched_at=now, raw_json=None))
    fake_db.seed(RawPost(id=12, platform="x", kol_id=2, author_handle="bob", external_id="b-1", url="https://x.com/bob/1", content_text="b", posted_at=now, fetched_at=now, raw_json=None))

    fake_db.seed(PostExtraction(id=101, raw_post_id=11, status=ExtractionStatus.pending, extracted_json={}, model_name="dummy", extractor_name="dummy", created_at=now))
    fake_db.seed(PostExtraction(id=102, raw_post_id=12, status=ExtractionStatus.pending, extracted_json={}, model_name="dummy", extractor_name="dummy", created_at=now))

    fake_db.seed(KolView(id=201, kol_id=1, asset_id=1, stance=Stance.bull, horizon=Horizon.one_week, confidence=70, summary="x", source_url="https://x.com/alice/1", as_of=now.date(), created_at=now))
    fake_db.seed(KolView(id=202, kol_id=2, asset_id=1, stance=Stance.bull, horizon=Horizon.one_week, confidence=70, summary="y", source_url="https://x.com/bob/1", as_of=now.date(), created_at=now))

    fake_db.seed(
        DailyDigest(
            id=301,
            profile_id=1,
            digest_date=now.date(),
            version=1,
            days=7,
            content={"per_asset_summary": [{"top_views_bull": [{"kol_id": 1}], "top_views_bear": [], "top_views_neutral": []}]},
            generated_at=now,
        )
    )

    client = _client_with_db(fake_db)
    response = client.delete("/admin/kols/1?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["derived_only"] is True
    assert body["counts"]["post_extractions"] == 1
    assert body["counts"]["kol_views"] == 1
    assert body["counts"]["raw_posts"] == 0
    assert 11 in fake_db._data[RawPost]
    assert fake_db._data[RawPost][11].kol_id is None
    assert 101 not in fake_db._data[PostExtraction]
    assert 102 in fake_db._data[PostExtraction]


def test_admin_delete_kol_cascade_requires_enable_cascade_for_raw_posts() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(RawPost(id=11, platform="x", kol_id=1, author_handle="alice", external_id="a-1", url="https://x.com/alice/1", content_text="a", posted_at=now, fetched_at=now, raw_json=None))
    fake_db.seed(PostExtraction(id=101, raw_post_id=11, status=ExtractionStatus.pending, extracted_json={}, model_name="dummy", extractor_name="dummy", created_at=now))

    client = _client_with_db(fake_db)
    blocked = client.delete("/admin/kols/1?confirm=YES&also_delete_raw_posts=true")
    assert blocked.status_code == 403
    assert 11 in fake_db._data[RawPost]
    assert 101 in fake_db._data[PostExtraction]

    allowed = client.delete("/admin/kols/1?confirm=YES&enable_cascade=true&also_delete_raw_posts=true")
    app.dependency_overrides.clear()

    assert allowed.status_code == 200
    body = allowed.json()
    assert body["counts"]["raw_posts"] == 1
    assert 11 not in fake_db._data[RawPost]
    assert 101 not in fake_db._data[PostExtraction]


def test_admin_delete_asset_default_derived_only_deletes_related_views_only() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Asset(id=2, symbol="ETH", name="Ethereum", market="CRYPTO", created_at=now))
    fake_db.seed(AssetAlias(id=31, asset_id=1, alias="bitcoin", created_at=now))

    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(KolView(id=11, kol_id=1, asset_id=1, stance=Stance.bull, horizon=Horizon.one_week, confidence=80, summary="btc", source_url="https://x.com/alice/11", as_of=now.date(), created_at=now))
    fake_db.seed(KolView(id=12, kol_id=1, asset_id=2, stance=Stance.bull, horizon=Horizon.one_week, confidence=70, summary="eth", source_url="https://x.com/alice/12", as_of=now.date(), created_at=now))

    fake_db.seed(PostExtraction(id=101, raw_post_id=999, status=ExtractionStatus.approved, extracted_json={}, model_name="dummy", extractor_name="dummy", applied_kol_view_id=11, created_at=now))

    client = _client_with_db(fake_db)
    response = client.delete("/admin/assets/1?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["derived_only"] is True
    assert body["counts"]["kol_views"] == 1
    assert body["counts"]["assets"] == 0
    assert 1 in fake_db._data[Asset]
    assert 31 in fake_db._data[AssetAlias]
    assert 11 not in fake_db._data[KolView]
    assert 12 in fake_db._data[KolView]
    assert fake_db._data[PostExtraction][101].applied_kol_view_id is None
