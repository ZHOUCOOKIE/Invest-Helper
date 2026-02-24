from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import Horizon, Stance
from main import app, reset_runtime_counters
from models import Asset, DailyDigest, Kol, KolView, ProfileKolWeight, ProfileMarket, RawPost, UserProfile
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
            Kol: {},
            KolView: {},
            DailyDigest: {},
            UserProfile: {},
            ProfileKolWeight: {},
            ProfileMarket: {},
            RawPost: {},
        }
        self._new: list[object] = []
        self._to_delete: list[object] = []

    def seed(self, obj: object) -> None:
        self._data[type(obj)][getattr(obj, "id")] = obj

    def add(self, obj: object) -> None:
        self._new.append(obj)

    async def delete(self, obj: object) -> None:
        self._to_delete.append(obj)

    async def get(self, model: type[object], obj_id: int) -> object | None:
        return self._data.get(model, {}).get(obj_id)

    async def execute(self, query):  # noqa: ANN001
        entity = query.column_descriptions[0]["entity"]
        items = list(self._data.get(entity, {}).values())
        for criterion in getattr(query, "_where_criteria", ()):
            key = criterion.left.key
            value = criterion.right.value
            operator_name = getattr(getattr(criterion, "operator", None), "__name__", "eq")
            if operator_name == "eq":
                items = [item for item in items if getattr(item, key) == value]
            elif operator_name == "ge":
                items = [item for item in items if getattr(item, key) >= value]
            elif operator_name == "le":
                items = [item for item in items if getattr(item, key) <= value]
            else:
                items = [item for item in items if getattr(item, key) == value]

        if entity is KolView:
            items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        elif entity is DailyDigest:
            items.sort(key=lambda item: (item.digest_date, item.version, item.id), reverse=True)
        elif entity is RawPost:
            items.sort(key=lambda item: (item.posted_at, item.id), reverse=True)
        else:
            items.sort(key=lambda item: item.id)
        return FakeResult(items)

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

    def _apply_delete(self) -> None:
        for obj in self._to_delete:
            bucket = self._data.get(type(obj), {})
            obj_id = getattr(obj, "id", None)
            if obj_id in bucket:
                del bucket[obj_id]
        self._to_delete.clear()

    def _persist_new(self) -> None:
        now = datetime.now(UTC)
        for obj in self._new:
            model = type(obj)
            bucket = self._data[model]
            if getattr(obj, "id", None) is None:
                setattr(obj, "id", max(bucket.keys(), default=0) + 1)
            if hasattr(obj, "created_at") and getattr(obj, "created_at") is None:
                setattr(obj, "created_at", now)
            if hasattr(obj, "generated_at") and getattr(obj, "generated_at") is None:
                setattr(obj, "generated_at", now)
            bucket[getattr(obj, "id")] = obj
        self._new.clear()


def _client_with_db(fake_db: FakeAsyncSession) -> TestClient:
    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_generate_digest_and_get_latest_and_version_and_by_id() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    today = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Asset(id=2, symbol="ETH", name="Ethereum", market="CRYPTO", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="bob", display_name="Bob", enabled=True, created_at=now))

    fake_db.seed(
        KolView(
            id=11,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=92,
            summary="BTC momentum strong",
            source_url="https://x.com/alice/11",
            as_of=today,
            created_at=now - timedelta(hours=2),
        )
    )
    fake_db.seed(
        KolView(
            id=12,
            kol_id=2,
            asset_id=1,
            stance=Stance.bear,
            horizon=Horizon.one_week,
            confidence=70,
            summary="BTC may pull back",
            source_url="https://x.com/bob/12",
            as_of=today,
            created_at=now - timedelta(hours=3),
        )
    )

    client = _client_with_db(fake_db)

    first = client.post(f"/digests/generate?date={today.isoformat()}&days=7")
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["profile_id"] == 1
    assert first_body["version"] == 1
    assert first_body["metadata"]["time_field_used"] == "as_of"

    second = client.post(f"/digests/generate?date={today.isoformat()}&days=7")
    assert second.status_code == 200
    assert second.json()["version"] == 2

    latest = client.get(f"/digests?date={today.isoformat()}")
    assert latest.status_code == 200
    digest_id = latest.json()["id"]

    by_id = client.get(f"/digests/{digest_id}")
    assert by_id.status_code == 200
    assert by_id.json()["id"] == digest_id

    app.dependency_overrides.clear()


def test_digest_uses_as_of_window_over_created_at() -> None:
    fake_db = FakeAsyncSession()
    to_ts = datetime(2026, 2, 23, 12, 0, 0, tzinfo=UTC)
    digest_date = to_ts.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=to_ts))
    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=to_ts))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=to_ts))

    fake_db.seed(
        KolView(
            id=1,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=90,
            summary="old as_of, new created_at",
            source_url="https://x.com/alice/1",
            as_of=date(2026, 1, 1),
            created_at=to_ts - timedelta(hours=2),
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(
        "/digests/generate",
        params={
            "date": digest_date.isoformat(),
            "days": 7,
            "to_ts": to_ts.isoformat(),
        },
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["time_field_used"] == "as_of"
    assert body["top_assets"] == []


def test_digest_weighted_sort_and_top_view_weighted_score() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 2, 23, 12, 0, 0, tzinfo=UTC)
    today = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Asset(id=2, symbol="ETH", name="Ethereum", market="CRYPTO", created_at=now))

    fake_db.seed(Kol(id=1, platform="x", handle="a", display_name="A", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="b", display_name="B", enabled=True, created_at=now))

    fake_db.seed(ProfileKolWeight(id=1, profile_id=1, kol_id=1, weight=2.0, enabled=True, created_at=now))
    fake_db.seed(ProfileKolWeight(id=2, profile_id=1, kol_id=2, weight=0.5, enabled=True, created_at=now))

    fake_db.seed(
        KolView(
            id=11,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=60,
            summary="btc by heavy kol",
            source_url="https://x.com/a/11",
            as_of=today,
            created_at=now - timedelta(hours=1),
        )
    )
    fake_db.seed(
        KolView(
            id=12,
            kol_id=2,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=99,
            summary="btc by light kol",
            source_url="https://x.com/b/12",
            as_of=today,
            created_at=now - timedelta(hours=2),
        )
    )
    fake_db.seed(
        KolView(
            id=21,
            kol_id=2,
            asset_id=2,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=95,
            summary="eth by light kol",
            source_url="https://x.com/b/21",
            as_of=today,
            created_at=now - timedelta(hours=1),
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(
        "/digests/generate",
        params={"date": today.isoformat(), "days": 7, "to_ts": now.isoformat(), "profile_id": 1},
    )
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["top_assets"][0]["symbol"] == "BTC"
    assert body["top_assets"][0]["weighted_views_24h"] > body["top_assets"][1]["weighted_views_24h"]

    btc_summary = next(item for item in body["per_asset_summary"] if item["symbol"] == "BTC")
    assert btc_summary["top_views_bull"][0]["kol_handle"] == "a"
    assert btc_summary["top_views_bull"][0]["weighted_score"] > btc_summary["top_views_bull"][1]["weighted_score"]


def test_digest_profile_filters_kols_and_markets() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 2, 23, 12, 0, 0, tzinfo=UTC)
    today = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(UserProfile(id=2, name="crypto-only", created_at=now))

    fake_db.seed(Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(Asset(id=2, symbol="AAPL", name="Apple", market="US", created_at=now))

    fake_db.seed(Kol(id=1, platform="x", handle="a", display_name="A", enabled=True, created_at=now))
    fake_db.seed(Kol(id=2, platform="x", handle="b", display_name="B", enabled=True, created_at=now))

    fake_db.seed(ProfileKolWeight(id=11, profile_id=2, kol_id=1, weight=1.5, enabled=True, created_at=now))
    fake_db.seed(ProfileKolWeight(id=12, profile_id=2, kol_id=2, weight=1.0, enabled=False, created_at=now))
    fake_db.seed(ProfileMarket(id=21, profile_id=2, market="CRYPTO", created_at=now))

    fake_db.seed(
        KolView(
            id=31,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=80,
            summary="btc allowed",
            source_url="https://x.com/a/31",
            as_of=today,
            created_at=now - timedelta(hours=2),
        )
    )
    fake_db.seed(
        KolView(
            id=32,
            kol_id=2,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=88,
            summary="btc by disabled kol",
            source_url="https://x.com/b/32",
            as_of=today,
            created_at=now - timedelta(hours=2),
        )
    )
    fake_db.seed(
        KolView(
            id=33,
            kol_id=1,
            asset_id=2,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=90,
            summary="us asset filtered",
            source_url="https://x.com/a/33",
            as_of=today,
            created_at=now - timedelta(hours=2),
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(
        "/digests/generate",
        params={"date": today.isoformat(), "days": 7, "to_ts": now.isoformat(), "profile_id": 2},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == 2
    assert [item["symbol"] for item in body["top_assets"]] == ["BTC"]

    load_response = client.get("/digests", params={"date": today.isoformat(), "profile_id": 2})
    assert load_response.status_code == 200
    assert load_response.json()["profile_id"] == 2

    app.dependency_overrides.clear()
