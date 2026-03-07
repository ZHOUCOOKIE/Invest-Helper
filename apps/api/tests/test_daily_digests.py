from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from db import get_db
from enums import ExtractionStatus
from main import app, reset_runtime_counters
from models import DailyDigest, Kol, PostExtraction, ProfileKolWeight, RawPost, UserProfile, WeeklyDigest
from services.digests import _author_summary_field_guide_text
from settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
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
            DailyDigest: {},
            WeeklyDigest: {},
            Kol: {},
            PostExtraction: {},
            ProfileKolWeight: {},
            RawPost: {},
            UserProfile: {},
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

        if entity is PostExtraction:
            items.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        elif entity is DailyDigest:
            items.sort(key=lambda item: (item.digest_date, item.id), reverse=True)
        elif entity is WeeklyDigest:
            items.sort(key=lambda item: (item.anchor_date, item.id), reverse=True)
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


def test_generate_digest_uses_latest_hasview_extraction_only() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    today = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))

    fake_db.seed(
        RawPost(
            id=11,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p1",
            url="https://x.com/alice/1",
            content_text="raw content one",
            posted_at=now - timedelta(hours=4),
            fetched_at=now,
            raw_json={"title": "title 1"},
        )
    )
    fake_db.seed(
        PostExtraction(
            id=101,
            raw_post_id=11,
            status=ExtractionStatus.approved,
            extracted_json={
                "as_of": today.isoformat(),
                "hasview": 0,
                "summary": "older summary should not be used",
                "asset_views": [],
            },
            model_name="dummy",
            extractor_name="dummy",
            created_at=now - timedelta(hours=1),
        )
    )

    # same raw_post newer rejected but hasview=1 => this post should be included
    fake_db.seed(
        PostExtraction(
            id=102,
            raw_post_id=11,
            status=ExtractionStatus.rejected,
            extracted_json={"as_of": today.isoformat(), "hasview": 1, "summary": "newer hasview summary"},
            model_name="dummy",
            extractor_name="dummy",
            created_at=now,
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(f"/digests/generate?date={today.isoformat()}&profile_id=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["digest_date"] == today.isoformat()
    assert len(body["post_summaries"]) == 1
    assert body["post_summaries"][0]["extraction_id"] == 102
    assert body["post_summaries"][0]["summary"] == "newer hasview summary"
    assert body["metadata"]["source_post_count"] == 1


def test_generate_digest_window_and_time_priority_sorting() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    digest_date = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))

    # row A: as_of => business_ts=2026-03-05 00:00 UTC
    fake_db.seed(
        RawPost(
            id=21,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p21",
            url="https://x.com/alice/21",
            content_text="content a",
            posted_at=datetime(2026, 3, 5, 6, 0, 0, tzinfo=UTC),
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=201,
            raw_post_id=21,
            status=ExtractionStatus.approved,
            extracted_json={"as_of": "2026-03-05", "hasview": 1, "summary": "A"},
            model_name="dummy",
            extractor_name="dummy",
            created_at=now - timedelta(hours=6),
        )
    )

    # row B: no as_of => use posted_at=2026-03-06 03:00 UTC
    fake_db.seed(
        RawPost(
            id=22,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p22",
            url="https://x.com/alice/22",
            content_text="content b",
            posted_at=datetime(2026, 3, 6, 3, 0, 0, tzinfo=UTC),
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=202,
            raw_post_id=22,
            status=ExtractionStatus.approved,
            extracted_json={"hasview": 1, "summary": "B", "asset_views": [{"summary": "B1"}]},
            model_name="dummy",
            extractor_name="dummy",
            created_at=now - timedelta(hours=5),
        )
    )

    # row C: no as_of and no posted_at => fallback created_at
    fake_db.seed(
        RawPost(
            id=23,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p23",
            url="https://x.com/alice/23",
            content_text="content c",
            posted_at=datetime(2026, 3, 7, 1, 0, 0, tzinfo=UTC),  # outside window but should be ignored by posted_at if used
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=203,
            raw_post_id=23,
            status=ExtractionStatus.approved,
            extracted_json={"as_of": "", "hasview": 1, "summary": "C"},
            model_name="dummy",
            extractor_name="dummy",
            created_at=datetime(2026, 3, 6, 5, 0, 0, tzinfo=UTC),
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(f"/digests/generate?date={digest_date.isoformat()}&profile_id=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    rows = body["post_summaries"]

    # row 23 excluded because posted_at is preferred over created_at and posted_at is out of window
    assert [item["raw_post_id"] for item in rows] == [21, 22]
    assert rows[0]["time_field_used"] == "as_of"
    assert rows[1]["time_field_used"] == "posted_at"


def test_generate_digest_replaces_same_profile_date() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    digest_date = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        RawPost(
            id=31,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p31",
            url="https://x.com/alice/31",
            content_text="content",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=301,
            raw_post_id=31,
            status=ExtractionStatus.approved,
            extracted_json={"as_of": digest_date.isoformat(), "hasview": 1, "summary": "v1"},
            model_name="dummy",
            extractor_name="dummy",
            created_at=now,
        )
    )

    client = _client_with_db(fake_db)
    first = client.post(f"/digests/generate?date={digest_date.isoformat()}&profile_id=1")
    extraction = fake_db._data[PostExtraction][301]
    extraction.extracted_json = {"as_of": digest_date.isoformat(), "hasview": 1, "summary": "v2"}
    extraction.created_at = now + timedelta(minutes=1)
    second = client.post(f"/digests/generate?date={digest_date.isoformat()}&profile_id=1")
    app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake_db._data[DailyDigest]) == 1
    assert second.json()["post_summaries"][0]["summary"] == "v2"


def test_get_digest_not_found() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))

    client = _client_with_db(fake_db)
    response = client.get("/digests", params={"date": date(2026, 3, 6).isoformat(), "profile_id": 1})
    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "digest not found"


def test_generate_digest_content_is_json_safe_before_persist() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    digest_date = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        RawPost(
            id=41,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p41",
            url="https://x.com/alice/41",
            content_text="content",
            posted_at=now - timedelta(hours=1),
            fetched_at=now,
            raw_json={"title": "title"},
        )
    )
    fake_db.seed(
        PostExtraction(
            id=401,
            raw_post_id=41,
            status=ExtractionStatus.approved,
            extracted_json={"as_of": digest_date.isoformat(), "hasview": 1, "summary": "summary"},
            model_name="dummy",
            extractor_name="dummy",
            created_at=now,
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(f"/digests/generate?date={digest_date.isoformat()}&profile_id=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    stored = next(iter(fake_db._data[DailyDigest].values()))
    metadata = stored.content["metadata"]
    assert isinstance(metadata["generated_at"], str)
    assert isinstance(metadata["window_start"], str)
    assert isinstance(metadata["window_end"], str)
    json.dumps(stored.content, ensure_ascii=False)


def test_generate_digest_returns_json_when_service_raises(monkeypatch) -> None:  # noqa: ANN001
    async def _boom(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("digest route boom")

    monkeypatch.setattr(main_module, "generate_daily_digest_service", _boom)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/digests/generate?date=2026-03-06&profile_id=1")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert isinstance(body.get("request_id"), str)
    assert body.get("error_code") == "digest_generate_failed"
    assert body.get("message") == "Generate digest failed"
    assert body.get("detail") == "Generate digest failed"
    assert "digest route boom" not in str(body.get("detail"))


def test_generate_digest_ai_author_summaries_include_asset_fields() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 6, 12, 0, 0, tzinfo=UTC)
    digest_date = now.date()

    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        RawPost(
            id=51,
            platform="x",
            kol_id=1,
            author_handle="alice",
            external_id="p51",
            url="https://x.com/alice/51",
            content_text="content",
            posted_at=now - timedelta(hours=1),
            fetched_at=now,
            raw_json={"title": "title"},
        )
    )
    fake_db.seed(
        PostExtraction(
            id=501,
            raw_post_id=51,
            status=ExtractionStatus.approved,
            extracted_json={
                "as_of": digest_date.isoformat(),
                "hasview": 1,
                "summary": "fallback summary",
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "market": "CRYPTO",
                        "stance": "bull",
                        "horizon": "1w",
                        "summary": "btc summary",
                    }
                ],
            },
            model_name="dummy",
            extractor_name="dummy",
            created_at=now,
        )
    )

    client = _client_with_db(fake_db)
    response = client.post(f"/digests/generate?date={digest_date.isoformat()}&profile_id=1")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert len(body["ai_input_by_author"]) == 1
    summaries = body["ai_input_by_author"][0]["summaries"]
    assert len(summaries) == 1
    assert summaries[0]["symbol"] == "BTC"
    assert summaries[0]["market"] == "CRYPTO"
    assert summaries[0]["stance"] == "bull"
    assert summaries[0]["horizon"] == "1w"
    assert summaries[0]["summary"] == "btc summary"


def test_author_summary_field_guide_mentions_core_fields() -> None:
    text = _author_summary_field_guide_text()
    assert "symbol" in text
    assert "market" in text
    assert "stance" in text
    assert "horizon" in text
    assert "bull=看多" in text
    assert "bear=看空" in text
    assert "neutral=中性" in text
    assert "summary" not in text
    assert "source_url" not in text
    assert "title" not in text


def test_list_digest_dates_keeps_recent_3_days_and_purges_older() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    today = now.date()
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(DailyDigest(id=1, profile_id=1, digest_date=today, version=1, days=2, content={}, generated_at=now))
    fake_db.seed(
        DailyDigest(id=2, profile_id=1, digest_date=today - timedelta(days=1), version=1, days=2, content={}, generated_at=now)
    )
    fake_db.seed(
        DailyDigest(id=3, profile_id=1, digest_date=today - timedelta(days=2), version=1, days=2, content={}, generated_at=now)
    )
    fake_db.seed(
        DailyDigest(id=4, profile_id=1, digest_date=today - timedelta(days=3), version=1, days=2, content={}, generated_at=now)
    )

    client = _client_with_db(fake_db)
    response = client.get("/digests/dates")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == [today.isoformat(), (today - timedelta(days=1)).isoformat(), (today - timedelta(days=2)).isoformat()]
    assert set(fake_db._data[DailyDigest].keys()) == {1, 2, 3}


def test_get_digest_out_of_recent_3_days_returns_404() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    today = now.date()
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(
        DailyDigest(id=1, profile_id=1, digest_date=today - timedelta(days=3), version=1, days=2, content={}, generated_at=now)
    )

    client = _client_with_db(fake_db)
    response = client.get("/digests", params={"date": (today - timedelta(days=3)).isoformat()})
    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "digest not found"


def test_generate_digest_rejects_out_of_recent_3_days() -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)
    today = now.date()
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))

    client = _client_with_db(fake_db)
    response = client.post(f"/digests/generate?date={(today - timedelta(days=3)).isoformat()}")
    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert response.json()["detail"] == "digest_date must be within recent 3 days"


def test_generate_weekly_digest_this_week_window_starts_from_latest_sunday() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)  # Wednesday
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))

    for idx, as_of in enumerate(["2026-03-08", "2026-03-10", "2026-03-07"], start=1):
        fake_db.seed(
            RawPost(
                id=600 + idx,
                platform="x",
                kol_id=1,
                author_handle="alice",
                external_id=f"w-{idx}",
                url=f"https://x.com/alice/w{idx}",
                content_text=f"content {idx}",
                posted_at=now,
                fetched_at=now,
                raw_json=None,
            )
        )
        fake_db.seed(
            PostExtraction(
                id=700 + idx,
                raw_post_id=600 + idx,
                status=ExtractionStatus.approved,
                extracted_json={"as_of": as_of, "hasview": 1, "summary": f"summary {idx}"},
                model_name="dummy",
                extractor_name="dummy",
                created_at=now,
            )
        )

    client = _client_with_db(fake_db)
    response = client.post("/weekly-digests/generate?kind=this_week&date=2026-03-11")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["report_kind"] == "this_week"
    assert body["anchor_date"] == "2026-03-08"
    assert body["metadata"]["window_start"].startswith("2026-03-08")
    assert body["metadata"]["window_end"].startswith("2026-03-12")
    # 2026-03-07 should be excluded for this_week
    assert {item["raw_post_id"] for item in body["post_summaries"]} == {601, 602}


def test_generate_weekly_digest_last_week_window_and_dates_listing() -> None:
    fake_db = FakeAsyncSession()
    now = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
    fake_db.seed(UserProfile(id=1, name="default", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))

    for idx, as_of in enumerate(["2026-03-01", "2026-03-07", "2026-03-08"], start=1):
        fake_db.seed(
            RawPost(
                id=800 + idx,
                platform="x",
                kol_id=1,
                author_handle="alice",
                external_id=f"lw-{idx}",
                url=f"https://x.com/alice/lw{idx}",
                content_text=f"last week {idx}",
                posted_at=now,
                fetched_at=now,
                raw_json=None,
            )
        )
        fake_db.seed(
            PostExtraction(
                id=900 + idx,
                raw_post_id=800 + idx,
                status=ExtractionStatus.approved,
                extracted_json={"as_of": as_of, "hasview": 1, "summary": f"last week summary {idx}"},
                model_name="dummy",
                extractor_name="dummy",
                created_at=now,
            )
        )

    client = _client_with_db(fake_db)
    generated = client.post("/weekly-digests/generate?kind=last_week&date=2026-03-11")
    dates = client.get("/weekly-digests/dates?kind=last_week")
    fetched = client.get("/weekly-digests?kind=last_week&anchor_date=2026-03-01")
    app.dependency_overrides.clear()

    assert generated.status_code == 200
    body = generated.json()
    assert body["anchor_date"] == "2026-03-01"
    assert body["metadata"]["window_start"].startswith("2026-03-01")
    assert body["metadata"]["window_end"].startswith("2026-03-08")
    # 2026-03-08 belongs to this_week and should be excluded for last_week
    assert {item["raw_post_id"] for item in body["post_summaries"]} == {801, 802}

    assert dates.status_code == 200
    assert dates.json() == ["2026-03-01"]
    assert fetched.status_code == 200
    assert fetched.json()["report_kind"] == "last_week"
