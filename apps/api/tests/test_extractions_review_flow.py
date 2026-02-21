from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus
from main import app
from models import Asset, Kol, KolView, PostExtraction, RawPost


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
            Asset: {},
            Kol: {},
            KolView: {},
            RawPost: {},
            PostExtraction: {},
        }
        self._new: list[object] = []

    def seed(self, obj: object) -> None:
        obj_id = getattr(obj, "id")
        self._data[type(obj)][obj_id] = obj

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


def _seed_pending_extraction(db: FakeAsyncSession, *, extraction_id: int = 1) -> None:
    asset = Asset(id=1, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=datetime.now(UTC))
    kol = Kol(
        id=1,
        platform="x",
        handle="alice",
        display_name="Alice",
        enabled=True,
        created_at=datetime.now(UTC),
    )
    raw_post = RawPost(
        id=1,
        platform="x",
        author_handle="alice",
        external_id="post-1",
        url="https://x.com/alice/status/post-1",
        content_text="BTC may bounce this week.",
        posted_at=datetime(2026, 2, 21, 9, 0, tzinfo=UTC),
        fetched_at=datetime.now(UTC),
        raw_json=None,
    )
    extraction = PostExtraction(
        id=extraction_id,
        raw_post_id=raw_post.id,
        status=ExtractionStatus.pending,
        extracted_json={
            "summary": "Potential short-term rebound",
            "source_url": raw_post.url,
        },
        model_name="dummy-v1",
        created_at=datetime.now(UTC),
    )

    db.seed(asset)
    db.seed(kol)
    db.seed(raw_post)
    db.seed(extraction)


def test_approve_invalid_enum_returns_422() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_extraction(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    response = client.post(
        "/extractions/1/approve",
        json={
            "kol_id": 1,
            "asset_id": 1,
            "stance": "moon",
            "horizon": "10y",
            "confidence": 60,
            "summary": "invalid enum",
            "source_url": "https://example.com/view",
            "as_of": "2026-02-21",
        },
    )

    app.dependency_overrides.clear()
    assert response.status_code == 422


def test_approve_success_updates_extraction_and_asset_views() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_extraction(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    approve_response = client.post(
        "/extractions/1/approve",
        json={
            "kol_id": 1,
            "asset_id": 1,
            "stance": "bull",
            "horizon": "1w",
            "confidence": 72,
            "summary": "Momentum improving",
            "source_url": "https://x.com/alice/status/post-1",
            "as_of": "2026-02-21",
        },
    )

    assert approve_response.status_code == 200
    body = approve_response.json()
    assert body["status"] == "approved"
    assert body["applied_kol_view_id"] is not None

    views_response = client.get("/assets/1/views")
    app.dependency_overrides.clear()

    assert views_response.status_code == 200
    groups = views_response.json()["groups"]
    one_week = next(group for group in groups if group["horizon"] == "1w")
    assert len(one_week["bull"]) == 1
    assert one_week["bull"][0]["summary"] == "Momentum improving"


def test_reject_pending_success_and_reject_approved_conflict() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_extraction(fake_db, extraction_id=1)

    approved = PostExtraction(
        id=2,
        raw_post_id=1,
        status=ExtractionStatus.approved,
        extracted_json={"summary": "Already done"},
        model_name="dummy-v1",
        reviewed_at=datetime.now(UTC),
        reviewed_by="human-review",
        applied_kol_view_id=1,
        created_at=datetime.now(UTC),
    )
    fake_db.seed(approved)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    reject_pending = client.post("/extractions/1/reject", json={"reason": "noise"})
    assert reject_pending.status_code == 200
    assert reject_pending.json()["status"] == "rejected"

    reject_approved = client.post("/extractions/2/reject", json={"reason": "late"})
    app.dependency_overrides.clear()

    assert reject_approved.status_code == 409
