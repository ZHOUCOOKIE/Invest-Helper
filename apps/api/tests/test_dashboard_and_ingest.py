from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, Horizon, Stance
from main import app
from models import Asset, Kol, KolView, PostExtraction, RawPost


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

    async def flush(self) -> None:
        self._persist_new()

    async def commit(self) -> None:
        self._persist_new()

    async def refresh(self, obj: object) -> None:
        return None

    async def rollback(self) -> None:
        return None

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
