from datetime import UTC, date, datetime
from pathlib import Path
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import Horizon, Stance
from main import app, build_asset_views_response


async def _fake_db():
    yield SimpleNamespace()


def _view(
    *,
    view_id: int,
    kol_id: int,
    asset_id: int,
    horizon: Horizon,
    stance: Stance,
    confidence: int,
    as_of: date,
    created_at: datetime,
):
    return SimpleNamespace(
        id=view_id,
        kol_id=kol_id,
        asset_id=asset_id,
        horizon=horizon,
        stance=stance,
        confidence=confidence,
        summary=f"summary-{view_id}",
        source_url=f"https://example.com/{view_id}",
        as_of=as_of,
        created_at=created_at,
    )


def test_create_kol_view_invalid_enum_returns_422():
    app.dependency_overrides[get_db] = _fake_db
    client = TestClient(app)

    payload = {
        "kol_id": 1,
        "asset_id": 1,
        "stance": "moon",
        "horizon": "10y",
        "confidence": 50,
        "summary": "invalid enum test",
        "source_url": "https://example.com",
        "as_of": "2026-02-21",
    }
    response = client.post("/kol-views", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_build_asset_views_response_keeps_latest_version():
    views = [
        _view(
            view_id=1,
            kol_id=10,
            asset_id=1,
            horizon=Horizon.one_month,
            stance=Stance.bull,
            confidence=90,
            as_of=date(2026, 2, 10),
            created_at=datetime(2026, 2, 10, 8, 0, tzinfo=UTC),
        ),
        _view(
            view_id=2,
            kol_id=10,
            asset_id=1,
            horizon=Horizon.one_month,
            stance=Stance.bear,
            confidence=40,
            as_of=date(2026, 2, 12),
            created_at=datetime(2026, 2, 12, 8, 0, tzinfo=UTC),
        ),
    ]

    result = build_asset_views_response(asset_id=1, views=views)
    group_1m = next(group for group in result.groups if group.horizon == Horizon.one_month)

    assert len(group_1m.bull) == 0
    assert len(group_1m.bear) == 1
    assert group_1m.bear[0].id == 2


def test_build_asset_views_response_group_and_confidence_sort():
    views = [
        _view(
            view_id=11,
            kol_id=1,
            asset_id=1,
            horizon=Horizon.one_month,
            stance=Stance.bull,
            confidence=55,
            as_of=date(2026, 2, 20),
            created_at=datetime(2026, 2, 20, 8, 0, tzinfo=UTC),
        ),
        _view(
            view_id=12,
            kol_id=2,
            asset_id=1,
            horizon=Horizon.one_month,
            stance=Stance.bull,
            confidence=88,
            as_of=date(2026, 2, 20),
            created_at=datetime(2026, 2, 20, 9, 0, tzinfo=UTC),
        ),
        _view(
            view_id=13,
            kol_id=3,
            asset_id=1,
            horizon=Horizon.one_week,
            stance=Stance.neutral,
            confidence=60,
            as_of=date(2026, 2, 20),
            created_at=datetime(2026, 2, 20, 9, 30, tzinfo=UTC),
        ),
    ]

    result = build_asset_views_response(asset_id=1, views=views)

    assert [group.horizon for group in result.groups] == [Horizon.one_week, Horizon.one_month]
    one_month = next(group for group in result.groups if group.horizon == Horizon.one_month)
    assert [item.confidence for item in one_month.bull] == [88, 55]
