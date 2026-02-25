from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, Horizon, Stance
from main import app, reset_runtime_counters
from models import Asset, Kol, KolView, PostExtraction, RawPost
from settings import get_settings
from test_dashboard_and_ingest import FakeAsyncSession


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    get_settings.cache_clear()
    reset_runtime_counters()


def _seed_wrong_extraction_with_applied_view(fake_db: FakeAsyncSession) -> None:
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p-1",
            url="https://x.com/alice/status/1",
            content_text="supply chain",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(Asset(id=1, symbol="HYNIX", name="Hynix", market="STOCK", created_at=now))
    fake_db.seed(Kol(id=1, platform="x", handle="alice", display_name="Alice", enabled=True, created_at=now))
    fake_db.seed(
        KolView(
            id=44,
            kol_id=1,
            asset_id=1,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=78,
            summary="HBM strong",
            source_url="https://x.com/alice/status/1",
            as_of=now.date(),
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=476,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={
                "assets": [],
                "asset_views": [],
                "meta": {"asset_views_original_count": 1, "asset_views_final_count": 0, "asset_views_cap_reason": "no_direct_mentions_no_macro"},
            },
            parsed_model_output=None,
            raw_model_output=None,
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            auto_applied_count=1,
            auto_applied_kol_view_ids=[44],
            created_at=now,
        )
    )


def test_refresh_wrong_extracted_json_updates_from_applied_views() -> None:
    fake_db = FakeAsyncSession()
    _seed_wrong_extraction_with_applied_view(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=false")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 1
    assert body["updated"] == 1
    assert 476 in body["updated_ids"]
    refreshed = fake_db._data[PostExtraction][476].extracted_json
    assert isinstance(refreshed.get("asset_views"), list)
    assert refreshed["asset_views"][0]["symbol"] == "HYNIX"
    assert refreshed["meta"]["asset_views_final_count"] == 1
    assert refreshed["meta"]["asset_views_cap_reason"] is None


def test_refresh_wrong_extracted_json_dry_run_does_not_write() -> None:
    fake_db = FakeAsyncSession()
    _seed_wrong_extraction_with_applied_view(fake_db)
    before = fake_db._data[PostExtraction][476].extracted_json

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=true")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 1
    assert body["updated"] == 1
    assert body["dry_run"] is True
    assert fake_db._data[PostExtraction][476].extracted_json == before


def test_refresh_wrong_extracted_json_is_idempotent_on_second_run() -> None:
    fake_db = FakeAsyncSession()
    _seed_wrong_extraction_with_applied_view(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=false")
    second = client.post("/admin/extractions/refresh-wrong-extracted-json?confirm=YES&days=365&limit=2000&dry_run=false")
    app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["updated"] == 1
    assert second.json()["updated"] == 0
