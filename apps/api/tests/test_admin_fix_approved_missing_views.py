from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, Horizon, ReviewStatus, Stance
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


def test_fix_approved_missing_views_backfills_from_parsed_model_output_without_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    now = datetime.now(UTC)

    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p-1",
            url="https://x.com/alice/status/1",
            content_text="今天看供应链，后续继续观察。",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.approved,
            reviewed_by="auto",
            reviewed_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=1,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={"assets": [], "asset_views": [], "meta": {"asset_views_cap_reason": "no_direct_mentions_no_macro"}},
            parsed_model_output={
                "assets": [{"symbol": "HYNIX", "market": "STOCK"}],
                "asset_views": [
                    {
                        "symbol": "HYNIX",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 78,
                        "summary": "HBM demand keeps strong",
                        "reasoning": "memory cycle turns up",
                    }
                ],
                "summary": "Korean memory strength",
                "source_url": "https://x.com/alice/status/1",
            },
            raw_model_output=None,
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            reviewed_by="auto",
            reviewed_at=now,
            auto_applied_count=0,
            auto_applied_kol_view_ids=None,
            created_at=now,
        )
    )

    fake_db.seed(Kol(id=10, platform="x", handle="bob", display_name="Bob", enabled=True, created_at=now))
    fake_db.seed(Asset(id=10, symbol="BTC", name="Bitcoin", market="CRYPTO", created_at=now))
    fake_db.seed(
        KolView(
            id=500,
            kol_id=10,
            asset_id=10,
            stance=Stance.bull,
            horizon=Horizon.one_week,
            confidence=80,
            summary="existing",
            source_url="https://x.com/bob/status/10",
            as_of=now.date(),
            created_at=now,
        )
    )
    fake_db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="bob",
            external_id="p-2",
            url="https://x.com/bob/status/2",
            content_text="BTC up",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.approved,
            reviewed_by="auto",
            reviewed_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=2,
            raw_post_id=2,
            status=ExtractionStatus.approved,
            extracted_json={"asset_views": [{"symbol": "BTC", "stance": "bull", "horizon": "1w", "confidence": 80}]},
            parsed_model_output=None,
            raw_model_output=None,
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            reviewed_by="auto",
            reviewed_at=now,
            auto_applied_count=1,
            auto_applied_kol_view_ids=[500],
            applied_kol_view_id=500,
            created_at=now,
        )
    )

    def fail_if_called(*args, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("extractor must not be called during fix endpoint")

    monkeypatch.setattr("main.select_extractor", fail_if_called)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    dry_run = client.post("/admin/fix/approved-missing-views?confirm=YES&days=90&limit=200&dry_run=true")
    assert dry_run.status_code == 200
    dry_body = dry_run.json()
    assert dry_body["scanned"] == 1
    assert dry_body["fixed"] == 1
    assert dry_body["skipped"] == 0
    assert dry_body["dry_run"] is True

    run = client.post("/admin/fix/approved-missing-views?confirm=YES&days=90&limit=200&dry_run=false")
    assert run.status_code == 200
    run_body = run.json()
    assert run_body["scanned"] == 1
    assert run_body["fixed"] == 1
    assert run_body["dry_run"] is False

    extraction = fake_db._data[PostExtraction][1]
    assert extraction.auto_applied_count == 1
    assert isinstance(extraction.auto_applied_kol_view_ids, list)
    assert len(extraction.auto_applied_kol_view_ids) == 1
    assert any(asset.symbol == "HYNIX" for asset in fake_db._data[Asset].values())

    before_count = len(fake_db._data[KolView])
    rerun = client.post("/admin/fix/approved-missing-views?confirm=YES&days=90&limit=200&dry_run=false")
    app.dependency_overrides.clear()
    assert rerun.status_code == 200
    rerun_body = rerun.json()
    assert rerun_body["scanned"] == 0
    assert rerun_body["fixed"] == 0
    assert len(fake_db._data[KolView]) == before_count


def test_fix_approved_missing_views_refreshes_extracted_json_for_already_applied_rows() -> None:
    fake_db = FakeAsyncSession()
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
            review_status=ReviewStatus.approved,
            reviewed_by="auto",
            reviewed_at=now,
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
            extracted_json={"assets": [], "asset_views": [], "meta": {"asset_views_final_count": 0}},
            parsed_model_output={
                "assets": [{"symbol": "HYNIX", "market": "STOCK"}],
                "asset_views": [
                    {
                        "symbol": "HYNIX",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 78,
                        "summary": "HBM strong",
                        "reasoning": "cycle up",
                    }
                ],
            },
            raw_model_output=None,
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            reviewed_by="auto",
            reviewed_at=now,
            auto_applied_count=1,
            auto_applied_kol_view_ids=[44],
            created_at=now,
        )
    )

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/fix/approved-missing-views?confirm=YES&days=90&limit=200&dry_run=false")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 1
    assert body["fixed"] == 1
    extraction = fake_db._data[PostExtraction][476]
    assert isinstance(extraction.extracted_json.get("asset_views"), list)
    assert extraction.extracted_json["asset_views"][0]["symbol"] == "HYNIX"
    assert len(fake_db._data[KolView]) == 1
