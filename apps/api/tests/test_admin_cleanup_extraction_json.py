from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus
from main import app, reset_runtime_counters
from models import PostExtraction, RawPost
from settings import get_settings
from test_dashboard_and_ingest import FakeAsyncSession


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    reset_runtime_counters()
    yield
    get_settings.cache_clear()
    reset_runtime_counters()


def _seed_data(fake_db: FakeAsyncSession) -> None:
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p-1",
            url="https://x.com/alice/status/1",
            content_text="post",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=1893,
            raw_post_id=1,
            status=ExtractionStatus.rejected,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/alice/status/1",
                "islibrary": 0,
                "hasview": 0,
                "asset_views": [],
                "library_entry": None,
                "meta": {
                    "auto_rejected": True,
                    "auto_reject_reason": "hasview_zero",
                    "auto_reject_threshold": 80,
                    "auto_policy_applied": "no_auto_review_user_trigger",
                    "summary_language": "zh",
                    "summary_language_violation": False,
                    "asset_views_original_count": 1,
                    "asset_views_final_count": 0,
                    "raw_saved_len": 401,
                    "raw_truncated": False,
                },
            },
            parsed_model_output={
                "meta": {"raw_len": 123},
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/alice/status/1",
                "islibrary": 0,
                "hasview": 0,
                "asset_views": [],
                "library_entry": None,
                "legacy": "drop-me",
            },
            model_name="gpt-4o-mini",
            extractor_name="openrouter_json_mode",
            created_at=now,
        )
    )


def test_cleanup_extractions_json_rewrites_legacy_keys() -> None:
    fake_db = FakeAsyncSession()
    _seed_data(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/cleanup-json?confirm=YES&days=365&limit=2000&dry_run=false")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 1
    cleaned = fake_db._data[PostExtraction][1893]
    meta = cleaned.extracted_json["meta"]
    assert meta["auto_review_reason"] == "hasview_zero"
    assert meta["auto_review_threshold"] == 80
    assert "auto_reject_reason" not in meta
    assert "auto_reject_threshold" not in meta
    assert "auto_policy_applied" not in meta
    assert "summary_language" not in meta
    assert "summary_language_violation" not in meta
    assert "asset_views_original_count" not in meta
    assert "asset_views_final_count" not in meta
    assert "raw_saved_len" not in meta
    assert "raw_truncated" not in meta
    assert cleaned.parsed_model_output is not None
    assert "meta" not in cleaned.parsed_model_output
    assert "legacy" not in cleaned.parsed_model_output
    assert set(cleaned.parsed_model_output.keys()) == {
        "as_of",
        "source_url",
        "islibrary",
        "hasview",
        "asset_views",
        "library_entry",
    }
