from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import get_db
from enums import ExtractionStatus, ReviewStatus
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


def _seed_pending_with_confidence(fake_db: FakeAsyncSession, *, extraction_id: int, confidence: int) -> None:
    now = datetime.now(UTC)
    raw_post_id = extraction_id
    fake_db.seed(
        RawPost(
            id=raw_post_id,
            platform="x",
            author_handle=f"user_{raw_post_id}",
            external_id=f"post-{raw_post_id}",
            url=f"https://x.com/user_{raw_post_id}/status/{raw_post_id}",
            content_text="historical pending",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.unreviewed,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=extraction_id,
            raw_post_id=raw_post_id,
            status=ExtractionStatus.pending,
            extracted_json={
                "summary": "hist",
                "confidence": confidence,
                "meta": {},
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": confidence,
                        "summary": "hist view",
                    }
                ],
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            prompt_version="extract_v1",
            prompt_text=None,
            prompt_hash=None,
            raw_model_output=None,
            parsed_model_output=None,
            model_latency_ms=None,
            model_input_tokens=None,
            model_output_tokens=None,
            last_error=None,
            reviewed_at=None,
            reviewed_by=None,
            review_note=None,
            applied_kol_view_id=None,
            auto_applied_count=0,
            auto_policy=None,
            auto_applied_kol_view_ids=None,
            created_at=now,
        )
    )


def test_backfill_auto_review_rejects_pending_confidence_69() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_with_confidence(fake_db, extraction_id=1, confidence=69)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/backfill-auto-review?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 1
    assert body["rejected_count"] == 1
    assert body["approved_count"] == 0
    extraction = fake_db._data[PostExtraction][1]
    raw_post = fake_db._data[RawPost][1]
    assert extraction.status == ExtractionStatus.rejected
    assert raw_post.review_status == ReviewStatus.rejected
    assert extraction.extracted_json["meta"]["auto_rejected"] is True


def test_backfill_auto_review_approves_pending_confidence_70() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_with_confidence(fake_db, extraction_id=1, confidence=70)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/backfill-auto-review?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 1
    assert body["approved_count"] == 1
    assert body["rejected_count"] == 0
    extraction = fake_db._data[PostExtraction][1]
    raw_post = fake_db._data[RawPost][1]
    assert extraction.status == ExtractionStatus.approved
    assert raw_post.review_status == ReviewStatus.approved
    assert extraction.extracted_json["meta"]["auto_approved"] is True


def test_backfill_auto_review_is_idempotent_on_second_run() -> None:
    fake_db = FakeAsyncSession()
    _seed_pending_with_confidence(fake_db, extraction_id=1, confidence=69)
    _seed_pending_with_confidence(fake_db, extraction_id=2, confidence=70)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    first = client.post("/admin/extractions/backfill-auto-review?confirm=YES")
    second = client.post("/admin/extractions/backfill-auto-review?confirm=YES")
    app.dependency_overrides.clear()

    assert first.status_code == 200
    first_body = first.json()
    assert first_body["scanned"] == 2
    assert first_body["approved_count"] == 1
    assert first_body["rejected_count"] == 1
    assert first_body["skipped_already_terminal_count"] == 0

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["scanned"] == 2
    assert second_body["approved_count"] == 0
    assert second_body["rejected_count"] == 0
    assert second_body["skipped_already_terminal_count"] == 2
