from __future__ import annotations

from datetime import UTC, datetime, timedelta
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


def _seed_pending(fake_db: FakeAsyncSession, *, raw_post_id: int, extraction_id: int) -> None:
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=raw_post_id,
            platform="x",
            author_handle=f"user_{raw_post_id}",
            external_id=f"pending-{raw_post_id}",
            url=f"https://x.com/user_{raw_post_id}/status/{raw_post_id}",
            content_text=f"pending post {raw_post_id}",
            posted_at=now - timedelta(minutes=2),
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
            extracted_json={"summary": None, "assets": [], "asset_views": [], "reasoning": None},
            model_name="dummy-v2",
            extractor_name="dummy",
            created_at=now - timedelta(minutes=1),
        )
    )


def _latest_extraction_for_raw_post(fake_db: FakeAsyncSession, raw_post_id: int) -> PostExtraction:
    rows = [item for item in fake_db._data[PostExtraction].values() if item.raw_post_id == raw_post_id]
    rows.sort(key=lambda item: ((item.created_at or datetime.min.replace(tzinfo=UTC)), item.id or 0), reverse=True)
    return rows[0]


def test_bulk_retry_pending_unwraps_and_updates_extracted_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_pending(fake_db, raw_post_id=1, extraction_id=11)
    _seed_pending(fake_db, raw_post_id=2, extraction_id=22)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "extracted_json": {
                "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
                "stance": "bull",
                "horizon": "1w",
                "confidence": 66,
                "summary": f"bulk fresh {raw_post.id}",
                "source_url": raw_post.url,
                "as_of": "2026-02-26",
                "event_tags": [],
                "asset_views": [],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/reextract-pending?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["scanned"] == 2
    assert body["created"] == 2
    assert body["succeeded_parse"] == 2
    assert body["failed_parse"] == 0

    latest_1 = _latest_extraction_for_raw_post(fake_db, 1).extracted_json
    latest_2 = _latest_extraction_for_raw_post(fake_db, 2).extracted_json
    for payload in (latest_1, latest_2):
        assert "extracted_json" not in payload
        assert payload["stance"] == "bull"
        assert payload["source_url"].startswith("https://x.com/")
        assert isinstance(payload["summary"], str) and payload["summary"].startswith("bulk fresh")
        assert payload["meta"]["force_reextract"] is True
        assert payload["meta"]["force_reextract_triggered_by"] == "bulk_pending_retry"


def test_bulk_retry_pending_runs_auto_review_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_pending(fake_db, raw_post_id=1, extraction_id=11)
    _seed_pending(fake_db, raw_post_id=2, extraction_id=22)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        confidence = 80 if raw_post.id == 1 else 30
        return {
            "extracted_json": {
                "assets": [{"symbol": "BTC", "market": "CRYPTO"}],
                "stance": "bull",
                "horizon": "1w",
                "confidence": confidence,
                "summary": f"c{confidence}",
                "source_url": raw_post.url,
                "as_of": "2026-02-26",
                "event_tags": [],
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": confidence,
                        "summary": f"c{confidence}",
                    }
                ],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/reextract-pending?confirm=YES")
    pending_list = client.get("/extractions?status=pending&limit=20&offset=0")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["auto_approved"] == 1
    assert body["auto_rejected"] == 1

    latest_1 = _latest_extraction_for_raw_post(fake_db, 1)
    latest_2 = _latest_extraction_for_raw_post(fake_db, 2)
    assert latest_1.status == ExtractionStatus.approved
    assert latest_2.status == ExtractionStatus.rejected
    assert latest_1.reviewed_by == "auto"
    assert latest_2.reviewed_by == "auto"
    assert pending_list.status_code == 200
    assert pending_list.json() == []


def test_bulk_retry_pending_noneany_rejects_even_if_manual_rule_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_db = FakeAsyncSession()
    _seed_pending(fake_db, raw_post_id=1, extraction_id=11)
    monkeypatch.setenv("EXTRACTOR_MODE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_extract(self, raw_post: RawPost):  # noqa: ANN001
        return {
            "extracted_json": {
                "assets": ["NoneAny"],
                "stance": "bull",
                "horizon": "1w",
                "confidence": 80,
                "summary": "noneany",
                "source_url": raw_post.url,
                "as_of": "2026-02-26",
                "event_tags": [],
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 80,
                        "summary": "would approve without noneany",
                    }
                ],
            }
        }

    monkeypatch.setattr("services.extraction.OpenAIExtractor.extract", fake_extract)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/reextract-pending?confirm=YES")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["noneany_rejected"] == 1
    assert body["auto_approved"] == 0
    assert body["auto_rejected"] == 0

    latest = _latest_extraction_for_raw_post(fake_db, 1)
    assert latest.status == ExtractionStatus.rejected
    assert latest.extracted_json["meta"]["noneany_detected"] is True
    assert latest.extracted_json["meta"]["auto_review_reason"] == "no_investment_asset_noneany"
