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


def _seed_data(fake_db: FakeAsyncSession) -> None:
    now = datetime.now(UTC)
    fake_db.seed(
        RawPost(
            id=1,
            platform="x",
            author_handle="alice",
            external_id="p-1",
            url="https://x.com/alice/status/1",
            content_text="p1",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.unreviewed,
        )
    )
    fake_db.seed(
        RawPost(
            id=2,
            platform="x",
            author_handle="bob",
            external_id="p-2",
            url="https://x.com/bob/status/2",
            content_text="p2",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.unreviewed,
        )
    )
    fake_db.seed(
        RawPost(
            id=3,
            platform="x",
            author_handle="carol",
            external_id="p-3",
            url="https://x.com/carol/status/3",
            content_text="p3",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.unreviewed,
        )
    )
    fake_db.seed(
        RawPost(
            id=4,
            platform="x",
            author_handle="dave",
            external_id="p-4",
            url="https://x.com/dave/status/4",
            content_text="p4",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.unreviewed,
        )
    )
    fake_db.seed(
        RawPost(
            id=5,
            platform="x",
            author_handle="eve",
            external_id="p-5",
            url="https://x.com/eve/status/5",
            content_text="p5",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.approved,
        )
    )
    fake_db.seed(
        RawPost(
            id=6,
            platform="x",
            author_handle="frank",
            external_id="p-6",
            url="https://x.com/frank/status/6",
            content_text="p6",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.rejected,
        )
    )
    fake_db.seed(
        RawPost(
            id=7,
            platform="x",
            author_handle="gina",
            external_id="p-7",
            url="https://x.com/gina/status/7",
            content_text="p7",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.rejected,
        )
    )
    fake_db.seed(
        RawPost(
            id=8,
            platform="x",
            author_handle="helen",
            external_id="p-8",
            url="https://x.com/helen/status/8",
            content_text="p8",
            posted_at=now,
            fetched_at=now,
            raw_json=None,
            review_status=ReviewStatus.rejected,
        )
    )

    fake_db.seed(
        PostExtraction(
            id=101,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/alice/status/1",
                "islibrary": 0,
                "hasview": 0,
                "asset_views": [],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=102,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/bob/status/2",
                "islibrary": 0,
                "hasview": 1,
                "asset_views": [
                    {
                        "symbol": "BTC",
                        "market": "CRYPTO",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 80,
                        "summary": "看多",
                    }
                ],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=103,
            raw_post_id=3,
            status=ExtractionStatus.pending,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/carol/status/3",
                "islibrary": 1,
                "hasview": 0,
                "asset_views": [
                    {
                        "symbol": "ETH",
                        "market": "CRYPTO",
                        "stance": "neutral",
                        "horizon": "1w",
                        "confidence": 80,
                        "summary": "中性",
                    }
                ],
                "library_entry": {"tag": "thesis", "summary": "测试"},
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=104,
            raw_post_id=4,
            status=ExtractionStatus.approved,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/dave/status/4",
                "islibrary": 0,
                "hasview": 1,
                "asset_views": [
                    {
                        "symbol": "SOL",
                        "market": "CRYPTO",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 88,
                        "summary": "看多",
                    }
                ],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=105,
            raw_post_id=5,
            status=ExtractionStatus.approved,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/eve/status/5",
                "islibrary": 0,
                "hasview": 0,
                "asset_views": [],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=106,
            raw_post_id=6,
            status=ExtractionStatus.rejected,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/frank/status/6",
                "islibrary": 0,
                "hasview": 1,
                "asset_views": [
                    {
                        "symbol": "SOL",
                        "market": "CRYPTO",
                        "stance": "bull",
                        "horizon": "1w",
                        "confidence": 80,
                        "summary": "看多",
                    }
                ],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=107,
            raw_post_id=7,
            status=ExtractionStatus.rejected,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/gina/status/7",
                "islibrary": 0,
                "hasview": 0,
                "asset_views": [],
                "library_entry": None,
            },
            parsed_model_output={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/gina/status/7",
                "islibrary": 0,
                "hasview": 1,
                "asset_views": [
                    {
                        "symbol": "OKB",
                        "market": "CRYPTO",
                        "stance": "bull",
                        "horizon": "3m",
                        "confidence": 70,
                        "summary": "看多",
                    }
                ],
                "library_entry": None,
            },
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )
    fake_db.seed(
        PostExtraction(
            id=108,
            raw_post_id=8,
            status=ExtractionStatus.rejected,
            extracted_json={
                "as_of": now.date().isoformat(),
                "source_url": "https://x.com/helen/status/8",
                "islibrary": 1,
                "hasview": 0,
                "asset_views": [],
                "library_entry": {"tag": "macro", "summary": "测试"},
            },
            parsed_model_output=None,
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            created_at=now,
        )
    )


def test_admin_recompute_extraction_statuses_updates_all_statuses_by_rules() -> None:
    fake_db = FakeAsyncSession()
    _seed_data(fake_db)

    async def override_get_db():
        yield fake_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    response = client.post("/admin/extractions/recompute-statuses?confirm=YES&days=365&limit=5000&dry_run=false")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 5
    assert body["approved_count"] == 3
    assert body["rejected_count"] == 5
    assert body["skipped_terminal_count"] == 0
    assert set(body["updated_ids"]) == {101, 102, 103, 105, 106}

    assert fake_db._data[PostExtraction][101].status == ExtractionStatus.rejected
    assert fake_db._data[PostExtraction][102].status == ExtractionStatus.approved
    assert fake_db._data[PostExtraction][103].status == ExtractionStatus.rejected
    assert fake_db._data[PostExtraction][104].status == ExtractionStatus.approved
    assert fake_db._data[PostExtraction][105].status == ExtractionStatus.rejected
    assert fake_db._data[PostExtraction][106].status == ExtractionStatus.approved
    assert fake_db._data[PostExtraction][107].status == ExtractionStatus.rejected
    assert fake_db._data[PostExtraction][108].status == ExtractionStatus.rejected
