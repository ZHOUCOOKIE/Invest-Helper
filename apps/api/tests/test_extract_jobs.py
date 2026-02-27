from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
import time

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as main_module
from db import get_db
from enums import ExtractionStatus, ReviewStatus
from main import app, reset_runtime_counters
from models import Kol, PostExtraction, RawPost
from settings import get_settings
from test_dashboard_and_ingest import FakeAsyncSession


class _SessionCtx:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _seed_kol(db: FakeAsyncSession, *, kol_id: int = 1, handle: str = "alice", enabled: bool = True) -> None:
    db.seed(
        Kol(
            id=kol_id,
            platform="x",
            handle=handle,
            display_name=handle,
            enabled=enabled,
            created_at=datetime.now(UTC),
        )
    )


def _seed_raw_post(
    db: FakeAsyncSession,
    *,
    raw_post_id: int,
    kol_id: int = 1,
    handle: str = "alice",
    review_status: ReviewStatus = ReviewStatus.unreviewed,
) -> None:
    db.seed(
        RawPost(
            id=raw_post_id,
            platform="x",
            kol_id=kol_id,
            author_handle=handle,
            external_id=f"ext-{raw_post_id}",
            url=f"https://x.com/{handle}/status/{raw_post_id}",
            content_text=f"post {raw_post_id}",
            posted_at=datetime.now(UTC),
            fetched_at=datetime.now(UTC),
            review_status=review_status,
            reviewed_at=None,
            reviewed_by=None,
            raw_json=None,
        )
    )


def _poll_job_done(client: TestClient, job_id: str, *, timeout_sec: float = 12.0) -> dict:
    started = time.time()
    last_body: dict = {}
    while time.time() - started < timeout_sec:
        resp = client.get(f"/extract-jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        last_body = body
        if body.get("is_terminal") is True or body["status"] in {"done", "completed", "failed", "cancelled", "timeout"}:
            return body
        time.sleep(0.03)
    raise AssertionError(f"extract job did not finish in {timeout_sec}s, last={last_body}")


def test_extract_job_happy_path_counts(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    _seed_raw_post(fake_db, raw_post_id=2)
    _seed_raw_post(fake_db, raw_post_id=3, review_status=ReviewStatus.rejected)

    fake_db.seed(
        PostExtraction(
            id=10,
            raw_post_id=2,
            status=ExtractionStatus.pending,
            extracted_json={"sentiment": "neutral", "assets": []},
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
            created_at=datetime.now(UTC),
        )
    )

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"sentiment": "neutral", "assets": []},
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1, 2, 3], "mode": "pending_or_failed", "batch_size": 2, "batch_sleep_ms": 0},
    )
    assert create.status_code == 201
    job_id = create.json()["job_id"]

    result = _poll_job_done(client, job_id)
    assert result["status"] == "completed"
    assert result["is_terminal"] is True
    assert result["last_updated_at"]
    assert result["requested_count"] == 3
    assert result["success_count"] == 1
    assert result["failed_count"] == 0
    assert result["skipped_count"] == 2
    assert result["requested_count"] == result["success_count"] + result["failed_count"] + result["skipped_count"]
    assert result["skipped_already_pending_count"] == 0
    assert result["skipped_already_success_count"] == 1
    assert result["skipped_already_rejected_count"] == 1
    assert result["skipped_count"] == result["skipped_already_success_count"] + result["skipped_already_rejected_count"]
    assert result["skipped_not_followed_count"] == 0

    app.dependency_overrides.clear()


def test_extract_job_reupload_resumes_only_failed(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    _seed_raw_post(fake_db, raw_post_id=2)
    _seed_raw_post(fake_db, raw_post_id=3, review_status=ReviewStatus.rejected)

    calls: list[int] = []
    attempts: dict[int, int] = {}

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        calls.append(raw_post.id)
        attempts[raw_post.id] = attempts.get(raw_post.id, 0) + 1
        if raw_post.id == 2 and attempts[raw_post.id] == 1:
            raise RuntimeError("first-pass-error")
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"sentiment": "neutral", "assets": []},
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    client = TestClient(app)

    first_create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1, 2, 3], "mode": "pending_or_failed", "batch_size": 3, "batch_sleep_ms": 0},
    )
    assert first_create.status_code == 201
    first = _poll_job_done(client, first_create.json()["job_id"])
    assert first["status"] == "completed"
    assert first["success_count"] == 1
    assert first["failed_count"] == 1
    assert first["skipped_already_rejected_count"] == 1

    calls.clear()
    second_create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1, 2, 3], "mode": "pending_or_failed", "batch_size": 3, "batch_sleep_ms": 0},
    )
    assert second_create.status_code == 201
    second = _poll_job_done(client, second_create.json()["job_id"])

    assert second["status"] == "completed"
    assert calls == [2]
    assert second["success_count"] == 1
    assert second["failed_count"] == 0
    assert second["skipped_already_pending_count"] == 0
    assert second["skipped_already_success_count"] == 1
    assert second["skipped_already_rejected_count"] == 1

    app.dependency_overrides.clear()


def test_extract_job_pending_with_result_skips_already_has_result(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    fake_db.seed(
        PostExtraction(
            id=101,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "has result", "confidence": 65, "assets": []},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=datetime.now(UTC),
        )
    )

    called_ids: list[int] = []

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "new"},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1], "mode": "pending_or_failed", "batch_size": 1, "batch_sleep_ms": 0},
    )
    assert create.status_code == 201
    result = _poll_job_done(client, create.json()["job_id"])
    app.dependency_overrides.clear()

    assert result["status"] == "completed"
    assert result["success_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped_already_success_count"] == 1
    assert result["skipped_already_has_result_count"] == 1
    assert result["auto_approved_count"] == 0
    assert result["auto_rejected_count"] == 0
    assert called_ids == []


def test_extract_job_rejected_history_counts_as_has_result_and_skips_reupload(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    now = datetime.now(UTC)
    fake_db.seed(
        PostExtraction(
            id=201,
            raw_post_id=1,
            status=ExtractionStatus.rejected,
            extracted_json={"summary": "old rejected", "confidence": 69, "assets": []},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=now - timedelta(minutes=5),
        )
    )
    fake_db.seed(
        PostExtraction(
            id=202,
            raw_post_id=1,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "latest failed", "meta": {"parse_error": True}},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error="parse failed",
            created_at=now,
        )
    )

    called_ids: list[int] = []

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "should not run"},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1], "mode": "pending_or_failed", "batch_size": 1, "batch_sleep_ms": 0},
    )
    assert create.status_code == 201
    result = _poll_job_done(client, create.json()["job_id"])
    app.dependency_overrides.clear()

    assert result["status"] == "completed"
    assert result["success_count"] == 0
    assert result["skipped_count"] == 1
    assert result["skipped_already_has_result_count"] == 1
    assert result["skipped_already_rejected_count"] == 0
    assert called_ids == []


def test_progress_and_job_counters_use_same_terminal_status_logic(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={
                "assets": [],
                "asset_views": [],
                "meta": {
                    "extraction_mode": "text_json",
                    "output_mode_used": "text_json",
                    "provider_detected": "openrouter",
                    "parse_strategy_used": "failed",
                    "raw_len": 0,
                    "repaired": False,
                    "parse_error": True,
                },
            },
            model_name="qwen",
            extractor_name="openrouter_json_mode",
            prompt_version="extract_v1",
            prompt_text=None,
            prompt_hash=None,
            raw_model_output=None,
            parsed_model_output=None,
            model_latency_ms=None,
            model_input_tokens=None,
            model_output_tokens=None,
            last_error="RuntimeError: OpenAI content is not valid JSON object text",
            reviewed_at=None,
            reviewed_by=None,
            review_note=None,
            applied_kol_view_id=None,
            auto_applied_count=0,
            auto_policy=None,
            auto_applied_kol_view_ids=None,
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1], "mode": "pending_or_failed", "batch_size": 1, "batch_sleep_ms": 0},
    )
    assert create.status_code == 201
    job = _poll_job_done(client, create.json()["job_id"])

    progress = client.get("/ingest/x/progress")
    app.dependency_overrides.clear()

    assert job["status"] == "completed"
    assert job["success_count"] == 0
    assert job["failed_count"] == 1
    assert progress.status_code == 200
    progress_body = progress.json()
    assert progress_body["extracted_success_count"] == 0
    assert progress_body["pending_count"] == 0
    assert progress_body["failed_count"] == 1


def test_frontend_job_creation_guard_or_backend_idempotency(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        await asyncio.sleep(0.15)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"assets": [], "asset_views": [], "meta": {}},
            model_name="qwen",
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    async def _slow_extract_batch(payload, db):  # noqa: ANN001
        await asyncio.sleep(0.2)
        return main_module.RawPostsExtractBatchRead(
            requested_count=len(payload.raw_post_ids),
            success_count=len(payload.raw_post_ids),
            skipped_count=0,
            skipped_already_extracted_count=0,
            failed_count=0,
        )

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_extract_raw_posts_batch_core", _slow_extract_batch)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    first = client.post(
        "/extract-jobs",
        json={
            "raw_post_ids": [1],
            "mode": "pending_or_failed",
            "batch_size": 1,
            "batch_sleep_ms": 0,
            "idempotency_key": "ingest:pending_or_failed:1:1",
        },
    )
    second = client.post(
        "/extract-jobs",
        json={
            "raw_post_ids": [1],
            "mode": "pending_or_failed",
            "batch_size": 1,
            "batch_sleep_ms": 0,
            "idempotency_key": "ingest:pending_or_failed:1:1",
        },
    )
    assert first.status_code == 201
    assert second.status_code == 201
    first_job_id = first.json()["job_id"]
    second_job_id = second.json()["job_id"]
    if first_job_id != second_job_id:
        first_state = client.get(f"/extract-jobs/{first_job_id}")
        assert first_state.status_code == 200
        assert first_state.json()["is_terminal"] is True
    else:
        assert first_job_id == second_job_id
    app.dependency_overrides.clear()


def test_extract_job_deduplicates_inflight_raw_post_ids_without_idempotency_key(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    _seed_raw_post(fake_db, raw_post_id=2)

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        await asyncio.sleep(0.15)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"assets": [], "asset_views": [], "meta": {}},
            model_name="qwen",
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    async def _slow_extract_batch(payload, db):  # noqa: ANN001
        await asyncio.sleep(0.2)
        return main_module.RawPostsExtractBatchRead(
            requested_count=len(payload.raw_post_ids),
            success_count=len(payload.raw_post_ids),
            skipped_count=0,
            skipped_already_extracted_count=0,
            failed_count=0,
        )

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_extract_raw_posts_batch_core", _slow_extract_batch)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    first = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1, 2], "mode": "pending_or_failed", "batch_size": 2, "batch_sleep_ms": 0},
    )
    second = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1], "mode": "pending_or_failed", "batch_size": 1, "batch_sleep_ms": 0},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    first_job_id = first.json()["job_id"]
    second_job_id = second.json()["job_id"]
    if first_job_id != second_job_id:
        first_state = client.get(f"/extract-jobs/{first_job_id}")
        assert first_state.status_code == 200
        assert first_state.json()["is_terminal"] is True
    else:
        assert first_job_id == second_job_id
    app.dependency_overrides.clear()


def test_extract_job_no_actionable_items_still_reaches_terminal(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    fake_db.seed(
        PostExtraction(
            id=301,
            raw_post_id=1,
            status=ExtractionStatus.approved,
            extracted_json={"summary": "already done", "assets": []},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=datetime.now(UTC),
        )
    )
    called_ids: list[int] = []

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        called_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"summary": "new"},
            model_name="gpt-4o-mini",
            extractor_name="openai_structured",
            last_error=None,
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={"raw_post_ids": [1], "mode": "pending_or_failed", "batch_size": 1, "batch_sleep_ms": 0},
    )
    assert create.status_code == 201
    result = _poll_job_done(client, create.json()["job_id"])
    app.dependency_overrides.clear()

    assert result["is_terminal"] is True
    assert result["status"] == "completed"
    assert result["success_count"] == 0
    assert result["skipped_count"] == 1
    assert called_ids == []


def test_get_extract_job_is_read_only_and_does_not_trigger_auto_approve(monkeypatch) -> None:  # noqa: ANN001
    called = 0

    async def _fake_auto_apply(*args, **kwargs):  # noqa: ANN001
        nonlocal called
        called += 1

    monkeypatch.setattr(main_module, "_auto_apply_extraction_views", _fake_auto_apply)
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    now = datetime.now(UTC)
    main_module.EXTRACT_JOBS["job-readonly"] = {
        "job_id": "job-readonly",
        "status": "running",
        "mode": "pending_or_failed",
        "batch_size": 1,
        "batch_sleep_ms": 0,
        "raw_post_ids": [1],
        "idempotency_key": None,
        "last_error_summary": None,
        "created_at": now,
        "last_updated_at": now,
        "started_at": now,
        "finished_at": None,
        **main_module._empty_extract_counters(requested_count=1),
    }
    before_updated_at = main_module.EXTRACT_JOBS["job-readonly"]["last_updated_at"]
    client = TestClient(app)

    first_response = client.get("/extract-jobs/job-readonly")
    second_response = client.get("/extract-jobs/job-readonly")
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    body = second_response.json()

    assert body["job_id"] == "job-readonly"
    assert body["is_terminal"] is False
    assert called == 0
    assert main_module.EXTRACT_JOBS["job-readonly"]["last_updated_at"] == before_updated_at


def test_exception_handler_returns_json_and_frontend_can_render_text_fallback(monkeypatch) -> None:  # noqa: ANN001
    def _boom():
        raise RuntimeError("boom-json")

    monkeypatch.setattr(main_module, "get_settings", _boom)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/extractor-status")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert isinstance(body.get("request_id"), str)
    assert body.get("error_code") == "internal_server_error"
    assert body.get("message") == "Internal Server Error"
    assert "boom-json" in str(body.get("detail"))
    assert response.headers.get("x-request-id") == body.get("request_id")


def test_extract_job_import_ai_limit_caps_calls_and_marks_skipped(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    for raw_post_id in range(1, 6):
        _seed_raw_post(fake_db, raw_post_id=raw_post_id)

    model_call_ids: list[int] = []

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        limiter = kwargs.get("openai_call_limiter")
        if limiter is not None and not limiter.try_consume():
            raise main_module.ImportAICallLimitReached("import_limit")
        model_call_ids.append(raw_post.id)
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"assets": [], "asset_views": [], "meta": {}},
            model_name="qwen",
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={
            "raw_post_ids": [1, 2, 3, 4, 5],
            "mode": "pending_or_failed",
            "batch_size": 5,
            "batch_sleep_ms": 0,
            "ai_call_limit_total": 2,
        },
    )
    assert create.status_code == 201
    result = _poll_job_done(client, create.json()["job_id"])
    app.dependency_overrides.clear()

    assert result["status"] == "completed"
    assert result["ai_call_limit_total"] == 2
    assert result["ai_call_used"] == 2
    assert len(model_call_ids) == 2
    assert result["success_count"] == 2
    assert result["skipped_due_to_import_limit_count"] == 3
    assert result["skipped_count"] == 3
    assert result["failed_count"] == 0


def test_extract_job_disables_global_call_budget_by_default(monkeypatch) -> None:  # noqa: ANN001
    fake_db = FakeAsyncSession()
    _seed_kol(fake_db)
    _seed_raw_post(fake_db, raw_post_id=1)
    _seed_raw_post(fake_db, raw_post_id=2)
    _seed_raw_post(fake_db, raw_post_id=3)

    seen_runtime_budget_overrides: list[int | None] = []

    async def _fake_create_pending_extraction(db, raw_post, **kwargs):  # noqa: ANN001
        seen_runtime_budget_overrides.append(kwargs.get("runtime_budget_total_override"))
        extraction = PostExtraction(
            raw_post_id=raw_post.id,
            status=ExtractionStatus.pending,
            extracted_json={"assets": [], "asset_views": [], "meta": {}},
            model_name="qwen",
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
            created_at=datetime.now(UTC),
        )
        db.add(extraction)
        await db.flush()
        return extraction

    async def override_get_db():
        yield fake_db

    monkeypatch.setattr(main_module, "create_pending_extraction", _fake_create_pending_extraction)
    monkeypatch.setattr(main_module, "_check_reextract_rate_limit", lambda raw_post_id: None)
    monkeypatch.setattr(
        main_module,
        "_get_runtime_throttle",
        lambda settings: {"max_concurrency": 1, "max_rpm": 100000, "batch_size": 100, "batch_sleep_ms": 0},
    )
    monkeypatch.setattr(main_module, "EXTRACT_JOB_SESSION_FACTORY", lambda: _SessionCtx(fake_db))

    app.dependency_overrides[get_db] = override_get_db
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    client = TestClient(app)

    create = client.post(
        "/extract-jobs",
        json={
            "raw_post_ids": [1, 2, 3],
            "mode": "pending_or_failed",
            "batch_size": 3,
            "batch_sleep_ms": 0,
        },
    )
    assert create.status_code == 201
    result = _poll_job_done(client, create.json()["job_id"])
    app.dependency_overrides.clear()

    assert result["status"] == "completed"
    assert result["success_count"] == 3
    assert result["failed_count"] == 0
    assert seen_runtime_budget_overrides == [0, 0, 0]


def test_extract_job_read_includes_openai_counters() -> None:
    now = datetime.now(UTC)
    state = {
        "job_id": "job-counters",
        "status": "running",
        "mode": "pending_or_failed",
        "batch_size": 10,
        "batch_sleep_ms": 0,
        "ai_call_limit_total": None,
        "ai_call_used": 4,
        "requested_count": 10,
        "success_count": 4,
        "skipped_count": 3,
        "skipped_already_extracted_count": 3,
        "failed_count": 3,
        "openai_call_attempted_count": 7,
        "openai_call_succeeded_count": 4,
        "openai_call_429_count": 2,
        "openai_call_timeout_count": 1,
        "openai_total_sleep_ms": 1234,
        "max_concurrency_used": 3,
        "last_error_summary": None,
        "created_at": now,
        "last_updated_at": now,
        "started_at": now,
        "finished_at": None,
        **main_module._empty_extract_counters(requested_count=10),
    }
    state.update(
        {
            "requested_count": 10,
            "success_count": 4,
            "skipped_count": 3,
            "failed_count": 3,
            "skipped_already_extracted_count": 3,
            "openai_call_attempted_count": 7,
            "openai_call_succeeded_count": 4,
            "openai_call_429_count": 2,
            "openai_call_timeout_count": 1,
            "openai_total_sleep_ms": 1234,
            "max_concurrency_used": 3,
        }
    )

    read = main_module._extract_job_read(state).model_dump()
    assert read["openai_call_attempted_count"] == 7
    assert read["openai_call_succeeded_count"] == 4
    assert read["openai_call_429_count"] == 2
    assert read["openai_call_timeout_count"] == 1
    assert read["openai_total_sleep_ms"] == 1234
    assert read["max_concurrency_used"] == 3


def teardown_function(function):  # noqa: ANN001
    app.dependency_overrides.clear()
    main_module.EXTRACT_JOBS.clear()
    main_module.EXTRACT_JOB_TASKS.clear()
    main_module.EXTRACT_JOB_IDEMPOTENCY.clear()
    main_module.EXTRACT_JOB_SESSION_FACTORY = main_module.AsyncSessionLocal
    reset_runtime_counters()
    get_settings.cache_clear()
