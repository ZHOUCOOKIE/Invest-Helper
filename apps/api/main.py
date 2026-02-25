from collections import Counter, defaultdict, deque
import asyncio
from datetime import UTC, date, datetime, timedelta
from enum import Enum
import hashlib
import json
import random
import re
import time
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import Body, Depends, FastAPI, HTTPException, Path, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import AsyncSessionLocal, get_db
from enums import HORIZON_ORDER, Horizon, ReviewStatus, Stance
from enums import ExtractionStatus
from models import Asset, AssetAlias, DailyDigest, Kol, KolView, PostExtraction, ProfileKolWeight, RawPost
from schemas import (
    AssetCreate,
    AssetAliasCreate,
    AssetAliasMapRead,
    AssetAliasRead,
    AssetViewFeedItemRead,
    AssetViewsGroupRead,
    AssetViewsFeedRead,
    AssetViewsMetaRead,
    AssetRead,
    AdminDeletePendingExtractionsRead,
    AdminBackfillAutoReviewRead,
    AdminHardDeleteRead,
    AssetViewsRead,
    AutoAppliedViewRead,
    DashboardActiveKolAssetRead,
    DashboardActiveKolRead,
    DashboardAssetLatestViewRead,
    DashboardAssetRead,
    DashboardClarityRead,
    DashboardExtractionStatsRead,
    DashboardPendingExtractionRead,
    DashboardRead,
    DashboardTopAssetRead,
    DailyDigestRead,
    ExtractJobCreateRead,
    ExtractJobCreateRequest,
    ExtractJobRead,
    ExtractorStatusRead,
    ExtractionApproveRequest,
    ExtractionApproveBatchRequest,
    ExtractionRejectRequest,
    KolCreate,
    KolRead,
    KolViewCreate,
    KolViewRead,
    ManualIngestCreate,
    ManualIngestRead,
    PostExtractionRead,
    PostExtractionWithRawPostRead,
    ProfileKolsUpdateRequest,
    ProfileMarketsUpdateRequest,
    ProfileRead,
    ProfileSummaryRead,
    RawPostsExtractBatchRead,
    RawPostsExtractBatchRequest,
    RawPostCreate,
    RawPostRead,
    RuntimeCallBudgetUpdateRequest,
    RuntimeBurstUpdateRequest,
    RuntimeThrottleUpdateRequest,
    RuntimeSettingsRead,
    XImportItemCreate,
    XConvertResponseRead,
    XConvertErrorRead,
    XIngestProgressRead,
    XRetryFailedRead,
    XImportStatsRead,
    XImportedByHandleRead,
    XCreatedKolRead,
    XHandleSummaryRead,
    XSkippedNotFollowedRead,
    XImportTemplateRead,
    XFollowingImportErrorRead,
    XFollowingImportKolRead,
    XFollowingImportStatsRead,
    AutoReviewBackfillErrorRead,
)
from services.extraction import (
    EXTRACTION_OUTPUT_TEXT_JSON,
    DummyExtractor,
    OpenAIFallbackError,
    OpenAIExtractor,
    detect_provider_from_base_url,
    default_extracted_json,
    normalize_extracted_json,
    get_openai_call_budget_remaining,
    resolve_extraction_output_mode,
    reset_openai_call_budget_counter,
    select_extractor,
    try_consume_openai_call_budget,
)
from services.digests import (
    generate_daily_digest as generate_daily_digest_service,
    get_daily_digest_by_date as get_daily_digest_by_date_service,
    get_daily_digest_by_id as get_daily_digest_by_id_service,
    list_daily_digest_dates as list_daily_digest_dates_service,
)
from services.profiles import (
    get_profile as get_profile_service,
    list_profiles as list_profiles_service,
    update_profile_kols as update_profile_kols_service,
    update_profile_markets as update_profile_markets_service,
)
from services.prompts import build_extract_prompt
from scripts.x_import_converter import convert_records, load_records_from_bytes
from settings import get_settings

app = FastAPI(title="InvestPulse API")
EXTRACTION_REVIEWER = "human-review"
AUTO_EXTRACTION_REVIEWER = "auto"
AUTO_REVIEW_CONFIDENCE_THRESHOLD = 70
REEXTRACT_ATTEMPTS: dict[int, deque[datetime]] = defaultdict(deque)
RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE: int | None = None
UNLIMITED_SAFE_BURST_BUDGET = 100000
RUNTIME_BURST_STATE: dict[str, Any] = {
    "enabled": False,
    "mode": None,
    "call_budget": None,
    "expires_at": None,
}
RUNTIME_THROTTLE_OVERRIDE: dict[str, int] = {}
RUNTIME_BUDGET_WINDOW_START: datetime | None = None
RUNTIME_BUDGET_WINDOW_END: datetime | None = None
EXTRACT_JOBS: dict[str, dict[str, Any]] = {}
EXTRACT_JOBS_LOCK = asyncio.Lock()
EXTRACT_JOB_TASKS: dict[str, asyncio.Task[None]] = {}
EXTRACT_JOB_IDEMPOTENCY: dict[str, str] = {}
EXTRACT_JOB_SESSION_FACTORY = AsyncSessionLocal

# 先放开本地前端跨域，后面再收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


def _json_error(
    request: Request,
    *,
    status_code: int,
    error_code: str,
    message: str,
    detail: Any,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    body = {
        "request_id": request_id,
        "error_code": error_code,
        "message": message,
        "detail": detail,
    }
    return JSONResponse(status_code=status_code, content=body, headers={"x-request-id": request_id})


@app.exception_handler(HTTPException)
async def handle_http_exception(request: Request, exc: HTTPException):
    detail = exc.detail if exc.detail is not None else "HTTP error"
    message = detail if isinstance(detail, str) else "HTTP error"
    return _json_error(
        request,
        status_code=exc.status_code,
        error_code="http_error",
        message=message,
        detail=detail,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_exception(request: Request, exc: RequestValidationError):
    return _json_error(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="request_validation_error",
        message="Request validation failed",
        detail=exc.errors(),
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):
    return _json_error(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="internal_server_error",
        message="Internal Server Error",
        detail=str(exc),
    )


def is_newer_view(candidate: KolView, current: KolView) -> bool:
    return (candidate.as_of, candidate.created_at, candidate.id) > (
        current.as_of,
        current.created_at,
        current.id,
    )


def select_latest_views(views: list[KolView]) -> list[KolView]:
    latest_by_key: dict[tuple[int, int, str], KolView] = {}
    for view in views:
        horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
        key = (view.kol_id, view.asset_id, horizon_value)
        prev = latest_by_key.get(key)
        if prev is None or is_newer_view(view, prev):
            latest_by_key[key] = view
    return list(latest_by_key.values())


def build_asset_views_response(asset_id: int, views: list[KolView]) -> AssetViewsRead:
    latest_views = select_latest_views(views)

    grouped: dict[str, dict[Stance, list[KolViewRead]]] = defaultdict(
        lambda: {Stance.bull: [], Stance.bear: [], Stance.neutral: []}
    )
    for view in latest_views:
        horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
        stance_value = view.stance.value if hasattr(view.stance, "value") else str(view.stance)
        stance = Stance(stance_value) if stance_value in {"bull", "bear", "neutral"} else Stance.neutral
        grouped[horizon_value][stance].append(KolViewRead.model_validate(view))

    groups: list[AssetViewsGroupRead] = []
    order_index = {value: idx for idx, value in enumerate(HORIZON_ORDER)}
    for horizon in sorted(grouped.keys(), key=lambda value: order_index.get(value, 999)):
        bull = sorted(grouped[horizon][Stance.bull], key=lambda item: item.confidence, reverse=True)
        bear = sorted(grouped[horizon][Stance.bear], key=lambda item: item.confidence, reverse=True)
        neutral = sorted(grouped[horizon][Stance.neutral], key=lambda item: item.confidence, reverse=True)
        groups.append(
            AssetViewsGroupRead(
                horizon=horizon,
                bull=bull,
                bear=bear,
                neutral=neutral,
            )
        )

    return AssetViewsRead(
        asset_id=asset_id,
        groups=groups,
        meta=AssetViewsMetaRead(
            sort="confidence_desc",
            generated_at=datetime.now(UTC),
            version_policy="latest_per_kol_asset_horizon",
        ),
    )


@app.get("/health")
def health():
    return {"ok": True}


def build_external_id(url: str) -> str:
    url = url.strip()
    if not url:
        return uuid4().hex
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]


def calc_clarity(bull_count: int, bear_count: int) -> float:
    total = bull_count + bear_count
    return abs(bull_count - bear_count) / max(1, total)


def _build_last_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _extractor_provider_output(extractor: DummyExtractor | OpenAIExtractor) -> tuple[str, str]:
    if isinstance(extractor, OpenAIExtractor):
        return extractor.provider_detected, extractor.output_mode
    return ("dummy", "dummy")


def _build_extraction_meta(
    *,
    extracted_json: dict[str, Any],
    extractor: DummyExtractor | OpenAIExtractor,
    fallback_reason: str | None,
    last_error: str | None,
) -> dict[str, Any]:
    base_meta = extracted_json.get("meta") if isinstance(extracted_json.get("meta"), dict) else {}
    safe_meta = dict(base_meta)
    provider_detected, output_mode_used = _extractor_provider_output(extractor)
    safe_meta.setdefault("provider_detected", provider_detected)
    safe_meta.setdefault("output_mode_used", output_mode_used)
    safe_meta.setdefault("parse_strategy_used", "none" if isinstance(extractor, DummyExtractor) else "unknown")
    safe_meta.setdefault("raw_len", 0)
    safe_meta.setdefault("repaired", False)
    if "extraction_mode" not in safe_meta:
        safe_meta["extraction_mode"] = "dummy" if isinstance(extractor, DummyExtractor) else output_mode_used
    if fallback_reason:
        safe_meta["fallback_reason"] = fallback_reason
    if isinstance(extractor, DummyExtractor) and fallback_reason:
        safe_meta["dummy_fallback"] = True
    if last_error:
        safe_meta["parse_error"] = safe_meta.get("parse_error") or ("json" in last_error.lower() and "openai" in last_error.lower())
    return safe_meta


class ClassifiedExtractionState(str, Enum):
    approved = "APPROVED"
    rejected = "REJECTED"
    success = "SUCCESS"
    failed = "FAILED"
    pending = "PENDING"
    no_extraction = "NO_EXTRACTION"


def _is_valid_extracted_object(extracted_json: Any) -> bool:
    if isinstance(extracted_json, dict):
        return True
    if not isinstance(extracted_json, str):
        return False
    try:
        parsed = json.loads(extracted_json)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(parsed, dict)


def classify_extraction_state(
    extraction: PostExtraction | None,
    raw_post: RawPost | None,
) -> ClassifiedExtractionState:
    review_status = _review_status_key(raw_post) if raw_post is not None else ReviewStatus.unreviewed.value
    if review_status == ReviewStatus.rejected.value:
        return ClassifiedExtractionState.rejected
    if review_status == ReviewStatus.approved.value:
        return ClassifiedExtractionState.approved

    if extraction is None:
        return ClassifiedExtractionState.no_extraction

    status_value = extraction.status.value if hasattr(extraction.status, "value") else str(extraction.status)
    status_key = status_value.strip().lower()
    if status_key == ExtractionStatus.rejected.value:
        return ClassifiedExtractionState.rejected
    if status_key == ExtractionStatus.approved.value:
        return ClassifiedExtractionState.approved

    meta = extraction.extracted_json.get("meta") if isinstance(extraction.extracted_json, dict) else None
    parse_error = bool(meta.get("parse_error")) if isinstance(meta, dict) else False
    dummy_fallback = bool(meta.get("dummy_fallback")) if isinstance(meta, dict) else False
    has_last_error = bool((extraction.last_error or "").strip())
    is_dummy = (extraction.extractor_name or "").strip().lower() == "dummy"
    if has_last_error or parse_error or dummy_fallback or is_dummy:
        return ClassifiedExtractionState.failed

    if _is_valid_extracted_object(extraction.extracted_json):
        return ClassifiedExtractionState.success

    if status_key == ExtractionStatus.pending.value:
        return ClassifiedExtractionState.pending
    return ClassifiedExtractionState.pending


def _is_failed_extraction(extraction: PostExtraction | None, raw_post: RawPost | None = None) -> bool:
    return classify_extraction_state(extraction, raw_post) == ClassifiedExtractionState.failed


def _is_active_extraction(extraction: PostExtraction | None, raw_post: RawPost | None = None) -> bool:
    return classify_extraction_state(extraction, raw_post) == ClassifiedExtractionState.pending


def _is_successful_extraction(extraction: PostExtraction | None, raw_post: RawPost | None = None) -> bool:
    return classify_extraction_state(extraction, raw_post) in {
        ClassifiedExtractionState.success,
        ClassifiedExtractionState.approved,
    }


def _has_extracted_result(extraction: PostExtraction | None, raw_post: RawPost | None = None) -> bool:
    return classify_extraction_state(extraction, raw_post) in {
        ClassifiedExtractionState.success,
        ClassifiedExtractionState.approved,
        ClassifiedExtractionState.rejected,
    }


def _is_result_available_extraction(extraction: PostExtraction | None, raw_post: RawPost | None = None) -> bool:
    return _has_extracted_result(extraction, raw_post)


def _extraction_status_key(extraction: PostExtraction) -> str:
    status_value = extraction.status.value if hasattr(extraction.status, "value") else str(extraction.status)
    return status_value.strip().lower()


def _review_status_key(raw_post: RawPost) -> str:
    value = getattr(raw_post, "review_status", None)
    if hasattr(value, "value"):
        value = value.value
    status_key = str(value or "").strip().lower()
    return status_key or ReviewStatus.unreviewed.value


def _terminal_review_skip_kind(
    *,
    raw_post: RawPost,
    latest_extraction: PostExtraction | None,
) -> str | None:
    state = classify_extraction_state(latest_extraction, raw_post)
    if state == ClassifiedExtractionState.rejected:
        return "skipped_already_rejected"
    if state == ClassifiedExtractionState.approved:
        return "skipped_already_approved"
    return None


def _classify_last_error(last_error: str | None) -> str:
    message = (last_error or "").lower()
    if "status=429" in message or " 429" in message or "rate_limit" in message:
        return "rate_limited"
    if "timeout" in message:
        return "timeout"
    if "status=503" in message or " 503" in message:
        return "service_unavailable"
    return "other"


def _normalize_handle(value: str) -> str:
    return value.strip().lower()


def _normalize_author_handle(value: str) -> str:
    return value.strip().lstrip("@").lower()


def _coerce_row_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _raw_snippet(value: Any, *, max_chars: int = 240) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:  # noqa: BLE001
        text = repr(value)
    return text[:max_chars]


def _detect_x_export_kind(rows: list[Any]) -> str:
    if not rows:
        return "unknown"
    sampled = 0
    following_hits = 0
    timeline_hits = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        sampled += 1
        has_following = "following" in row
        has_screen_name = isinstance(row.get("screen_name"), str) and bool(row.get("screen_name").strip())
        has_name = isinstance(row.get("name"), str)
        has_timeline_id = row.get("id") is not None
        has_created = row.get("created_at") is not None
        has_text = row.get("full_text") is not None or row.get("text") is not None
        if has_following and has_screen_name and has_name:
            following_hits += 1
        if has_timeline_id and has_screen_name and has_created and has_text:
            timeline_hits += 1
        if sampled >= 25:
            break
    if following_hits > 0 and following_hits >= timeline_hits:
        return "following"
    if timeline_hits > 0:
        return "timeline"
    return "generic"


async def _enabled_x_kol_maps(db: AsyncSession) -> tuple[dict[int, Kol], dict[str, Kol]]:
    result = await db.execute(select(Kol).where(Kol.platform == "x"))
    enabled = [item for item in result.scalars().all() if bool(item.enabled)]
    return (
        {item.id: item for item in enabled},
        {_normalize_author_handle(item.handle): item for item in enabled},
    )


async def _raw_post_matches_enabled_x_kol(db: AsyncSession, raw_post: RawPost) -> bool:
    if raw_post.platform.strip().lower() != "x":
        return True
    enabled_by_id, enabled_by_handle = await _enabled_x_kol_maps(db)
    if not enabled_by_id and not enabled_by_handle:
        return True
    if raw_post.kol_id is not None and raw_post.kol_id in enabled_by_id:
        return True
    handle_key = _normalize_author_handle(raw_post.author_handle or "")
    if not handle_key:
        return False
    return handle_key in enabled_by_handle


def _is_destructive_admin_enabled() -> bool:
    settings = get_settings()
    env_key = (settings.env or "").strip().lower()
    return env_key in {"local", "dev", "development"} or bool(settings.debug)


def _require_destructive_admin_guard(*, confirm: str) -> None:
    if not _is_destructive_admin_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="forbidden: destructive endpoint enabled only when ENV in {local,dev} or DEBUG=true",
        )
    if confirm != "YES":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="destructive operation: set query confirm=YES to continue",
        )


def _require_raw_delete_cascade_guard(*, also_delete_raw_posts: bool, enable_cascade: bool) -> None:
    if also_delete_raw_posts and not enable_cascade:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="raw_posts deletion requires enable_cascade=true and confirm=YES",
        )


def _admin_delete_counts_template() -> dict[str, int]:
    return {
        "kols": 0,
        "assets": 0,
        "asset_aliases": 0,
        "raw_posts": 0,
        "post_extractions": 0,
        "kol_views": 0,
        "daily_digests": 0,
        "profile_kol_weights": 0,
    }


def _digest_mentions_kol(content: dict | None, *, kol_id: int) -> bool:
    if not isinstance(content, dict):
        return False
    per_asset_summary = content.get("per_asset_summary")
    if not isinstance(per_asset_summary, list):
        return False
    for item in per_asset_summary:
        if not isinstance(item, dict):
            continue
        for stance_key in ("top_views_bull", "top_views_bear", "top_views_neutral"):
            top_views = item.get(stance_key)
            if not isinstance(top_views, list):
                continue
            for view in top_views:
                if isinstance(view, dict) and view.get("kol_id") == kol_id:
                    return True
    return False


def _digest_mentions_asset(content: dict | None, *, asset_id: int) -> bool:
    if not isinstance(content, dict):
        return False
    top_assets = content.get("top_assets")
    if isinstance(top_assets, list):
        for item in top_assets:
            if isinstance(item, dict) and item.get("asset_id") == asset_id:
                return True
    per_asset_summary = content.get("per_asset_summary")
    if isinstance(per_asset_summary, list):
        for item in per_asset_summary:
            if isinstance(item, dict) and item.get("asset_id") == asset_id:
                return True
    return False


def _build_fk_conflict_detail(*, action: str, hint: str) -> str:
    return f"delete conflict during {action}: {hint}. Please delete dependent records first and retry."


def _format_retry_failure_reason(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return f"http_{exc.status_code}:{detail}"
    return type(exc).__name__


def reset_reextract_rate_limiter() -> None:
    REEXTRACT_ATTEMPTS.clear()


def reset_runtime_counters() -> None:
    global RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE, RUNTIME_BUDGET_WINDOW_START, RUNTIME_BUDGET_WINDOW_END
    RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE = None
    RUNTIME_BURST_STATE["enabled"] = False
    RUNTIME_BURST_STATE["mode"] = None
    RUNTIME_BURST_STATE["call_budget"] = None
    RUNTIME_BURST_STATE["expires_at"] = None
    RUNTIME_THROTTLE_OVERRIDE.clear()
    RUNTIME_BUDGET_WINDOW_START = None
    RUNTIME_BUDGET_WINDOW_END = None
    EXTRACT_JOB_IDEMPOTENCY.clear()
    reset_reextract_rate_limiter()
    reset_openai_call_budget_counter()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _get_default_call_budget_total(settings) -> int:
    fields_set = getattr(settings, "model_fields_set", set())
    if "call_budget_per_hour" in fields_set:
        return max(0, settings.call_budget_per_hour)
    if settings.openai_call_budget is not None:
        return max(0, settings.openai_call_budget)
    return max(0, settings.call_budget_per_hour)


def _window_for(now: datetime, minutes: int) -> tuple[datetime, datetime]:
    safe_minutes = max(1, minutes)
    window_seconds = safe_minutes * 60
    timestamp = int(now.timestamp())
    start_epoch = (timestamp // window_seconds) * window_seconds
    start = datetime.fromtimestamp(start_epoch, tz=UTC)
    end = start + timedelta(seconds=window_seconds)
    return start, end


def _refresh_runtime_budget_state(settings) -> None:
    global RUNTIME_BUDGET_WINDOW_START, RUNTIME_BUDGET_WINDOW_END
    now = _utc_now()
    burst_enabled = bool(RUNTIME_BURST_STATE.get("enabled"))
    expires_at = RUNTIME_BURST_STATE.get("expires_at")
    if burst_enabled and isinstance(expires_at, datetime) and now >= expires_at:
        RUNTIME_BURST_STATE["enabled"] = False
        RUNTIME_BURST_STATE["mode"] = None
        RUNTIME_BURST_STATE["call_budget"] = None
        RUNTIME_BURST_STATE["expires_at"] = None
        RUNTIME_BUDGET_WINDOW_START = None
        RUNTIME_BUDGET_WINDOW_END = None
        reset_openai_call_budget_counter()

    if RUNTIME_BUDGET_WINDOW_START is None or RUNTIME_BUDGET_WINDOW_END is None:
        start, end = _window_for(now, settings.call_budget_window_minutes)
        RUNTIME_BUDGET_WINDOW_START = start
        RUNTIME_BUDGET_WINDOW_END = end
        reset_openai_call_budget_counter()
        return

    if now >= RUNTIME_BUDGET_WINDOW_END:
        start, end = _window_for(now, settings.call_budget_window_minutes)
        RUNTIME_BUDGET_WINDOW_START = start
        RUNTIME_BUDGET_WINDOW_END = end
        reset_openai_call_budget_counter()


def _get_runtime_openai_call_budget_total(settings) -> int:
    _refresh_runtime_budget_state(settings)
    if RUNTIME_BURST_STATE["enabled"] and isinstance(RUNTIME_BURST_STATE["call_budget"], int):
        return max(0, RUNTIME_BURST_STATE["call_budget"])
    if RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE is not None:
        return max(0, RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE)
    return _get_default_call_budget_total(settings)


def _set_runtime_openai_call_budget_total(call_budget: int) -> None:
    global RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE
    normalized = max(0, call_budget)
    RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE = normalized
    _refresh_runtime_budget_state(get_settings())
    reset_openai_call_budget_counter()


def _clear_runtime_openai_call_budget_override() -> None:
    global RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE
    RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE = None
    _refresh_runtime_budget_state(get_settings())
    reset_openai_call_budget_counter()


def _set_runtime_burst(*, enabled: bool, mode: str, call_budget: int, duration_minutes: int) -> None:
    settings = get_settings()
    _refresh_runtime_budget_state(settings)
    if not enabled:
        RUNTIME_BURST_STATE["enabled"] = False
        RUNTIME_BURST_STATE["mode"] = None
        RUNTIME_BURST_STATE["call_budget"] = None
        RUNTIME_BURST_STATE["expires_at"] = None
        reset_openai_call_budget_counter()
        return
    now = _utc_now()
    resolved_mode = "unlimited_safe" if mode == "unlimited_safe" else "normal"
    resolved_budget = UNLIMITED_SAFE_BURST_BUDGET if resolved_mode == "unlimited_safe" else max(0, int(call_budget))
    RUNTIME_BURST_STATE["enabled"] = True
    RUNTIME_BURST_STATE["mode"] = resolved_mode
    RUNTIME_BURST_STATE["call_budget"] = resolved_budget
    RUNTIME_BURST_STATE["expires_at"] = now + timedelta(minutes=max(1, int(duration_minutes)))
    reset_openai_call_budget_counter()


def _runtime_throttle_limits(settings) -> tuple[int, int, int, int]:
    return (
        max(1, settings.extract_max_concurrency_max),
        max(1, settings.extract_max_rpm_max),
        max(1, settings.extract_batch_size_max),
        max(1, settings.extract_batch_sleep_ms_min),
    )


def _clamp_runtime_throttle(settings, *, max_concurrency: int, max_rpm: int, batch_size: int, batch_sleep_ms: int) -> dict[str, int]:
    c_max, rpm_max, batch_max, sleep_min = _runtime_throttle_limits(settings)
    return {
        "max_concurrency": max(1, min(int(max_concurrency), c_max)),
        "max_rpm": max(1, min(int(max_rpm), rpm_max)),
        "batch_size": max(1, min(int(batch_size), batch_max)),
        "batch_sleep_ms": max(sleep_min, int(batch_sleep_ms)),
    }


def _get_runtime_throttle(settings) -> dict[str, int]:
    defaults = _clamp_runtime_throttle(
        settings,
        max_concurrency=settings.extract_max_concurrency_default,
        max_rpm=settings.extract_max_rpm_default,
        batch_size=settings.extract_batch_size_default,
        batch_sleep_ms=settings.extract_batch_sleep_ms_default,
    )
    if not RUNTIME_THROTTLE_OVERRIDE:
        return defaults
    return _clamp_runtime_throttle(
        settings,
        max_concurrency=RUNTIME_THROTTLE_OVERRIDE.get("max_concurrency", defaults["max_concurrency"]),
        max_rpm=RUNTIME_THROTTLE_OVERRIDE.get("max_rpm", defaults["max_rpm"]),
        batch_size=RUNTIME_THROTTLE_OVERRIDE.get("batch_size", defaults["batch_size"]),
        batch_sleep_ms=RUNTIME_THROTTLE_OVERRIDE.get("batch_sleep_ms", defaults["batch_sleep_ms"]),
    )


def _set_runtime_throttle(settings, *, max_concurrency: int, max_rpm: int, batch_size: int, batch_sleep_ms: int) -> dict[str, int]:
    clamped = _clamp_runtime_throttle(
        settings,
        max_concurrency=max_concurrency,
        max_rpm=max_rpm,
        batch_size=batch_size,
        batch_sleep_ms=batch_sleep_ms,
    )
    RUNTIME_THROTTLE_OVERRIDE.clear()
    RUNTIME_THROTTLE_OVERRIDE.update(clamped)
    return clamped


def _check_reextract_rate_limit(raw_post_id: int) -> None:
    settings = get_settings()
    window = timedelta(seconds=max(1, settings.reextract_rate_limit_window_seconds))
    max_attempts = max(1, settings.reextract_rate_limit_max_attempts)
    now = datetime.now(UTC)

    attempts = REEXTRACT_ATTEMPTS[raw_post_id]
    while attempts and now - attempts[0] > window:
        attempts.popleft()

    if len(attempts) >= max_attempts:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"re-extract rate limit exceeded: max {max_attempts} per {window.seconds}s",
        )

    attempts.append(now)


class _RpmLimiter:
    def __init__(self, max_rpm: int) -> None:
        self.interval_seconds = 60.0 / max(1, max_rpm)
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait_seconds = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + self.interval_seconds
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)


def _is_retryable_extraction_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        return exc.status_code in {429, 503}
    if isinstance(exc, OpenAIFallbackError):
        return False
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {429, 503}:
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    message = str(exc).lower()
    return "timeout" in message and "http_" not in message


def _build_retry_backoff_seconds(*, attempt: int, base_ms: int, cap_ms: int) -> float:
    exp = max(0, attempt - 1)
    raw_ms = min(cap_ms, base_ms * (2**exp))
    jitter_multiplier = random.uniform(0.8, 1.2)
    return max(0.0, (raw_ms * jitter_multiplier) / 1000.0)


def _build_extraction_input(raw_post: RawPost) -> tuple[RawPost | SimpleNamespace, dict | None]:
    settings = get_settings()
    max_length = max(1, settings.extraction_max_content_chars)
    original_content = raw_post.content_text or ""
    original_length = len(original_content)
    if original_length <= max_length:
        return raw_post, None

    truncated_content = original_content[:max_length]
    extraction_input = SimpleNamespace(
        platform=raw_post.platform,
        author_handle=raw_post.author_handle,
        url=raw_post.url,
        posted_at=raw_post.posted_at,
        content_text=truncated_content,
    )
    return extraction_input, {
        "truncated": True,
        "original_length": original_length,
        "max_length": max_length,
    }


def _normalize_asset_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    symbol = value.strip().upper()
    return symbol or None


def _infer_auto_market(raw_post: RawPost | SimpleNamespace, symbol: str) -> str:
    if getattr(raw_post, "platform", "").strip().lower() in {"binance", "okx", "bybit"}:
        return "CRYPTO"
    if symbol.endswith("USD") or symbol in {"US10Y", "DXY", "VIX", "SPX", "QQQ"}:
        return "AUTO"
    return "AUTO"


async def _load_assets_for_prompt(db: AsyncSession, limit: int) -> list[dict[str, str | None]]:
    safe_limit = max(0, limit)
    if safe_limit == 0:
        return []
    result = await db.execute(select(Asset).order_by(Asset.id.asc()).limit(safe_limit))
    items = list(result.scalars().all())
    return [{"symbol": item.symbol, "name": item.name, "market": item.market} for item in items]


async def _load_aliases_for_prompt(db: AsyncSession) -> list[dict[str, str]]:
    result = await db.execute(
        select(AssetAlias, Asset.symbol)
        .join(Asset, Asset.id == AssetAlias.asset_id)
        .order_by(AssetAlias.id.asc())
    )
    aliases: list[dict[str, str]] = []
    for alias, symbol in result.all():
        aliases.append({"alias": alias.alias, "symbol": symbol})
    return aliases


async def _load_known_asset_symbols(db: AsyncSession) -> set[str]:
    result = await db.execute(select(Asset).order_by(Asset.id.asc()))
    symbols: set[str] = set()
    for asset in result.scalars().all():
        symbol = getattr(asset, "symbol", None)
        if isinstance(symbol, str) and symbol.strip():
            symbols.add(symbol.strip().upper())
    return symbols


def _normalize_alias_key(value: str) -> str:
    return value.strip().lower()


def _build_alias_to_symbol_map(aliases: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in aliases:
        alias = _normalize_alias_key(item.get("alias", ""))
        symbol = item.get("symbol", "").strip().upper()
        if alias and symbol:
            mapping[alias] = symbol
    return mapping


def _extract_directly_mentioned_symbols(
    text: str = "",
    *,
    content_text: str | None = None,
    alias_to_symbol: dict[str, str] | None = None,
    known_symbols: set[str] | None = None,
) -> list[str]:
    source_text = content_text if content_text is not None else text
    if not isinstance(source_text, str) or not source_text.strip():
        return []

    direct_mentions: list[str] = []
    seen: set[str] = set()

    def _push(symbol: str) -> None:
        normalized = symbol.strip().upper()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        direct_mentions.append(normalized)

    if known_symbols:
        normalized_known = sorted(
            {
                item.strip().upper()
                for item in known_symbols
                if isinstance(item, str) and item.strip()
            },
            key=len,
            reverse=True,
        )
        for symbol in normalized_known:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", source_text, flags=re.IGNORECASE):
                _push(symbol)

    if alias_to_symbol:
        lowered_text = source_text.lower()
        normalized_aliases = sorted(
            (
                (_normalize_alias_key(alias), symbol.strip().upper())
                for alias, symbol in alias_to_symbol.items()
                if isinstance(alias, str) and isinstance(symbol, str) and alias.strip() and symbol.strip()
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        for alias_key, symbol in normalized_aliases:
            if alias_key and alias_key in lowered_text:
                _push(symbol)

    return direct_mentions


async def upsert_asset(
    db: AsyncSession,
    *,
    symbol: str,
    name: str | None = None,
    market: str | None = None,
) -> Asset:
    symbol_normalized = symbol.strip().upper()
    if not symbol_normalized:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="symbol is required")

    existing_result = await db.execute(select(Asset).where(Asset.symbol == symbol_normalized))
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        updated = False
        normalized_name = (name or "").strip() or None
        normalized_market = (market or "").strip() or None
        if normalized_name and not existing.name:
            existing.name = normalized_name
            updated = True
        if normalized_market and not existing.market:
            existing.market = normalized_market
            updated = True
        if updated:
            await db.flush()
        return existing

    asset = Asset(
        symbol=symbol_normalized,
        name=(name or "").strip() or None,
        market=(market or "").strip() or None,
    )
    db.add(asset)
    await db.flush()
    return asset


async def ensure_assets_from_extracted_json(
    db: AsyncSession,
    *,
    raw_post: RawPost | SimpleNamespace,
    extracted_json: dict[str, Any],
) -> None:
    seen_symbols: set[str] = set()
    candidates: list[Any] = []
    assets = extracted_json.get("assets")
    if isinstance(assets, list):
        candidates.extend(assets)
    asset_views = extracted_json.get("asset_views")
    if isinstance(asset_views, list):
        candidates.extend(asset_views)
    if not candidates:
        return

    for item in candidates:
        if isinstance(item, str):
            symbol = _normalize_asset_symbol(item)
            name = None
            market = None
        elif isinstance(item, dict):
            symbol = _normalize_asset_symbol(item.get("symbol"))
            name = item.get("name") if isinstance(item.get("name"), str) else None
            market = item.get("market") if isinstance(item.get("market"), str) else None
        else:
            continue

        if symbol is None or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        await upsert_asset(
            db,
            symbol=symbol,
            name=name,
            market=market or _infer_auto_market(raw_post, symbol),
        )


def _resolve_extractor_name(
    *,
    extractor: DummyExtractor | OpenAIExtractor,
    settings_base_url: str,
    extracted_json: dict[str, Any],
) -> str:
    if isinstance(extractor, DummyExtractor):
        return extractor.extractor_name

    meta = extracted_json.get("meta")
    output_mode_used = meta.get("output_mode_used") if isinstance(meta, dict) else None
    extraction_mode = meta.get("extraction_mode") if isinstance(meta, dict) else None
    if (
        output_mode_used == EXTRACTION_OUTPUT_TEXT_JSON
        or extraction_mode == EXTRACTION_OUTPUT_TEXT_JSON
    ) and "openrouter.ai" in settings_base_url.lower():
        return "openrouter_json_mode"
    return extractor.extractor_name


async def _upsert_kol_by_author(db: AsyncSession, *, platform: str, handle: str) -> Kol:
    platform_normalized = platform.strip().lower()
    handle_normalized = handle.strip().lower()
    result = await db.execute(
        select(Kol).where(
            Kol.platform == platform_normalized,
            Kol.handle == handle_normalized,
        )
    )
    existing = result.scalar_one_or_none()
    if existing is not None:
        if not existing.enabled:
            existing.enabled = True
            await db.flush()
        return existing

    kol = Kol(
        platform=platform_normalized,
        handle=handle_normalized,
        display_name=None,
        enabled=True,
    )
    db.add(kol)
    await db.flush()
    return kol


async def _find_existing_kol_view(
    db: AsyncSession,
    *,
    kol_id: int,
    asset_id: int,
    horizon: str,
    as_of: date,
) -> KolView | None:
    result = await db.execute(
        select(KolView).where(
            KolView.kol_id == kol_id,
            KolView.asset_id == asset_id,
            KolView.horizon == horizon,
            KolView.as_of == as_of,
        )
    )
    candidates = list(result.scalars().all())
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.created_at or datetime.min.replace(tzinfo=UTC),
            item.id or 0,
        ),
        reverse=True,
    )
    keep = candidates[0]
    for duplicated in candidates[1:]:
        await db.delete(duplicated)
    if len(candidates) > 1:
        await db.flush()
    return keep


def _build_kol_asset_view_key(*, kol_id: int, asset_id: int, horizon: str, as_of: date) -> str:
    return f"{kol_id}:{asset_id}:{horizon}:{as_of.isoformat()}"


async def _insert_kol_view_if_absent(
    db: AsyncSession,
    *,
    kol_id: int,
    asset_id: int,
    stance: str,
    horizon: str,
    confidence: int,
    summary: str,
    source_url: str,
    as_of: date,
) -> tuple[KolView, bool, str]:
    key = _build_kol_asset_view_key(kol_id=kol_id, asset_id=asset_id, horizon=horizon, as_of=as_of)
    existing = await _find_existing_kol_view(
        db,
        kol_id=kol_id,
        asset_id=asset_id,
        horizon=horizon,
        as_of=as_of,
    )
    if existing is not None:
        return existing, False, key

    view = KolView(
        kol_id=kol_id,
        asset_id=asset_id,
        stance=stance,
        horizon=horizon,
        confidence=confidence,
        summary=summary,
        source_url=source_url,
        as_of=as_of,
    )
    db.add(view)
    await db.flush()
    return view, True, key


def _iter_asset_view_candidates(extracted_json: dict[str, Any]) -> list[dict[str, Any]]:
    asset_views = extracted_json.get("asset_views")
    if not isinstance(asset_views, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in asset_views:
        if not isinstance(item, dict):
            continue
        symbol = _normalize_asset_symbol(item.get("symbol"))
        if symbol is None:
            continue
        stance = item.get("stance")
        horizon = item.get("horizon")
        confidence_raw = item.get("confidence")
        if stance not in {"bull", "bear", "neutral"}:
            continue
        if horizon not in {"intraday", "1w", "1m", "3m", "1y"}:
            continue
        if not isinstance(confidence_raw, (int, float)):
            continue
        confidence = int(max(0, min(100, round(float(confidence_raw)))))
        reasoning = item.get("reasoning") if isinstance(item.get("reasoning"), str) else None
        summary = item.get("summary") if isinstance(item.get("summary"), str) else None
        candidates.append(
            {
                "symbol": symbol,
                "stance": stance,
                "horizon": horizon,
                "confidence": confidence,
                "reasoning": reasoning,
                "summary": summary,
            }
        )
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates


async def _auto_apply_extraction_views(
    db: AsyncSession,
    *,
    extraction: PostExtraction,
    raw_post: RawPost,
    force_apply: bool = False,
) -> None:
    settings = get_settings()
    extraction.auto_applied_count = 0
    extraction.auto_policy = None
    extraction.auto_applied_kol_view_ids = None
    setattr(extraction, "auto_applied_asset_view_keys", [])
    setattr(extraction, "auto_applied_views", [])
    if (not settings.auto_approve_enabled and not force_apply) or extraction.status != ExtractionStatus.pending:
        print(
            "[auto-approve]",
            {
                "AUTO_APPROVE_ENABLED": settings.auto_approve_enabled,
                "threshold": settings.auto_approve_confidence_threshold,
                "min_display": settings.auto_approve_min_display_confidence,
                "candidates": 0,
                "applied": 0,
                "policy": None,
            },
        )
        return

    candidates = _iter_asset_view_candidates(extraction.extracted_json)
    if not candidates:
        print(
            "[auto-approve]",
            {
                "AUTO_APPROVE_ENABLED": settings.auto_approve_enabled,
                "threshold": settings.auto_approve_confidence_threshold,
                "min_display": settings.auto_approve_min_display_confidence,
                "candidates": 0,
                "applied": 0,
                "policy": None,
            },
        )
        return
    min_display = max(0, min(100, settings.auto_approve_min_display_confidence))
    candidates = [item for item in candidates if item["confidence"] >= min_display]
    if not candidates:
        print(
            "[auto-approve]",
            {
                "AUTO_APPROVE_ENABLED": settings.auto_approve_enabled,
                "threshold": settings.auto_approve_confidence_threshold,
                "min_display": settings.auto_approve_min_display_confidence,
                "candidates": 0,
                "applied": 0,
                "policy": None,
            },
        )
        return
    max_views = max(1, settings.auto_approve_max_views)
    threshold = max(0, min(100, settings.auto_approve_confidence_threshold))
    candidates = candidates[:max_views]
    selected = [item for item in candidates if item["confidence"] >= threshold]
    policy: str | None = None
    if selected:
        policy = "threshold"
    else:
        selected = candidates[:1]
        policy = "top1_fallback" if selected else None
    if not selected:
        return

    kol = await _upsert_kol_by_author(db, platform=raw_post.platform, handle=raw_post.author_handle)
    created_ids: list[int] = []
    applied_keys: list[str] = []
    applied_views: list[AutoAppliedViewRead] = []
    skipped_count = 0
    for item in selected:
        asset = await upsert_asset(
            db,
            symbol=item["symbol"],
            market=_infer_auto_market(raw_post, item["symbol"]),
        )
        summary = (item.get("summary") or item.get("reasoning") or extraction.extracted_json.get("summary") or "").strip()
        if not summary:
            summary = f"{item['symbol']} {item['stance']} ({item['horizon']})"
        source_url = (extraction.extracted_json.get("source_url") or raw_post.url or "").strip()
        if not source_url:
            source_url = raw_post.url
        as_of = raw_post.posted_at.date()
        extracted_as_of = extraction.extracted_json.get("as_of")
        if isinstance(extracted_as_of, str):
            try:
                as_of = datetime.fromisoformat(extracted_as_of).date()
            except ValueError:
                pass
        view, inserted, key = await _insert_kol_view_if_absent(
            db,
            kol_id=kol.id,
            asset_id=asset.id,
            stance=item["stance"],
            horizon=item["horizon"],
            confidence=item["confidence"],
            summary=summary[:1024],
            source_url=source_url[:1024],
            as_of=as_of,
        )
        if not inserted:
            skipped_count += 1
            continue
        created_ids.append(view.id)
        applied_keys.append(key)
        applied_views.append(
            AutoAppliedViewRead(
                kol_view_id=view.id,
                symbol=asset.symbol,
                asset_id=asset.id,
                stance=view.stance,
                horizon=view.horizon,
                as_of=view.as_of,
                confidence=view.confidence,
            )
        )

    extraction.auto_applied_count = len(created_ids)
    extraction.auto_policy = policy
    extraction.auto_applied_kol_view_ids = created_ids
    if created_ids and extraction.status == ExtractionStatus.pending:
        extraction.status = ExtractionStatus.approved
        extraction.reviewed_by = AUTO_EXTRACTION_REVIEWER
        extraction.reviewed_at = datetime.now(UTC)
        if not extraction.review_note:
            extraction.review_note = "auto-approved"
        raw_post.review_status = ReviewStatus.approved
        raw_post.reviewed_by = AUTO_EXTRACTION_REVIEWER
        raw_post.reviewed_at = extraction.reviewed_at
        await db.flush()
    setattr(extraction, "auto_applied_asset_view_keys", applied_keys)
    setattr(extraction, "auto_applied_views", applied_views)
    print(
        "[auto-approve]",
        {
            "AUTO_APPROVE_ENABLED": settings.auto_approve_enabled,
            "threshold": threshold,
            "min_display": min_display,
            "candidates": len(candidates),
            "applied": len(created_ids),
            "skipped": skipped_count,
            "policy": policy,
        },
    )


async def _attach_auto_applied_metadata(db: AsyncSession, extraction: PostExtraction) -> None:
    kol_view_ids = extraction.auto_applied_kol_view_ids
    if not isinstance(kol_view_ids, list) or not kol_view_ids:
        setattr(extraction, "auto_applied_asset_view_keys", None)
        setattr(extraction, "auto_applied_views", None)
        return

    auto_applied_asset_view_keys: list[str] = []
    auto_applied_views: list[AutoAppliedViewRead] = []
    for kol_view_id in kol_view_ids:
        if not isinstance(kol_view_id, int):
            continue
        kol_view = await db.get(KolView, kol_view_id)
        if kol_view is None:
            continue
        asset = await db.get(Asset, kol_view.asset_id)
        symbol = asset.symbol if asset is not None else str(kol_view.asset_id)
        horizon = kol_view.horizon.value if hasattr(kol_view.horizon, "value") else str(kol_view.horizon)
        stance = kol_view.stance.value if hasattr(kol_view.stance, "value") else str(kol_view.stance)
        auto_applied_asset_view_keys.append(
            _build_kol_asset_view_key(
                kol_id=kol_view.kol_id,
                asset_id=kol_view.asset_id,
                horizon=horizon,
                as_of=kol_view.as_of,
            )
        )
        auto_applied_views.append(
            AutoAppliedViewRead(
                kol_view_id=kol_view.id,
                symbol=symbol,
                asset_id=kol_view.asset_id,
                stance=stance,
                horizon=horizon,
                as_of=kol_view.as_of,
                confidence=kol_view.confidence,
            )
        )

    setattr(extraction, "auto_applied_asset_view_keys", auto_applied_asset_view_keys)
    setattr(extraction, "auto_applied_views", auto_applied_views)


def _attach_extraction_auto_approve_settings(extraction: PostExtraction) -> None:
    settings = get_settings()
    setattr(extraction, "auto_approve_confidence_threshold", settings.auto_approve_confidence_threshold)
    setattr(extraction, "auto_approve_min_display_confidence", settings.auto_approve_min_display_confidence)
    setattr(extraction, "auto_reject_confidence_threshold", max(0, min(100, settings.auto_reject_confidence_threshold)))


def _coerce_extraction_confidence(extracted_json: dict[str, Any]) -> int | None:
    confidence_raw = extracted_json.get("confidence")
    if not isinstance(confidence_raw, (int, float)):
        return None
    return int(max(0, min(100, round(float(confidence_raw)))))


async def postprocess_auto_review(
    *,
    db: AsyncSession,
    extraction: PostExtraction,
    raw_post: RawPost,
) -> str | None:
    threshold = AUTO_REVIEW_CONFIDENCE_THRESHOLD
    if classify_extraction_state(extraction, raw_post) != ClassifiedExtractionState.success:
        return None
    if not isinstance(extraction.extracted_json, dict):
        return None
    model_confidence = _coerce_extraction_confidence(extraction.extracted_json)
    if model_confidence is None:
        return None

    base_meta = extraction.extracted_json.get("meta") if isinstance(extraction.extracted_json.get("meta"), dict) else {}
    reviewed_at = datetime.now(UTC)
    if model_confidence < threshold:
        extraction.extracted_json = {
            **extraction.extracted_json,
            "meta": {
                **base_meta,
                "auto_rejected": True,
                "auto_review_threshold": threshold,
                "auto_review_reason": "confidence_below_threshold",
                "model_confidence": model_confidence,
                "auto_reject_reason": "confidence_below_threshold",
                "auto_reject_threshold": threshold,
            },
        }
        extraction.status = ExtractionStatus.rejected
        extraction.reviewed_by = AUTO_EXTRACTION_REVIEWER
        extraction.reviewed_at = reviewed_at
        extraction.review_note = "auto-rejected: confidence_below_threshold"
        raw_post.review_status = ReviewStatus.rejected
        raw_post.reviewed_by = AUTO_EXTRACTION_REVIEWER
        raw_post.reviewed_at = reviewed_at
        return "rejected"

    extraction.extracted_json = {
        **extraction.extracted_json,
        "meta": {
            **base_meta,
            "auto_approved": True,
            "auto_review_threshold": threshold,
            "auto_review_reason": "confidence_at_or_above_threshold",
            "model_confidence": model_confidence,
        },
    }
    await _auto_apply_extraction_views(db, extraction=extraction, raw_post=raw_post, force_apply=True)
    if extraction.status == ExtractionStatus.pending:
        extraction.status = ExtractionStatus.approved
        extraction.reviewed_by = AUTO_EXTRACTION_REVIEWER
        extraction.reviewed_at = reviewed_at
        if not extraction.review_note:
            extraction.review_note = "auto-approved"
        raw_post.review_status = ReviewStatus.approved
        raw_post.reviewed_by = AUTO_EXTRACTION_REVIEWER
        raw_post.reviewed_at = reviewed_at
    return "approved"


async def create_pending_extraction(
    db: AsyncSession,
    raw_post: RawPost,
    *,
    allow_budget_fallback: bool = True,
    raise_retryable_errors: bool = False,
    force_reextract: bool = False,
    force_reextract_triggered_by: str | None = None,
    source_extraction_id: int | None = None,
) -> PostExtraction:
    if not force_reextract:
        if isinstance(db, AsyncSession):
            await db.execute(select(RawPost.id).where(RawPost.id == raw_post.id).with_for_update())
        result = await db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post.id))
        existing_rows = list(result.scalars().all())
        existing_rows.sort(
            key=lambda item: (
                item.created_at or datetime.min.replace(tzinfo=UTC),
                item.id or 0,
            ),
            reverse=True,
        )
        active = next((item for item in existing_rows if _is_active_extraction(item, raw_post)), None)
        if active is not None:
            _attach_extraction_auto_approve_settings(active)
            return active
        latest = existing_rows[0] if existing_rows else None
        terminal_skip_kind = _terminal_review_skip_kind(raw_post=raw_post, latest_extraction=latest)
        if terminal_skip_kind is not None:
            if latest is not None:
                _attach_extraction_auto_approve_settings(latest)
                return latest
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"raw post already {terminal_skip_kind.replace('skipped_already_', '')}",
            )
        any_result = next((item for item in existing_rows if _is_result_available_extraction(item, raw_post)), None)
        if any_result is not None:
            _attach_extraction_auto_approve_settings(any_result)
            return any_result

    settings = get_settings()
    _refresh_runtime_budget_state(settings)
    extraction_input, truncation_meta = _build_extraction_input(raw_post)
    extractor = select_extractor(settings)
    assets_for_prompt = await _load_assets_for_prompt(db, settings.max_assets_in_prompt)
    try:
        aliases_for_prompt = await _load_aliases_for_prompt(db)
    except Exception:  # noqa: BLE001
        aliases_for_prompt = []
    alias_to_symbol = _build_alias_to_symbol_map(aliases_for_prompt)
    known_symbols = await _load_known_asset_symbols(db)
    prompt_bundle = build_extract_prompt(
        prompt_version=settings.prompt_version,
        platform=extraction_input.platform,
        author_handle=extraction_input.author_handle,
        url=extraction_input.url,
        posted_at=extraction_input.posted_at,
        content_text=extraction_input.content_text,
        assets=assets_for_prompt,
        aliases=aliases_for_prompt,
        max_assets_in_prompt=settings.max_assets_in_prompt,
    )
    extractor.set_prompt_bundle(prompt_bundle)
    extracted_json = default_extracted_json(extraction_input)
    last_error: str | None = None
    budget_exhausted = False
    fallback_reason: str | None = None
    force_fail_reason: str | None = None
    allow_dummy_fallback = bool(settings.dummy_fallback) or settings.extractor_mode.strip().lower() == "dummy"

    runtime_budget_total = _get_runtime_openai_call_budget_total(settings)
    if isinstance(extractor, OpenAIExtractor) and not try_consume_openai_call_budget(
        settings,
        budget_total=runtime_budget_total,
    ):
        if not allow_budget_fallback:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="openai call budget exhausted for current process; increase runtime call budget and retry",
            )
        fallback_reason = "budget_exhausted"
        if allow_dummy_fallback:
            budget_exhausted = True
            extractor = DummyExtractor()
            extractor.set_prompt_bundle(prompt_bundle)
        else:
            force_fail_reason = "budget_exhausted: call budget reached and dummy_fallback disabled"

    if force_fail_reason:
        last_error = force_fail_reason
    else:
        try:
            extracted_json = extractor.extract(extraction_input)
        except Exception as exc:  # noqa: BLE001
            if raise_retryable_errors and isinstance(extractor, OpenAIExtractor) and _is_retryable_extraction_error(exc):
                raise
            last_error = _build_last_error(exc)
            if isinstance(exc, OpenAIFallbackError):
                fallback_reason = exc.fallback_reason
            if isinstance(extractor, OpenAIExtractor) and allow_dummy_fallback:
                extractor = DummyExtractor()
                extractor.set_prompt_bundle(prompt_bundle)
                extracted_json = extractor.extract(extraction_input)

    if isinstance(extracted_json, dict):
        extracted_json = extracted_json.copy()
    else:
        extracted_json = default_extracted_json(extraction_input)
    extracted_json = normalize_extracted_json(
        extracted_json,
        posted_at=extraction_input.posted_at,
        include_meta=True,
        alias_to_symbol=alias_to_symbol,
        known_symbols=known_symbols,
    )

    extracted_json["meta"] = _build_extraction_meta(
        extracted_json=extracted_json,
        extractor=extractor,
        fallback_reason=fallback_reason,
        last_error=last_error,
    )

    if truncation_meta:
        extracted_json["meta"] = {
            **(extracted_json.get("meta") if isinstance(extracted_json.get("meta"), dict) else {}),
            **truncation_meta,
        }
    if budget_exhausted:
        if not last_error:
            last_error = "budget_exhausted: call budget reached, auto-fallback to dummy extractor"
        extracted_json["meta"] = {
            **(extracted_json.get("meta") if isinstance(extracted_json.get("meta"), dict) else {}),
            "fallback_reason": "budget_exhausted",
            "dummy_fallback": True,
        }
    if force_reextract:
        extracted_json["meta"] = {
            **(extracted_json.get("meta") if isinstance(extracted_json.get("meta"), dict) else {}),
            "force_reextract": True,
            "force_reextract_triggered_by": (force_reextract_triggered_by or EXTRACTION_REVIEWER),
            "source_extraction_id": source_extraction_id,
        }

    audit = extractor.get_audit()
    if audit.raw_model_output is None:
        audit.raw_model_output = json.dumps(extracted_json, ensure_ascii=False)
    if audit.parsed_model_output is None and isinstance(extracted_json, dict):
        audit.parsed_model_output = extracted_json

    await ensure_assets_from_extracted_json(
        db,
        raw_post=raw_post,
        extracted_json=extracted_json,
    )
    resolved_extractor_name = _resolve_extractor_name(
        extractor=extractor,
        settings_base_url=settings.openai_base_url,
        extracted_json=extracted_json,
    )

    extraction = PostExtraction(
        raw_post_id=raw_post.id,
        status=ExtractionStatus.pending,
        extracted_json=extracted_json,
        model_name=extractor.model_name,
        extractor_name=resolved_extractor_name,
        prompt_version=audit.prompt_version,
        prompt_text=audit.prompt_text,
        prompt_hash=audit.prompt_hash,
        raw_model_output=audit.raw_model_output,
        parsed_model_output=audit.parsed_model_output,
        model_latency_ms=audit.model_latency_ms,
        model_input_tokens=audit.model_input_tokens,
        model_output_tokens=audit.model_output_tokens,
        last_error=last_error,
    )
    db.add(extraction)
    await db.flush()
    auto_review_outcome = await postprocess_auto_review(db=db, extraction=extraction, raw_post=raw_post)
    setattr(extraction, "auto_rejected", auto_review_outcome == "rejected")
    setattr(extraction, "auto_approved", auto_review_outcome == "approved")
    try:
        if auto_review_outcome is None:
            await _auto_apply_extraction_views(db, extraction=extraction, raw_post=raw_post)
    except Exception as exc:  # noqa: BLE001
        auto_error = _build_last_error(exc)
        extraction.last_error = auto_error if not extraction.last_error else f"{extraction.last_error}; auto_apply={auto_error}"
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


async def _list_x_raw_posts(
    db: AsyncSession,
    *,
    author_handle: str | None = None,
) -> list[RawPost]:
    result = await db.execute(select(RawPost).where(RawPost.platform == "x"))
    posts = list(result.scalars().all())
    if author_handle is None:
        return posts
    handle_key = _normalize_handle(author_handle)
    return [item for item in posts if _normalize_handle(item.author_handle) == handle_key]


async def _latest_extractions_by_raw_post_id(
    db: AsyncSession,
    *,
    raw_post_ids: set[int],
) -> dict[int, PostExtraction]:
    if not raw_post_ids:
        return {}
    result = await db.execute(select(PostExtraction))
    rows = [item for item in result.scalars().all() if item.raw_post_id in raw_post_ids]
    rows.sort(
        key=lambda item: (
            item.created_at or datetime.min.replace(tzinfo=UTC),
            item.id or 0,
        ),
        reverse=True,
    )
    latest: dict[int, PostExtraction] = {}
    for row in rows:
        if row.raw_post_id in latest:
            continue
        latest[row.raw_post_id] = row
    return latest


async def _raw_post_ids_with_successful_extractions(
    db: AsyncSession,
    *,
    raw_post_ids: set[int],
) -> set[int]:
    if not raw_post_ids:
        return set()
    result = await db.execute(select(PostExtraction))
    successful_ids: set[int] = set()
    for extraction in result.scalars().all():
        if extraction.raw_post_id not in raw_post_ids:
            continue
        raw_post = await db.get(RawPost, extraction.raw_post_id)
        if _is_result_available_extraction(extraction, raw_post):
            successful_ids.add(extraction.raw_post_id)
    return successful_ids


async def _has_successful_extraction(
    db: AsyncSession,
    *,
    raw_post_id: int,
    raw_post: RawPost | None = None,
) -> bool:
    result = await db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post_id))
    target_raw_post = raw_post if raw_post is not None else await db.get(RawPost, raw_post_id)
    for extraction in result.scalars().all():
        if _is_result_available_extraction(extraction, target_raw_post):
            return True
    return False


async def _latest_extraction_for_raw_post(
    db: AsyncSession,
    *,
    raw_post_id: int,
) -> PostExtraction | None:
    result = await db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post_id))
    rows = list(result.scalars().all())
    if not rows:
        return None
    rows.sort(
        key=lambda item: (
            item.created_at or datetime.min.replace(tzinfo=UTC),
            item.id or 0,
        ),
        reverse=True,
    )
    return rows[0]


async def _failed_batch_retry_count(
    db: AsyncSession,
    *,
    raw_post_id: int,
) -> int:
    result = await db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post_id))
    total = 0
    for extraction in result.scalars().all():
        if not (extraction.last_error or "").strip():
            continue
        meta = extraction.extracted_json.get("meta") if isinstance(extraction.extracted_json, dict) else None
        if isinstance(meta, dict) and meta.get("retry_source") == "extract_batch":
            total += 1
    return total


async def _resolve_import_author_handle(
    *,
    item: XImportItemCreate,
    db: AsyncSession,
    x_kols_by_id: dict[int, Kol],
    x_kols_by_handle_key: dict[str, Kol],
    enabled_x_kols_by_handle_key: dict[str, Kol],
    only_followed: bool,
    allow_unknown_handles: bool,
) -> tuple[str | None, int | None, str | None, Kol | None]:
    if item.kol_id is not None:
        kol = x_kols_by_id.get(item.kol_id)
        if kol is None:
            return (
                None,
                None,
                f"kol_id={item.kol_id} not found on platform=x, external_id={item.external_id}",
                None,
            )
        if only_followed and not kol.enabled:
            return (
                None,
                None,
                f"kol_id={item.kol_id} is disabled; skipped by only_followed=true, external_id={item.external_id}",
                None,
            )
        return kol.handle, kol.id, None, None

    handle_key = _normalize_author_handle(item.resolved_author_handle or item.author_handle)
    if not handle_key:
        return None, None, f"missing author_handle, external_id={item.external_id}", None
    kol = enabled_x_kols_by_handle_key.get(handle_key) if only_followed else x_kols_by_handle_key.get(handle_key)
    if kol is not None:
        return kol.handle, kol.id, None, None

    if only_followed or not allow_unknown_handles:
        return (
            None,
            None,
            f"not_followed handle={handle_key}, external_id={item.external_id}",
            None,
        )

    created = await _upsert_kol_by_author(db, platform="x", handle=handle_key)
    x_kols_by_id[created.id] = created
    x_kols_by_handle_key[_normalize_author_handle(created.handle)] = created
    if created.enabled:
        enabled_x_kols_by_handle_key[_normalize_author_handle(created.handle)] = created
    return created.handle, created.id, None, created


def _detect_single_handle_from_items(items: list[XImportItemCreate]) -> tuple[str | None, list[str]]:
    handles = sorted(
        {
            _normalize_author_handle(item.author_handle)
            for item in items
            if isinstance(item.author_handle, str) and _normalize_author_handle(item.author_handle)
        }
    )
    if not handles:
        return None, []
    if len(handles) > 1:
        return None, handles
    return handles[0], handles


@app.get("/assets", response_model=list[AssetRead])
async def list_assets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Asset).order_by(Asset.id.asc()))
    return list(result.scalars().all())


@app.get("/assets/aliases", response_model=list[AssetAliasMapRead])
async def list_alias_symbol_map(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AssetAlias.alias, Asset.symbol)
        .join(Asset, Asset.id == AssetAlias.asset_id)
        .order_by(AssetAlias.id.asc())
    )
    rows: list[AssetAliasMapRead] = []
    for alias, symbol in result.all():
        rows.append(AssetAliasMapRead(alias=alias, symbol=symbol))
    return rows


@app.post("/assets", response_model=AssetRead, status_code=status.HTTP_201_CREATED)
async def create_asset(payload: AssetCreate, db: AsyncSession = Depends(get_db)):
    symbol = payload.symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="symbol is required")

    asset = Asset(symbol=symbol, name=payload.name, market=payload.market)
    db.add(asset)

    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="symbol already exists")

    await db.refresh(asset)
    return asset


@app.post("/assets/upsert", response_model=AssetRead)
async def upsert_asset_endpoint(payload: AssetCreate, db: AsyncSession = Depends(get_db)):
    asset = await upsert_asset(
        db,
        symbol=payload.symbol,
        name=payload.name,
        market=payload.market,
    )
    await db.commit()
    await db.refresh(asset)
    return asset


@app.get("/assets/{asset_id}/aliases", response_model=list[AssetAliasRead])
async def list_asset_aliases(asset_id: int, db: AsyncSession = Depends(get_db)):
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")
    result = await db.execute(
        select(AssetAlias).where(AssetAlias.asset_id == asset_id).order_by(AssetAlias.id.asc())
    )
    return list(result.scalars().all())


@app.post("/assets/{asset_id}/aliases", response_model=list[AssetAliasRead], status_code=status.HTTP_201_CREATED)
async def create_asset_alias(asset_id: int, payload: AssetAliasCreate, db: AsyncSession = Depends(get_db)):
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")
    alias = payload.alias.strip()
    if not alias:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="alias is required")
    alias_key = _normalize_alias_key(alias)

    existing_result = await db.execute(select(AssetAlias).order_by(AssetAlias.id.asc()))
    existing_aliases = list(existing_result.scalars().all())
    matched = next((item for item in existing_aliases if _normalize_alias_key(item.alias) == alias_key), None)
    if matched is None:
        db.add(AssetAlias(asset_id=asset_id, alias=alias))
    else:
        matched.asset_id = asset_id
        matched.alias = alias
        await db.flush()

    await db.commit()
    result = await db.execute(
        select(AssetAlias).where(AssetAlias.asset_id == asset_id).order_by(AssetAlias.id.asc())
    )
    return list(result.scalars().all())


@app.get("/kols", response_model=list[KolRead])
async def list_kols(enabled: bool | None = Query(default=None), db: AsyncSession = Depends(get_db)):
    query = select(Kol)
    if enabled is not None:
        query = query.where(Kol.enabled == enabled)
    result = await db.execute(query.order_by(Kol.id.asc()))
    return list(result.scalars().all())


@app.post("/kols", response_model=KolRead, status_code=status.HTTP_201_CREATED)
async def create_kol(payload: KolCreate, db: AsyncSession = Depends(get_db)):
    platform = payload.platform.strip().lower()
    handle = payload.handle.strip().lower()

    if not platform or not handle:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="platform and handle are required",
        )

    kol = Kol(platform=platform, handle=handle, display_name=payload.display_name)
    db.add(kol)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="kol already exists")

    await db.refresh(kol)
    return kol


@app.get("/profiles", response_model=list[ProfileSummaryRead])
async def list_profiles(db: AsyncSession = Depends(get_db)):
    return await list_profiles_service(db)


@app.get("/profiles/{profile_id}", response_model=ProfileRead)
async def get_profile(profile_id: int = Path(ge=1), db: AsyncSession = Depends(get_db)):
    return await get_profile_service(db, profile_id=profile_id)


@app.put("/profiles/{profile_id}/kols", response_model=ProfileRead)
async def update_profile_kols(
    payload: ProfileKolsUpdateRequest,
    profile_id: int = Path(ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await update_profile_kols_service(db, profile_id=profile_id, payload=payload)


@app.put("/profiles/{profile_id}/markets", response_model=ProfileRead)
async def update_profile_markets(
    payload: ProfileMarketsUpdateRequest,
    profile_id: int = Path(ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await update_profile_markets_service(db, profile_id=profile_id, payload=payload)


@app.post("/kol-views", response_model=KolViewRead, status_code=status.HTTP_201_CREATED)
async def create_kol_view(payload: KolViewCreate, db: AsyncSession = Depends(get_db)):
    kol = await db.get(Kol, payload.kol_id)
    if kol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kol not found")

    asset = await db.get(Asset, payload.asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")

    view = KolView(
        kol_id=payload.kol_id,
        asset_id=payload.asset_id,
        stance=payload.stance,
        horizon=payload.horizon,
        confidence=payload.confidence,
        summary=(payload.summary or "").strip(),
        source_url=(payload.source_url or "").strip(),
        as_of=payload.as_of or datetime.now(UTC).date(),
    )
    db.add(view)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate kol view")

    await db.refresh(view)
    return view


@app.get("/assets/{asset_id}/views", response_model=AssetViewsRead)
async def get_asset_views(asset_id: int, db: AsyncSession = Depends(get_db)):
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")

    result = await db.execute(
        select(KolView).where(KolView.asset_id == asset_id)
    )
    views = list(result.scalars().all())
    return build_asset_views_response(asset_id=asset_id, views=views)


@app.get("/assets/{asset_id}/views/feed", response_model=AssetViewsFeedRead)
async def get_asset_views_feed(
    asset_id: int,
    horizon: Horizon | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")

    query = select(KolView).where(KolView.asset_id == asset_id).order_by(KolView.created_at.desc(), KolView.id.desc())
    if horizon is not None:
        query = query.where(KolView.horizon == horizon)
    result = await db.execute(query)
    all_views = list(result.scalars().all())
    total = len(all_views)
    page = all_views[offset : offset + limit]

    kol_result = await db.execute(select(Kol))
    kol_map = {item.id: item for item in kol_result.scalars().all()}
    items: list[AssetViewFeedItemRead] = []
    for view in page:
        kol = kol_map.get(view.kol_id)
        items.append(
            AssetViewFeedItemRead(
                id=view.id,
                kol_id=view.kol_id,
                kol_display_name=kol.display_name if kol is not None else None,
                kol_handle=kol.handle if kol is not None else None,
                stance=view.stance,
                horizon=view.horizon,
                confidence=view.confidence,
                summary=view.summary,
                source_url=view.source_url,
                as_of=view.as_of,
                created_at=view.created_at,
            )
        )

    return AssetViewsFeedRead(
        asset_id=asset_id,
        horizon=horizon,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + len(items)) < total,
        items=items,
    )


@app.post("/raw-posts", response_model=RawPostRead, status_code=status.HTTP_201_CREATED)
async def create_raw_post(payload: RawPostCreate, db: AsyncSession = Depends(get_db)):
    raw_post = RawPost(
        platform=payload.platform.strip().lower(),
        author_handle=payload.author_handle.strip(),
        external_id=payload.external_id.strip(),
        url=payload.url.strip(),
        content_text=payload.content_text.strip(),
        posted_at=payload.posted_at,
        raw_json=payload.raw_json,
    )
    db.add(raw_post)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="raw post already exists")
    await db.refresh(raw_post)
    return raw_post


@app.post("/ingest/manual", response_model=ManualIngestRead, status_code=status.HTTP_201_CREATED)
async def ingest_manual(payload: ManualIngestCreate, db: AsyncSession = Depends(get_db)):
    platform = payload.platform.strip().lower()
    author_handle = payload.author_handle.strip()
    url = payload.url.strip()
    content_text = payload.content_text.strip()
    external_id = (payload.external_id or "").strip() or build_external_id(url)
    posted_at = payload.posted_at or datetime.now(UTC)

    raw_post = RawPost(
        platform=platform,
        author_handle=author_handle,
        external_id=external_id,
        url=url,
        content_text=content_text,
        posted_at=posted_at,
        raw_json=payload.raw_json,
    )
    db.add(raw_post)

    try:
        await db.flush()
        extraction = await create_pending_extraction(db, raw_post)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="raw post already exists")

    await db.refresh(raw_post)
    await db.refresh(extraction)
    return ManualIngestRead(raw_post=raw_post, extraction=extraction, extraction_id=extraction.id)


@app.get("/ingest/x/import/template", response_model=XImportTemplateRead)
def get_x_import_template():
    return XImportTemplateRead(
        required_fields=[
            "external_id",
            "author_handle",
            "url",
            "posted_at",
            "content_text",
        ],
        optional_fields=["kol_id", "raw_json"],
        notes=[
            "posted_at must be ISO-8601 datetime, for example: 2026-02-23T12:30:00Z",
            "external_id should be a stable tweet id or any stable unique id per post",
            "The endpoint always stores platform as 'x'",
            "When kol_id is provided, raw_post.author_handle is normalized to the matched kol.handle",
        ],
        example=[
            XImportItemCreate(
                kol_id=1,
                external_id="1893772190012345678",
                author_handle="some_kol",
                url="https://x.com/some_kol/status/1893772190012345678",
                posted_at=datetime(2026, 2, 20, 12, 30, tzinfo=UTC),
                content_text="BTC structure still constructive above 60k.",
                raw_json={"lang": "en"},
            ),
            XImportItemCreate(
                external_id="1893772190012345679",
                author_handle="some_kol",
                url="https://x.com/some_kol/status/1893772190012345679",
                posted_at=datetime(2026, 2, 20, 14, 0, tzinfo=UTC),
                content_text="Watching NVDA earnings reaction for AI supply chain read-through.",
                raw_json={"lang": "en"},
            ),
        ],
    )


@app.post("/ingest/x/following/import", response_model=XFollowingImportStatsRead)
async def import_x_following_to_kols(
    payload: Any = Body(...),
    filename: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    if isinstance(payload, (dict, list)):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    elif isinstance(payload, str):
        content = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        content = bytes(payload)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="unsupported request body")
    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="empty file")

    try:
        rows = load_records_from_bytes(content, filename=filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"invalid file content: {exc}")

    following_true_rows = 0
    skipped_count = 0
    errors: list[XFollowingImportErrorRead] = []
    candidate_name_by_handle: dict[str, str | None] = {}
    for idx, row_any in enumerate(rows, start=1):
        if not isinstance(row_any, dict):
            skipped_count += 1
            errors.append(
                XFollowingImportErrorRead(
                    row_index=idx,
                    reason=f"row is not an object (type={type(row_any).__name__})",
                    raw_snippet=_raw_snippet(row_any),
                )
            )
            continue
        if not _coerce_row_bool(row_any.get("following")):
            skipped_count += 1
            continue
        following_true_rows += 1
        handle_value = row_any.get("screen_name")
        handle = _normalize_author_handle(str(handle_value or ""))
        if not handle:
            skipped_count += 1
            errors.append(
                XFollowingImportErrorRead(
                    row_index=idx,
                    reason="missing screen_name for following=true row",
                    raw_snippet=_raw_snippet(row_any),
                )
            )
            continue
        if handle in candidate_name_by_handle:
            skipped_count += 1
            continue
        name_value = row_any.get("name")
        name = str(name_value).strip() if isinstance(name_value, str) else None
        candidate_name_by_handle[handle] = name or None

    kols_result = await db.execute(select(Kol).where(Kol.platform == "x"))
    x_kols_by_handle = {_normalize_author_handle(item.handle): item for item in kols_result.scalars().all()}
    created_kols: list[Kol] = []
    updated_kols: list[Kol] = []
    for handle, display_name in sorted(candidate_name_by_handle.items()):
        existing = x_kols_by_handle.get(handle)
        if existing is None:
            kol = Kol(
                platform="x",
                handle=handle,
                display_name=display_name,
                enabled=True,
            )
            db.add(kol)
            await db.flush()
            created_kols.append(kol)
            x_kols_by_handle[handle] = kol
            continue
        if not existing.enabled:
            existing.enabled = True
        if display_name:
            existing.display_name = display_name
        await db.flush()
        updated_kols.append(existing)

    await db.commit()
    return XFollowingImportStatsRead(
        received_rows=len(rows),
        following_true_rows=following_true_rows,
        created_kols_count=len(created_kols),
        updated_kols_count=len(updated_kols),
        skipped_count=skipped_count,
        created_kols=[XFollowingImportKolRead(id=item.id, handle=item.handle) for item in created_kols],
        updated_kols=[XFollowingImportKolRead(id=item.id, handle=item.handle) for item in updated_kols],
        errors=errors,
    )


@app.post("/ingest/x/convert", response_model=XConvertResponseRead)
async def convert_x_import_file(
    payload: Any = Body(...),
    filename: str | None = Query(default=None),
    author_handle: str | None = Query(default=None),
    kol_id: int | None = Query(default=None, ge=1),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    include_raw_json: bool = Query(default=True),
    only_followed: bool = Query(default=True),
    allow_unknown_handles: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="start_date must be <= end_date")

    if isinstance(payload, (dict, list)):
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    elif isinstance(payload, str):
        content = payload.encode("utf-8")
    elif isinstance(payload, (bytes, bytearray)):
        content = bytes(payload)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="unsupported request body")

    if not content:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="empty file")

    try:
        rows = load_records_from_bytes(content, filename=filename)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"invalid file content: {exc}")

    export_kind = _detect_x_export_kind(rows)
    rows_for_convert = rows
    skipped_not_followed_count = 0
    skipped_not_followed_samples: list[XSkippedNotFollowedRead] = []
    if export_kind == "timeline" and (only_followed or not allow_unknown_handles):
        enabled_handles: set[str] = set()
        try:
            _, enabled_by_handle = await _enabled_x_kol_maps(db)
            enabled_handles = set(enabled_by_handle.keys())
        except Exception:  # noqa: BLE001
            enabled_handles = set()
        if enabled_handles:
            kept_rows: list[Any] = []
            for idx, row_any in enumerate(rows, start=1):
                if not isinstance(row_any, dict):
                    kept_rows.append(row_any)
                    continue
                handle_key = _normalize_author_handle(str(row_any.get("screen_name") or row_any.get("author_handle") or ""))
                if not handle_key:
                    kept_rows.append(row_any)
                    continue
                if handle_key in enabled_handles:
                    kept_rows.append(row_any)
                    continue
                skipped_not_followed_count += 1
                if len(skipped_not_followed_samples) < 20:
                    external_id_value = row_any.get("id")
                    external_id = str(external_id_value).strip() if external_id_value is not None else None
                    skipped_not_followed_samples.append(
                        XSkippedNotFollowedRead(
                            row_index=idx,
                            author_handle=handle_key,
                            external_id=external_id or None,
                            reason="not_followed",
                        )
                    )
            rows_for_convert = kept_rows

    converted, stats = convert_records(
        rows_for_convert,
        author_handle=author_handle,
        kol_id=kol_id,
        start_date=start_date,
        end_date=end_date,
        include_raw_json=include_raw_json,
    )
    converted_models = [XImportItemCreate.model_validate(item) for item in converted]
    for item in converted_models:
        normalized = _normalize_author_handle(item.author_handle)
        item.author_handle = normalized
        item.resolved_author_handle = normalized

    x_kols_by_handle_key: dict[str, Kol] = {}
    try:
        kols_result = await db.execute(select(Kol).where(Kol.platform == "x"))
        x_kols_by_handle_key = {
            _normalize_author_handle(item.handle): item
            for item in kols_result.scalars().all()
        }
    except Exception:  # noqa: BLE001
        x_kols_by_handle_key = {}

    by_handle: dict[str, dict[str, Any]] = {}
    for item in converted_models:
        handle = item.resolved_author_handle or _normalize_author_handle(item.author_handle)
        if not handle:
            continue
        bucket = by_handle.get(handle)
        if bucket is None:
            bucket = {
                "count": 0,
                "earliest_posted_at": item.posted_at,
                "latest_posted_at": item.posted_at,
            }
            by_handle[handle] = bucket
        bucket["count"] += 1
        if bucket["earliest_posted_at"] is None or item.posted_at < bucket["earliest_posted_at"]:
            bucket["earliest_posted_at"] = item.posted_at
        if bucket["latest_posted_at"] is None or item.posted_at > bucket["latest_posted_at"]:
            bucket["latest_posted_at"] = item.posted_at

    handles_summary: list[XHandleSummaryRead] = []
    for handle in sorted(by_handle.keys()):
        summary = by_handle[handle]
        will_create_kol = (
            (handle not in x_kols_by_handle_key) and kol_id is None and (not only_followed) and allow_unknown_handles
        )
        handles_summary.append(
            XHandleSummaryRead(
                author_handle=handle,
                count=int(summary["count"]),
                earliest_posted_at=summary["earliest_posted_at"],
                latest_posted_at=summary["latest_posted_at"],
                will_create_kol=will_create_kol,
            )
        )

    resolved_author_handle: str | None = None
    resolved_kol_id: int | None = None
    if kol_id is None and not (author_handle or "").strip():
        resolved_author_handle, handles = _detect_single_handle_from_items(converted_models)
        if resolved_author_handle and len(handles) == 1:
            matched = x_kols_by_handle_key.get(resolved_author_handle)
            if matched is not None:
                resolved_kol_id = matched.id

    return XConvertResponseRead(
        converted_rows=len(rows),
        converted_ok=stats.output_count,
        converted_failed=stats.failed_count,
        errors=[
            XConvertErrorRead(
                row_index=item.row_index,
                external_id=item.external_id,
                url=item.url,
                reason=item.reason,
            )
            for item in (stats.errors or [])
        ],
        items=converted_models,
        handles_summary=handles_summary,
        resolved_author_handle=resolved_author_handle,
        resolved_kol_id=resolved_kol_id,
        kol_created=False,
        skipped_not_followed_count=skipped_not_followed_count,
        skipped_not_followed_samples=skipped_not_followed_samples,
    )


@app.post("/ingest/x/import", response_model=XImportStatsRead)
async def import_x_posts(
    payload: list[XImportItemCreate],
    trigger_extraction: bool = Query(default=False),
    author_handle_override: str | None = Query(default=None),
    only_followed: bool = Query(default=False),
    allow_unknown_handles: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
):
    inserted_raw_posts: list[RawPost] = []
    inserted_raw_post_ids: list[int] = []
    dedup_existing_raw_post_ids: list[int] = []
    dedup_skipped_count = 0
    warnings: list[str] = []
    imported_by_handle: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "received": 0,
            "inserted": 0,
            "dedup": 0,
            "warnings": 0,
            "raw_post_ids": [],
            "extract_success": 0,
            "extract_failed": 0,
            "skipped_already_extracted": 0,
        }
    )
    created_kols_by_handle: dict[str, Kol] = {}
    raw_post_handle_by_id: dict[int, str] = {}
    skipped_not_followed_count = 0
    skipped_not_followed_samples: list[XSkippedNotFollowedRead] = []

    kols_result = await db.execute(select(Kol).where(Kol.platform == "x"))
    x_kols = list(kols_result.scalars().all())
    x_kols_by_id = {item.id: item for item in x_kols}
    x_kols_by_handle_key = {_normalize_author_handle(item.handle): item for item in x_kols}
    enabled_x_kols_by_handle_key = {_normalize_author_handle(item.handle): item for item in x_kols if item.enabled}

    normalized_override = _normalize_author_handle(author_handle_override or "")
    resolved_author_handle: str | None = None
    resolved_kol_id: int | None = None
    kol_created = False

    for row_index, item in enumerate(payload, start=1):
        original_handle = _normalize_author_handle(item.resolved_author_handle or item.author_handle)
        if normalized_override and item.kol_id is None:
            item.resolved_author_handle = normalized_override
            item.author_handle = normalized_override
            original_handle = normalized_override
        if original_handle:
            imported_by_handle[original_handle]["received"] += 1

        author_handle, row_kol_id, warning, created_kol = await _resolve_import_author_handle(
            item=item,
            db=db,
            x_kols_by_id=x_kols_by_id,
            x_kols_by_handle_key=x_kols_by_handle_key,
            enabled_x_kols_by_handle_key=enabled_x_kols_by_handle_key,
            only_followed=only_followed,
            allow_unknown_handles=allow_unknown_handles,
        )
        if created_kol is not None:
            kol_created = True
            created_kols_by_handle[_normalize_author_handle(created_kol.handle)] = created_kol
        if warning is not None:
            warnings.append(warning)
            if original_handle:
                imported_by_handle[original_handle]["warnings"] += 1
            if "not_followed" in warning:
                skipped_not_followed_count += 1
                if len(skipped_not_followed_samples) < 20:
                    skipped_not_followed_samples.append(
                        XSkippedNotFollowedRead(
                            row_index=row_index,
                            author_handle=original_handle or None,
                            external_id=item.external_id,
                            reason="not_followed",
                        )
                    )
        if author_handle is None:
            continue

        external_id = item.external_id.strip()
        existing_result = await db.execute(
            select(RawPost).where(
                RawPost.platform == "x",
                RawPost.external_id == external_id,
            )
        )
        existing = existing_result.scalar_one_or_none()
        if existing is not None:
            dedup_skipped_count += 1
            dedup_existing_raw_post_ids.append(existing.id)
            handle_key = _normalize_author_handle(existing.author_handle or author_handle)
            imported_by_handle[handle_key]["dedup"] += 1
            if existing.id not in imported_by_handle[handle_key]["raw_post_ids"]:
                imported_by_handle[handle_key]["raw_post_ids"].append(existing.id)
            raw_post_handle_by_id[existing.id] = handle_key
            continue

        raw_post = RawPost(
            platform="x",
            kol_id=row_kol_id,
            author_handle=author_handle,
            external_id=external_id,
            url=item.url.strip(),
            content_text=item.content_text.strip(),
            posted_at=item.posted_at,
            raw_json=item.raw_json,
        )
        db.add(raw_post)
        inserted_raw_posts.append(raw_post)
        handle_key = _normalize_author_handle(author_handle)
        imported_by_handle[handle_key]["inserted"] += 1

    try:
        await db.flush()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate raw post in payload")

    for item in inserted_raw_posts:
        if item.id is None:
            continue
        handle_key = _normalize_author_handle(item.author_handle)
        raw_post_handle_by_id[item.id] = handle_key
        imported_by_handle[handle_key]["raw_post_ids"].append(item.id)

    extract_success_count = 0
    extract_failed_count = 0
    skipped_already_extracted_count = 0
    if trigger_extraction:
        extraction_target_ids = list(
            dict.fromkeys(
                [
                    *[item.id for item in inserted_raw_posts if item.id is not None],
                    *dedup_existing_raw_post_ids,
                ]
            )
        )
        latest_map = await _latest_extractions_by_raw_post_id(db, raw_post_ids=set(extraction_target_ids))
        for raw_post_id in extraction_target_ids:
            handle_key = raw_post_handle_by_id.get(raw_post_id)
            raw_post = await db.get(RawPost, raw_post_id)
            if raw_post is None:
                continue
            if not await _raw_post_matches_enabled_x_kol(db, raw_post):
                skipped_not_followed_count += 1
                if len(skipped_not_followed_samples) < 20:
                    skipped_not_followed_samples.append(
                        XSkippedNotFollowedRead(
                            row_index=0,
                            author_handle=_normalize_author_handle(raw_post.author_handle),
                            external_id=raw_post.external_id,
                            reason="not_followed",
                        )
                    )
                continue
            latest_extraction = latest_map.get(raw_post_id)
            if _terminal_review_skip_kind(raw_post=raw_post, latest_extraction=latest_extraction) is not None:
                skipped_already_extracted_count += 1
                if handle_key:
                    imported_by_handle[handle_key]["skipped_already_extracted"] += 1
                continue
            if _is_active_extraction(latest_extraction, raw_post):
                skipped_already_extracted_count += 1
                if handle_key:
                    imported_by_handle[handle_key]["skipped_already_extracted"] += 1
                continue
            if await _has_successful_extraction(db, raw_post_id=raw_post_id, raw_post=raw_post):
                skipped_already_extracted_count += 1
                if handle_key:
                    imported_by_handle[handle_key]["skipped_already_extracted"] += 1
                continue
            try:
                _check_reextract_rate_limit(raw_post.id)
                extraction = await create_pending_extraction(db, raw_post)
                if _is_failed_extraction(extraction, raw_post):
                    extract_failed_count += 1
                    if handle_key:
                        imported_by_handle[handle_key]["extract_failed"] += 1
                else:
                    extract_success_count += 1
                    if handle_key:
                        imported_by_handle[handle_key]["extract_success"] += 1
            except Exception:  # noqa: BLE001
                extract_failed_count += 1
                if handle_key:
                    imported_by_handle[handle_key]["extract_failed"] += 1

    inserted_raw_post_ids = [item.id for item in inserted_raw_posts]
    payload_handles = {
        _normalize_author_handle(item.resolved_author_handle or item.author_handle)
        for item in payload
        if _normalize_author_handle(item.resolved_author_handle or item.author_handle)
    }
    if len(payload_handles) == 1:
        resolved_author_handle = next(iter(payload_handles))
        resolved = x_kols_by_handle_key.get(resolved_author_handle)
        if resolved is not None:
            resolved_kol_id = resolved.id

    await db.commit()
    return XImportStatsRead(
        received_count=len(payload),
        inserted_raw_posts_count=len(inserted_raw_posts),
        inserted_raw_post_ids=inserted_raw_post_ids,
        dedup_existing_raw_post_ids=list(dict.fromkeys(dedup_existing_raw_post_ids)),
        dedup_skipped_count=dedup_skipped_count,
        extract_success_count=extract_success_count,
        extract_failed_count=extract_failed_count,
        skipped_already_extracted_count=skipped_already_extracted_count,
        warnings_count=len(warnings),
        warnings=warnings[:50],
        imported_by_handle={
            handle: XImportedByHandleRead.model_validate(stats)
            for handle, stats in sorted(imported_by_handle.items())
        },
        created_kols=[
            XCreatedKolRead(
                id=item.id,
                handle=item.handle,
                name=item.display_name,
            )
            for _, item in sorted(created_kols_by_handle.items(), key=lambda pair: pair[0])
        ],
        resolved_author_handle=resolved_author_handle,
        resolved_kol_id=resolved_kol_id,
        kol_created=kol_created,
        skipped_not_followed_count=skipped_not_followed_count,
        skipped_not_followed_samples=skipped_not_followed_samples,
    )


def _empty_extract_counters(*, requested_count: int = 0) -> dict[str, int]:
    return {
        "requested_count": requested_count,
        "success_count": 0,
        "skipped_count": 0,
        "skipped_already_extracted_count": 0,
        "skipped_already_pending_count": 0,
        "skipped_already_success_count": 0,
        "skipped_already_has_result_count": 0,
        "skipped_already_rejected_count": 0,
        "skipped_already_approved_count": 0,
        "skipped_not_followed_count": 0,
        "failed_count": 0,
        "auto_approved_count": 0,
        "auto_rejected_count": 0,
        "resumed_requested_count": 0,
        "resumed_success": 0,
        "resumed_failed": 0,
        "resumed_skipped": 0,
    }


def _merge_extract_counters(total: dict[str, int], chunk: RawPostsExtractBatchRead) -> None:
    total["success_count"] += chunk.success_count
    total["skipped_count"] += chunk.skipped_count
    total["skipped_already_extracted_count"] += chunk.skipped_already_extracted_count
    total["skipped_already_pending_count"] += chunk.skipped_already_pending_count
    total["skipped_already_success_count"] += chunk.skipped_already_success_count
    total["skipped_already_has_result_count"] += chunk.skipped_already_has_result_count
    total["skipped_already_rejected_count"] += chunk.skipped_already_rejected_count
    total["skipped_already_approved_count"] += chunk.skipped_already_approved_count
    total["skipped_not_followed_count"] += chunk.skipped_not_followed_count
    total["failed_count"] += chunk.failed_count
    total["auto_approved_count"] += chunk.auto_approved_count
    total["auto_rejected_count"] += chunk.auto_rejected_count
    total["resumed_requested_count"] += chunk.resumed_requested_count
    total["resumed_success"] += chunk.resumed_success
    total["resumed_failed"] += chunk.resumed_failed
    total["resumed_skipped"] += chunk.resumed_skipped


async def _extract_raw_posts_batch_core(payload: RawPostsExtractBatchRequest, db: AsyncSession) -> RawPostsExtractBatchRead:
    requested_ids = list(dict.fromkeys(payload.raw_post_ids))
    success_count = 0
    skipped_count = 0
    skipped_already_extracted_count = 0
    skipped_already_pending_count = 0
    skipped_already_success_count = 0
    skipped_already_has_result_count = 0
    skipped_already_rejected_count = 0
    skipped_already_approved_count = 0
    skipped_not_followed_count = 0
    failed_count = 0
    auto_approved_count = 0
    auto_rejected_count = 0
    resumed_requested_count = 0
    resumed_success = 0
    resumed_failed = 0
    resumed_skipped = 0

    settings = get_settings()
    throttle = _get_runtime_throttle(settings)
    retry_max = max(0, settings.extract_retry_max)
    max_resume_retries = max(1, retry_max)
    backoff_base_ms = max(1, settings.extract_retry_backoff_base_ms)
    backoff_cap_ms = max(backoff_base_ms, settings.extract_retry_backoff_max_ms)
    limiter = _RpmLimiter(throttle["max_rpm"])
    semaphore = asyncio.Semaphore(max(1, throttle["max_concurrency"]))
    failed_errors: dict[int, tuple[str, str, int]] = {}
    chunk_size = max(1, throttle["batch_size"])
    batch_sleep_seconds = max(0.0, throttle["batch_sleep_ms"] / 1000.0)

    async def _process_one(raw_post_id: int, local_db: AsyncSession) -> tuple[str, str | None, bool, str | None]:
        async with semaphore:
            raw_post = await local_db.get(RawPost, raw_post_id)
            if raw_post is None:
                return ("skipped", None, False, None)
            if not await _raw_post_matches_enabled_x_kol(local_db, raw_post):
                return ("skipped_not_followed", None, False, None)

            rows_result = await local_db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post_id))
            extraction_rows = list(rows_result.scalars().all())
            extraction_rows.sort(
                key=lambda item: (
                    item.created_at or datetime.min.replace(tzinfo=UTC),
                    item.id or 0,
                ),
                reverse=True,
            )
            latest_extraction = extraction_rows[0] if extraction_rows else None
            has_result_already = any(_is_result_available_extraction(item, raw_post) for item in extraction_rows)
            resume_candidate = payload.mode == "pending_or_failed" and (
                latest_extraction is None or _is_failed_extraction(latest_extraction, raw_post)
            )
            if payload.mode in {"pending_only", "pending_or_failed"}:
                terminal_skip_kind = _terminal_review_skip_kind(raw_post=raw_post, latest_extraction=latest_extraction)
                if terminal_skip_kind is not None:
                    return (terminal_skip_kind, None, resume_candidate, None)
            if payload.mode in {"pending_only", "pending_or_failed"} and _is_active_extraction(latest_extraction, raw_post):
                return ("skipped_already_pending", None, resume_candidate, None)

            if payload.mode in {"pending_only", "pending_or_failed"} and has_result_already:
                return ("skipped_already_has_result", None, resume_candidate, None)

            if payload.mode == "pending_only" and latest_extraction is not None:
                return ("skipped", None, resume_candidate, None)

            if payload.mode == "pending_or_failed" and _is_failed_extraction(latest_extraction, raw_post):
                failed_retries = await _failed_batch_retry_count(local_db, raw_post_id=raw_post_id)
                if failed_retries >= max_resume_retries:
                    return ("skipped_retry_limited", None, resume_candidate, None)

            _check_reextract_rate_limit(raw_post.id)
            for attempt in range(1, retry_max + 2):
                try:
                    await limiter.acquire()
                    try:
                        extraction = await create_pending_extraction(
                            local_db,
                            raw_post,
                            raise_retryable_errors=True,
                            force_reextract=payload.mode == "force",
                        )
                    except TypeError as exc:
                        # Allow monkeypatched test doubles that only accept (db, raw_post).
                        if "unexpected keyword argument" not in str(exc):
                            raise
                        extraction = await create_pending_extraction(local_db, raw_post)
                    await local_db.commit()
                    if _is_failed_extraction(extraction, raw_post):
                        return ("failed_existing", (extraction.last_error or "extraction_failed"), resume_candidate, None)
                    auto_outcome = "rejected" if bool(getattr(extraction, "auto_rejected", False)) else (
                        "approved" if bool(getattr(extraction, "auto_approved", False)) else None
                    )
                    return ("success", None, resume_candidate, auto_outcome)
                except Exception as exc:  # noqa: BLE001
                    if not _is_retryable_extraction_error(exc) or attempt > retry_max:
                        await local_db.rollback()
                        return ("failed", _build_last_error(exc), resume_candidate, None)
                    await local_db.rollback()
                    await asyncio.sleep(
                        _build_retry_backoff_seconds(
                            attempt=attempt,
                            base_ms=backoff_base_ms,
                            cap_ms=backoff_cap_ms,
                        )
                    )
            return ("failed", "retry_exhausted", resume_candidate, None)

    for offset in range(0, len(requested_ids), chunk_size):
        chunk = requested_ids[offset : offset + chunk_size]
        if isinstance(db, AsyncSession):
            async def _run_with_task_db(raw_post_id: int) -> tuple[str, str | None, bool, str | None]:
                async with AsyncSessionLocal() as task_db:
                    return await _process_one(raw_post_id, task_db)

            results = await asyncio.gather(*[_run_with_task_db(raw_post_id) for raw_post_id in chunk], return_exceptions=False)
        else:
            results = []
            for raw_post_id in chunk:
                result = await _process_one(raw_post_id, db)
                results.append(result)
        for idx, result in enumerate(results):
            kind, last_error, resume_candidate, auto_outcome = result
            raw_post_id = chunk[idx]
            if payload.mode == "pending_or_failed" and resume_candidate:
                resumed_requested_count += 1
            if kind == "success":
                success_count += 1
                if auto_outcome == "rejected":
                    auto_rejected_count += 1
                elif auto_outcome == "approved":
                    auto_approved_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_success += 1
            elif kind == "skipped_already_pending":
                skipped_already_pending_count += 1
                skipped_already_extracted_count += 1
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind in {"skipped_already_success", "skipped_already_has_result"}:
                skipped_already_success_count += 1
                skipped_already_has_result_count += 1
                skipped_already_extracted_count += 1
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind == "skipped_already_rejected":
                skipped_already_rejected_count += 1
                skipped_already_extracted_count += 1
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind == "skipped_already_approved":
                skipped_already_approved_count += 1
                skipped_already_extracted_count += 1
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind == "skipped_retry_limited":
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind == "skipped_not_followed":
                skipped_not_followed_count += 1
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            elif kind == "skipped":
                skipped_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_skipped += 1
            else:
                failed_count += 1
                if payload.mode == "pending_or_failed" and resume_candidate:
                    resumed_failed += 1
                if kind == "failed_existing":
                    continue
                if last_error:
                    error_category = _classify_last_error(last_error)
                    if isinstance(db, AsyncSession):
                        async with AsyncSessionLocal() as task_db:
                            batch_retry_count = (await _failed_batch_retry_count(task_db, raw_post_id=raw_post_id)) + 1
                    else:
                        batch_retry_count = (await _failed_batch_retry_count(db, raw_post_id=raw_post_id)) + 1
                    failed_errors[raw_post_id] = (last_error, error_category, batch_retry_count)
        if offset + chunk_size < len(requested_ids):
            await asyncio.sleep(batch_sleep_seconds)

    for raw_post_id, (last_error, error_category, batch_retry_count) in failed_errors.items():
        if isinstance(db, AsyncSession):
            async with AsyncSessionLocal() as task_db:
                raw_post = await task_db.get(RawPost, raw_post_id)
                if raw_post is None:
                    continue
                await _create_failed_extraction(
                    task_db,
                    raw_post=raw_post,
                    last_error=last_error,
                    error_category=error_category,
                    batch_retry_count=batch_retry_count,
                )
                await task_db.commit()
        else:
            raw_post = await db.get(RawPost, raw_post_id)
            if raw_post is None:
                continue
            await _create_failed_extraction(
                db,
                raw_post=raw_post,
                last_error=last_error,
                error_category=error_category,
                batch_retry_count=batch_retry_count,
            )

    if not isinstance(db, AsyncSession):
        await db.commit()
    resume_mode = payload.mode == "pending_or_failed"
    return RawPostsExtractBatchRead(
        requested_count=len(requested_ids),
        success_count=success_count,
        skipped_count=skipped_count,
        skipped_already_extracted_count=skipped_already_extracted_count,
        skipped_already_pending_count=skipped_already_pending_count,
        skipped_already_success_count=skipped_already_success_count,
        skipped_already_has_result_count=skipped_already_has_result_count,
        skipped_already_rejected_count=skipped_already_rejected_count,
        skipped_already_approved_count=skipped_already_approved_count,
        skipped_not_followed_count=skipped_not_followed_count,
        failed_count=failed_count,
        auto_approved_count=auto_approved_count,
        auto_rejected_count=auto_rejected_count,
        resumed_requested_count=resumed_requested_count if resume_mode else 0,
        resumed_success=resumed_success if resume_mode else 0,
        resumed_failed=resumed_failed if resume_mode else 0,
        resumed_skipped=resumed_skipped if resume_mode else 0,
    )


@app.post("/raw-posts/extract-batch", response_model=RawPostsExtractBatchRead)
async def extract_raw_posts_batch(payload: RawPostsExtractBatchRequest, db: AsyncSession = Depends(get_db)):
    return await _extract_raw_posts_batch_core(payload, db)


def _extract_job_read(state: dict[str, Any]) -> ExtractJobRead:
    return ExtractJobRead(
        job_id=state["job_id"],
        status=state["status"],
        mode=state["mode"],
        batch_size=state["batch_size"],
        batch_sleep_ms=state["batch_sleep_ms"],
        requested_count=state["requested_count"],
        success_count=state["success_count"],
        skipped_count=state["skipped_count"],
        skipped_already_extracted_count=state["skipped_already_extracted_count"],
        skipped_already_pending_count=state["skipped_already_pending_count"],
        skipped_already_success_count=state["skipped_already_success_count"],
        skipped_already_has_result_count=state["skipped_already_has_result_count"],
        skipped_already_rejected_count=state["skipped_already_rejected_count"],
        skipped_already_approved_count=state["skipped_already_approved_count"],
        skipped_not_followed_count=state["skipped_not_followed_count"],
        failed_count=state["failed_count"],
        auto_approved_count=state["auto_approved_count"],
        auto_rejected_count=state["auto_rejected_count"],
        resumed_requested_count=state["resumed_requested_count"],
        resumed_success=state["resumed_success"],
        resumed_failed=state["resumed_failed"],
        resumed_skipped=state["resumed_skipped"],
        last_error_summary=state.get("last_error_summary"),
        created_at=state["created_at"],
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
    )


async def _latest_error_summary_for_job(db: AsyncSession, raw_post_ids: set[int]) -> str | None:
    if not raw_post_ids:
        return None
    latest_map = await _latest_extractions_by_raw_post_id(db, raw_post_ids=raw_post_ids)
    latest_by_time = sorted(
        (
            item for item in latest_map.values() if (item.last_error or "").strip()
        ),
        key=lambda item: item.created_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    if not latest_by_time:
        return None
    return (latest_by_time[0].last_error or "").strip()[:240] or None


async def _run_extract_job(job_id: str) -> None:
    async with EXTRACT_JOBS_LOCK:
        state = EXTRACT_JOBS.get(job_id)
        if state is None:
            EXTRACT_JOB_TASKS.pop(job_id, None)
            return
        state["status"] = "running"
        state["started_at"] = _utc_now()
        raw_post_ids = list(state["raw_post_ids"])
        mode = state["mode"]
        batch_size = state["batch_size"]
        batch_sleep_seconds = max(0.0, state["batch_sleep_ms"] / 1000.0)

    try:
        for offset in range(0, len(raw_post_ids), batch_size):
            chunk = raw_post_ids[offset : offset + batch_size]
            payload = RawPostsExtractBatchRequest(raw_post_ids=chunk, mode=mode)
            async with EXTRACT_JOB_SESSION_FACTORY() as task_db:
                result = await _extract_raw_posts_batch_core(payload, task_db)
                chunk_error_summary = await _latest_error_summary_for_job(task_db, set(chunk))
            async with EXTRACT_JOBS_LOCK:
                state = EXTRACT_JOBS.get(job_id)
                if state is None:
                    EXTRACT_JOB_TASKS.pop(job_id, None)
                    return
                _merge_extract_counters(state, result)
                if chunk_error_summary:
                    state["last_error_summary"] = chunk_error_summary
            if offset + batch_size < len(raw_post_ids):
                await asyncio.sleep(batch_sleep_seconds)
        async with EXTRACT_JOBS_LOCK:
            state = EXTRACT_JOBS.get(job_id)
            if state is not None:
                state["status"] = "done"
                state["finished_at"] = _utc_now()
    except Exception as exc:  # noqa: BLE001
        async with EXTRACT_JOBS_LOCK:
            state = EXTRACT_JOBS.get(job_id)
            if state is not None:
                state["status"] = "failed"
                state["finished_at"] = _utc_now()
                state["last_error_summary"] = _build_last_error(exc)[:240]
    finally:
        EXTRACT_JOB_TASKS.pop(job_id, None)


@app.post("/extract-jobs", response_model=ExtractJobCreateRead, status_code=status.HTTP_201_CREATED)
async def create_extract_job(payload: ExtractJobCreateRequest):
    raw_post_ids = list(dict.fromkeys(payload.raw_post_ids))
    idempotency_key = (payload.idempotency_key or "").strip() or None
    if idempotency_key is not None:
        idempotency_key = idempotency_key[:256]

    async with EXTRACT_JOBS_LOCK:
        if idempotency_key:
            existing_job_id = EXTRACT_JOB_IDEMPOTENCY.get(idempotency_key)
            existing_state = EXTRACT_JOBS.get(existing_job_id or "")
            if existing_state is not None and existing_state["status"] in {"queued", "running"}:
                return ExtractJobCreateRead(job_id=existing_job_id)
        job_id = uuid4().hex
        counters = _empty_extract_counters(requested_count=len(raw_post_ids))
        state: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "mode": payload.mode,
            "batch_size": payload.batch_size,
            "batch_sleep_ms": payload.batch_sleep_ms,
            "raw_post_ids": raw_post_ids,
            "idempotency_key": idempotency_key,
            "last_error_summary": None,
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            **counters,
        }
        EXTRACT_JOBS[job_id] = state
        if idempotency_key:
            EXTRACT_JOB_IDEMPOTENCY[idempotency_key] = job_id
        EXTRACT_JOB_TASKS[job_id] = asyncio.create_task(_run_extract_job(job_id))
    return ExtractJobCreateRead(job_id=job_id)


@app.get("/extract-jobs/{job_id}", response_model=ExtractJobRead)
async def get_extract_job(job_id: str = Path(..., min_length=1)):
    async with EXTRACT_JOBS_LOCK:
        state = EXTRACT_JOBS.get(job_id)
        if state is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extract job not found")
        return _extract_job_read(state)


@app.get("/ingest/x/progress", response_model=XIngestProgressRead)
async def get_x_ingest_progress(
    author_handle: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    posts = await _list_x_raw_posts(db, author_handle=author_handle)
    post_ids = {item.id for item in posts}
    latest_map = await _latest_extractions_by_raw_post_id(db, raw_post_ids=post_ids)

    extracted_success_count = 0
    pending_count = 0
    failed_count = 0
    no_extraction_count = 0
    latest_error_summary: str | None = None
    latest_error_at: datetime | None = None
    latest_extraction_at: datetime | None = None

    for post in posts:
        latest = latest_map.get(post.id)
        if latest is not None and (
            latest_extraction_at is None or (latest.created_at is not None and latest.created_at > latest_extraction_at)
        ):
            latest_extraction_at = latest.created_at

        state = classify_extraction_state(latest, post)
        if state == ClassifiedExtractionState.failed or state == ClassifiedExtractionState.rejected:
            failed_count += 1
            if latest_error_at is None or (latest.created_at is not None and latest.created_at > latest_error_at):
                latest_error_at = latest.created_at
                latest_error_summary = (latest.last_error or "").strip()[:240]
            continue

        if state in {ClassifiedExtractionState.success, ClassifiedExtractionState.approved}:
            extracted_success_count += 1
            continue
        if state == ClassifiedExtractionState.pending:
            pending_count += 1
            continue
        if state == ClassifiedExtractionState.no_extraction:
            no_extraction_count += 1
    return XIngestProgressRead(
        scope="author" if author_handle else "global",
        author_handle=author_handle,
        total_raw_posts=len(posts),
        extracted_success_count=extracted_success_count,
        pending_count=pending_count,
        failed_count=failed_count,
        no_extraction_count=no_extraction_count,
        latest_error_summary=latest_error_summary,
        latest_extraction_at=latest_extraction_at,
    )


@app.post("/ingest/x/retry-failed", response_model=XRetryFailedRead)
async def retry_failed_x_extractions(
    author_handle: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
):
    posts = await _list_x_raw_posts(db, author_handle=author_handle)
    post_map = {item.id: item for item in posts}
    latest_map = await _latest_extractions_by_raw_post_id(db, raw_post_ids=set(post_map.keys()))

    failed_candidates: list[tuple[RawPost, PostExtraction]] = []
    for raw_post_id, extraction in latest_map.items():
        raw_post = post_map.get(raw_post_id)
        if not _is_failed_extraction(extraction, raw_post):
            continue
        if raw_post is None:
            continue
        if _terminal_review_skip_kind(raw_post=raw_post, latest_extraction=extraction) is not None:
            continue
        if await _has_successful_extraction(db, raw_post_id=raw_post.id, raw_post=raw_post):
            continue
        failed_candidates.append((raw_post, extraction))
    failed_candidates.sort(
        key=lambda item: (
            item[1].created_at or datetime.min.replace(tzinfo=UTC),
            item[1].id or 0,
        ),
        reverse=True,
    )
    targets = failed_candidates[:limit]

    reason_counter: Counter[str] = Counter()
    success_count = 0
    failed_count = 0
    skipped_count = 0
    for raw_post, _ in targets:
        if not await _raw_post_matches_enabled_x_kol(db, raw_post):
            skipped_count += 1
            continue
        try:
            _check_reextract_rate_limit(raw_post.id)
            await create_pending_extraction(db, raw_post)
            success_count += 1
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            reason_counter[_format_retry_failure_reason(exc)] += 1

    await db.commit()
    retried_count = success_count + failed_count
    return XRetryFailedRead(
        author_handle=author_handle,
        requested_limit=limit,
        retried_count=retried_count,
        success_count=success_count,
        failed_count=failed_count,
        skipped_count=skipped_count,
        failure_reasons=dict(reason_counter),
    )


@app.post("/admin/extractions/backfill-auto-review", response_model=AdminBackfillAutoReviewRead)
async def backfill_auto_review_by_confidence(
    confirm: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)

    result = await db.execute(select(PostExtraction).order_by(PostExtraction.created_at.desc(), PostExtraction.id.desc()))
    extractions = list(result.scalars().all())
    scanned = len(extractions)
    approved_count = 0
    rejected_count = 0
    skipped_no_result_count = 0
    skipped_no_confidence_count = 0
    skipped_already_terminal_count = 0
    errors: list[AutoReviewBackfillErrorRead] = []

    for extraction in extractions:
        extraction_id = getattr(extraction, "id", None)
        raw_post = await db.get(RawPost, extraction.raw_post_id)
        if raw_post is None:
            if len(errors) < 20:
                errors.append(
                    AutoReviewBackfillErrorRead(
                        extraction_id=extraction_id,
                        raw_post_id=extraction.raw_post_id,
                        error="raw_post_not_found",
                    )
                )
            continue
        state = classify_extraction_state(extraction, raw_post)
        if state in {ClassifiedExtractionState.approved, ClassifiedExtractionState.rejected}:
            skipped_already_terminal_count += 1
            continue
        if state != ClassifiedExtractionState.success:
            skipped_no_result_count += 1
            continue
        if not isinstance(extraction.extracted_json, dict):
            skipped_no_result_count += 1
            continue
        model_confidence = _coerce_extraction_confidence(extraction.extracted_json)
        if model_confidence is None:
            skipped_no_confidence_count += 1
            continue

        try:
            outcome = await postprocess_auto_review(db=db, extraction=extraction, raw_post=raw_post)
        except Exception as exc:  # noqa: BLE001
            if len(errors) < 20:
                errors.append(
                    AutoReviewBackfillErrorRead(
                        extraction_id=extraction_id,
                        raw_post_id=raw_post.id,
                        error=_build_last_error(exc)[:240],
                    )
                )
            continue

        if outcome == "approved":
            approved_count += 1
        elif outcome == "rejected":
            rejected_count += 1
        else:
            skipped_no_result_count += 1

    await db.commit()
    return AdminBackfillAutoReviewRead(
        scanned=scanned,
        approved_count=approved_count,
        rejected_count=rejected_count,
        skipped_no_result_count=skipped_no_result_count,
        skipped_no_confidence_count=skipped_no_confidence_count,
        skipped_already_terminal_count=skipped_already_terminal_count,
        errors=errors[:20],
    )


@app.delete("/admin/extractions/pending", response_model=AdminDeletePendingExtractionsRead)
async def delete_pending_or_failed_extractions(
    confirm: str = Query(...),
    enable_cascade: bool = Query(default=False),
    also_delete_raw_posts: bool = Query(default=False),
    author_handle: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    _require_raw_delete_cascade_guard(also_delete_raw_posts=also_delete_raw_posts, enable_cascade=enable_cascade)

    scoped_handle = _normalize_author_handle(author_handle) if (author_handle or "").strip() else None
    result = await db.execute(
        select(PostExtraction).options(selectinload(PostExtraction.raw_post)).order_by(PostExtraction.id.asc())
    )
    all_extractions = list(result.scalars().all())
    filtered: list[PostExtraction] = []
    for extraction in all_extractions:
        if not (_is_active_extraction(extraction, extraction.raw_post) or _is_failed_extraction(extraction, extraction.raw_post)):
            continue
        raw_post = extraction.raw_post
        if scoped_handle and (
            raw_post is None or _normalize_author_handle(raw_post.author_handle) != scoped_handle
        ):
            continue
        filtered.append(extraction)

    targeted_extraction_ids = {item.id for item in filtered if item.id is not None}
    targeted_raw_post_ids = {item.raw_post_id for item in filtered if item.raw_post_id is not None}
    deleted_raw_posts_count = 0

    if also_delete_raw_posts and targeted_raw_post_ids:
        safe_to_delete_raw_post_ids: set[int] = set()
        for raw_post_id in targeted_raw_post_ids:
            rows_result = await db.execute(select(PostExtraction).where(PostExtraction.raw_post_id == raw_post_id))
            all_ids_for_post = {
                item.id
                for item in rows_result.scalars().all()
                if getattr(item, "id", None) is not None
            }
            if all_ids_for_post and all_ids_for_post.issubset(targeted_extraction_ids):
                safe_to_delete_raw_post_ids.add(raw_post_id)

        for raw_post_id in safe_to_delete_raw_post_ids:
            raw_post = await db.get(RawPost, raw_post_id)
            if raw_post is None:
                continue
            await db.delete(raw_post)
            deleted_raw_posts_count += 1

        filtered = [item for item in filtered if item.raw_post_id not in safe_to_delete_raw_post_ids]

    for extraction in filtered:
        await db.delete(extraction)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action="admin/extractions/pending",
                hint="raw_posts or related rows are still referenced by other records",
            ),
        ) from exc
    return AdminDeletePendingExtractionsRead(
        deleted_extractions_count=len(targeted_extraction_ids),
        deleted_raw_posts_count=deleted_raw_posts_count,
        scoped_author_handle=scoped_handle,
    )


@app.delete("/admin/kols/{kol_id}", response_model=AdminHardDeleteRead)
async def admin_delete_kol(
    kol_id: int = Path(ge=1),
    confirm: str = Query(...),
    enable_cascade: bool = Query(default=False),
    also_delete_raw_posts: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    _require_raw_delete_cascade_guard(also_delete_raw_posts=also_delete_raw_posts, enable_cascade=enable_cascade)

    kol = await db.get(Kol, kol_id)
    if kol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kol not found")

    counts = _admin_delete_counts_template()

    raw_posts_result = await db.execute(select(RawPost).order_by(RawPost.id.asc()))
    related_raw_posts = [item for item in raw_posts_result.scalars().all() if item.kol_id == kol_id]
    related_raw_post_ids = {item.id for item in related_raw_posts if item.id is not None}

    extractions_result = await db.execute(select(PostExtraction).order_by(PostExtraction.id.asc()))
    related_extractions = [
        item
        for item in extractions_result.scalars().all()
        if item.raw_post_id in related_raw_post_ids
    ]
    for extraction in related_extractions:
        await db.delete(extraction)
        counts["post_extractions"] += 1

    views_result = await db.execute(select(KolView).order_by(KolView.id.asc()))
    related_views = [item for item in views_result.scalars().all() if item.kol_id == kol_id]
    related_view_ids = {item.id for item in related_views if item.id is not None}

    if related_view_ids:
        all_extractions_result = await db.execute(select(PostExtraction).order_by(PostExtraction.id.asc()))
        for extraction in all_extractions_result.scalars().all():
            if extraction.applied_kol_view_id in related_view_ids:
                extraction.applied_kol_view_id = None

    for view in related_views:
        await db.delete(view)
        counts["kol_views"] += 1

    digests_result = await db.execute(select(DailyDigest).order_by(DailyDigest.id.asc()))
    related_digests = [
        item
        for item in digests_result.scalars().all()
        if _digest_mentions_kol(item.content, kol_id=kol_id)
    ]
    for digest in related_digests:
        await db.delete(digest)
        counts["daily_digests"] += 1

    profile_weights_result = await db.execute(select(ProfileKolWeight).order_by(ProfileKolWeight.id.asc()))
    related_profile_weights = [item for item in profile_weights_result.scalars().all() if item.kol_id == kol_id]
    for item in related_profile_weights:
        await db.delete(item)
        counts["profile_kol_weights"] += 1

    if also_delete_raw_posts:
        for raw_post in related_raw_posts:
            await db.delete(raw_post)
            counts["raw_posts"] += 1
    else:
        for raw_post in related_raw_posts:
            raw_post.kol_id = None

    await db.delete(kol)
    counts["kols"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action=f"admin/kols/{kol_id}",
                hint="delete dependent raw_posts/post_extractions or enable cascade for raw_posts",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_kol",
        target=f"kol:{kol_id}",
        derived_only=not also_delete_raw_posts,
        enable_cascade=enable_cascade,
        also_delete_raw_posts=also_delete_raw_posts,
        counts=counts,
    )


@app.delete("/admin/assets/{asset_id}", response_model=AdminHardDeleteRead)
async def admin_delete_asset(
    asset_id: int = Path(ge=1),
    confirm: str = Query(...),
    enable_cascade: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)

    asset = await db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")

    counts = _admin_delete_counts_template()
    views_result = await db.execute(select(KolView).order_by(KolView.id.asc()))
    related_views = [item for item in views_result.scalars().all() if item.asset_id == asset_id]
    related_view_ids = {item.id for item in related_views if item.id is not None}

    if related_view_ids:
        all_extractions_result = await db.execute(select(PostExtraction).order_by(PostExtraction.id.asc()))
        for extraction in all_extractions_result.scalars().all():
            if extraction.applied_kol_view_id in related_view_ids:
                extraction.applied_kol_view_id = None

    for view in related_views:
        await db.delete(view)
        counts["kol_views"] += 1

    digests_result = await db.execute(select(DailyDigest).order_by(DailyDigest.id.asc()))
    related_digests = [
        item
        for item in digests_result.scalars().all()
        if _digest_mentions_asset(item.content, asset_id=asset_id)
    ]
    for digest in related_digests:
        await db.delete(digest)
        counts["daily_digests"] += 1

    if enable_cascade:
        aliases_result = await db.execute(select(AssetAlias).order_by(AssetAlias.id.asc()))
        related_aliases = [item for item in aliases_result.scalars().all() if item.asset_id == asset_id]
        for alias in related_aliases:
            await db.delete(alias)
            counts["asset_aliases"] += 1

        await db.delete(asset)
        counts["assets"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action=f"admin/assets/{asset_id}",
                hint="delete dependent kol_views/aliases first, or retry with enable_cascade=true",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_asset",
        target=f"asset:{asset_id}",
        derived_only=not enable_cascade,
        enable_cascade=enable_cascade,
        also_delete_raw_posts=False,
        counts=counts,
    )


@app.delete("/admin/digests", response_model=AdminHardDeleteRead)
async def admin_delete_digests_by_date(
    confirm: str = Query(...),
    digest_date: date = Query(...),
    profile_id: int = Query(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    counts = _admin_delete_counts_template()

    result = await db.execute(select(DailyDigest).order_by(DailyDigest.id.asc()))
    matched = [
        item
        for item in result.scalars().all()
        if item.digest_date == digest_date and item.profile_id == profile_id
    ]
    if not matched:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")

    for item in matched:
        await db.delete(item)
        counts["daily_digests"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action="admin/digests",
                hint="digest rows are still referenced by other records",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_digests_by_date",
        target=f"profile:{profile_id},date:{digest_date.isoformat()}",
        derived_only=True,
        enable_cascade=False,
        also_delete_raw_posts=False,
        counts=counts,
    )


@app.delete("/admin/digests/{digest_id}", response_model=AdminHardDeleteRead)
async def admin_delete_digest_by_id(
    digest_id: int = Path(ge=1),
    confirm: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    counts = _admin_delete_counts_template()

    digest = await db.get(DailyDigest, digest_id)
    if digest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")

    await db.delete(digest)
    counts["daily_digests"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action=f"admin/digests/{digest_id}",
                hint="digest row is still referenced by other records",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_digest_by_id",
        target=f"digest:{digest_id}",
        derived_only=True,
        enable_cascade=False,
        also_delete_raw_posts=False,
        counts=counts,
    )


@app.delete("/admin/extractions/{extraction_id}", response_model=AdminHardDeleteRead)
async def admin_delete_extraction(
    extraction_id: int = Path(ge=1),
    confirm: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    counts = _admin_delete_counts_template()

    extraction = await db.get(PostExtraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")

    await db.delete(extraction)
    counts["post_extractions"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action=f"admin/extractions/{extraction_id}",
                hint="delete dependent records first",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_extraction",
        target=f"extraction:{extraction_id}",
        derived_only=True,
        enable_cascade=False,
        also_delete_raw_posts=False,
        counts=counts,
    )


@app.delete("/admin/kol-views/{kol_view_id}", response_model=AdminHardDeleteRead)
async def admin_delete_kol_view(
    kol_view_id: int = Path(ge=1),
    confirm: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    _require_destructive_admin_guard(confirm=confirm)
    counts = _admin_delete_counts_template()

    kol_view = await db.get(KolView, kol_view_id)
    if kol_view is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kol view not found")

    all_extractions_result = await db.execute(select(PostExtraction).order_by(PostExtraction.id.asc()))
    for extraction in all_extractions_result.scalars().all():
        if extraction.applied_kol_view_id == kol_view_id:
            extraction.applied_kol_view_id = None

    await db.delete(kol_view)
    counts["kol_views"] += 1

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=_build_fk_conflict_detail(
                action=f"admin/kol-views/{kol_view_id}",
                hint="delete dependent records first",
            ),
        ) from exc

    return AdminHardDeleteRead(
        operation="delete_kol_view",
        target=f"kol_view:{kol_view_id}",
        derived_only=True,
        enable_cascade=False,
        also_delete_raw_posts=False,
        counts=counts,
    )


async def _create_failed_extraction(
    db: AsyncSession,
    *,
    raw_post: RawPost,
    last_error: str,
    error_category: str,
    batch_retry_count: int,
) -> PostExtraction:
    settings = get_settings()
    provider_detected = detect_provider_from_base_url(settings.openai_base_url)
    output_mode = resolve_extraction_output_mode(settings.openai_base_url)
    extraction = PostExtraction(
        raw_post_id=raw_post.id,
        status=ExtractionStatus.pending,
        extracted_json={
            **default_extracted_json(raw_post),
            "meta": {
                "retry_exhausted": True,
                "retry_source": "extract_batch",
                "error_category": error_category,
                "batch_retry_count": batch_retry_count,
                "provider_detected": provider_detected,
                "output_mode_used": output_mode,
                "parse_strategy_used": "failed",
                "raw_len": 0,
                "repaired": False,
                "parse_error": True,
            },
        },
        model_name=settings.openai_model,
        extractor_name="openai_structured",
        prompt_version=settings.prompt_version,
        raw_model_output=None,
        parsed_model_output=None,
        last_error=last_error[:2048],
    )
    db.add(extraction)
    await db.flush()
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


def _build_runtime_settings_read(settings) -> RuntimeSettingsRead:
    _refresh_runtime_budget_state(settings)
    default_budget_total = _get_default_call_budget_total(settings)
    runtime_budget_total = _get_runtime_openai_call_budget_total(settings)
    throttle = _get_runtime_throttle(settings)
    provider_detected = detect_provider_from_base_url(settings.openai_base_url)
    extraction_output_mode = resolve_extraction_output_mode(settings.openai_base_url)
    return RuntimeSettingsRead(
        extractor_mode=settings.extractor_mode.strip().lower(),
        provider_detected=provider_detected,
        extraction_output_mode=extraction_output_mode,
        model=settings.openai_model,
        has_api_key=bool(settings.openai_api_key.strip()),
        base_url=settings.openai_base_url,
        budget_remaining=get_openai_call_budget_remaining(settings, budget_total=runtime_budget_total),
        budget_total=runtime_budget_total,
        default_budget_total=default_budget_total,
        call_budget_override_enabled=RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE is not None,
        call_budget_override_value=RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE,
        override_value=RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE,
        window_start=RUNTIME_BUDGET_WINDOW_START or _utc_now(),
        window_end=RUNTIME_BUDGET_WINDOW_END or _utc_now(),
        max_output_tokens=max(1, settings.openai_max_output_tokens),
        auto_reject_confidence_threshold=max(0, min(100, settings.auto_reject_confidence_threshold)),
        throttle=throttle,
        burst={
            "enabled": bool(RUNTIME_BURST_STATE["enabled"]),
            "mode": RUNTIME_BURST_STATE.get("mode"),
            "expires_at": RUNTIME_BURST_STATE["expires_at"],
        },
        runtime_overrides={
            "call_budget": RUNTIME_OPENAI_CALL_BUDGET_OVERRIDE is not None,
            "burst": bool(RUNTIME_BURST_STATE["enabled"]),
            "throttle": bool(RUNTIME_THROTTLE_OVERRIDE),
        },
    )


@app.post("/raw-posts/{raw_post_id}/extract", response_model=PostExtractionRead, status_code=status.HTTP_201_CREATED)
async def extract_raw_post(
    raw_post_id: int,
    force: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    raw_post = await db.get(RawPost, raw_post_id)
    if raw_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="raw post not found")
    _check_reextract_rate_limit(raw_post_id)

    extraction = await create_pending_extraction(db, raw_post, force_reextract=force)
    await db.commit()
    await db.refresh(extraction)
    return extraction


@app.get("/extractor-status", response_model=ExtractorStatusRead)
def get_extractor_status():
    settings = get_settings()
    runtime_budget_total = _get_runtime_openai_call_budget_total(settings)
    return ExtractorStatusRead(
        mode=settings.extractor_mode.strip().lower(),
        has_api_key=bool(settings.openai_api_key.strip()),
        default_model=settings.openai_model,
        base_url=settings.openai_base_url,
        call_budget_remaining=get_openai_call_budget_remaining(settings, budget_total=runtime_budget_total),
        max_output_tokens=max(1, settings.openai_max_output_tokens),
    )


@app.get("/settings/runtime", response_model=RuntimeSettingsRead)
def get_runtime_settings():
    settings = get_settings()
    return _build_runtime_settings_read(settings)


@app.post("/settings/runtime/call-budget", response_model=RuntimeSettingsRead)
def set_runtime_call_budget(payload: RuntimeCallBudgetUpdateRequest):
    # TODO: add auth before exposing this endpoint in non-local environments.
    _set_runtime_openai_call_budget_total(payload.call_budget)
    return get_runtime_settings()


@app.post("/settings/runtime/call-budget/clear", response_model=RuntimeSettingsRead)
def clear_runtime_call_budget_override():
    _clear_runtime_openai_call_budget_override()
    return get_runtime_settings()


@app.post("/settings/runtime/burst", response_model=RuntimeSettingsRead)
def set_runtime_burst(payload: RuntimeBurstUpdateRequest):
    _set_runtime_burst(
        enabled=payload.enabled,
        mode=payload.mode,
        call_budget=payload.call_budget,
        duration_minutes=payload.duration_minutes,
    )
    return get_runtime_settings()


@app.post("/settings/runtime/throttle", response_model=RuntimeSettingsRead)
def set_runtime_throttle(payload: RuntimeThrottleUpdateRequest):
    settings = get_settings()
    _set_runtime_throttle(
        settings,
        max_concurrency=payload.max_concurrency,
        max_rpm=payload.max_rpm,
        batch_size=payload.batch_size,
        batch_sleep_ms=payload.batch_sleep_ms,
    )
    return get_runtime_settings()


@app.get("/dashboard", response_model=DashboardRead)
async def get_dashboard(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=days)
    stats_cutoff = now - timedelta(hours=24)
    views_cutoff_24h = now - timedelta(hours=24)
    views_cutoff_7d = now - timedelta(days=7)

    pending_count_result = await db.execute(
        select(func.count(PostExtraction.id)).where(PostExtraction.status == ExtractionStatus.pending)
    )
    pending_extractions_count = pending_count_result.scalar() or 0

    latest_pending_result = await db.execute(
        select(PostExtraction)
        .options(selectinload(PostExtraction.raw_post))
        .where(PostExtraction.status == ExtractionStatus.pending)
        .order_by(PostExtraction.created_at.desc(), PostExtraction.id.desc())
        .limit(20)
    )
    latest_pending = list(latest_pending_result.scalars().all())
    latest_pending_extractions = [
        DashboardPendingExtractionRead(
            id=item.id,
            platform=item.raw_post.platform,
            author_handle=item.raw_post.author_handle,
            url=item.raw_post.url,
            posted_at=item.raw_post.posted_at,
            created_at=item.created_at,
        )
        for item in latest_pending
        if item.raw_post is not None
    ]

    assets_result = await db.execute(select(Asset).order_by(Asset.id.asc()))
    assets = list(assets_result.scalars().all())
    asset_map = {item.id: item for item in assets}

    kols_result = await db.execute(select(Kol).order_by(Kol.id.asc()))
    kols = list(kols_result.scalars().all())
    kol_map = {item.id: item for item in kols}

    views_result = await db.execute(select(KolView).order_by(KolView.created_at.desc(), KolView.id.desc()))
    all_views = list(views_result.scalars().all())

    asset_counts_24h: dict[int, int] = defaultdict(int)
    asset_counts_7d: dict[int, int] = defaultdict(int)
    latest_by_asset_horizon: dict[tuple[int, str], KolView] = {}
    kol_counts_7d: dict[int, int] = defaultdict(int)
    kol_asset_counts_7d: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for view in all_views:
        view_created = view.created_at
        if view_created is None:
            continue
        horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
        latest_key = (view.asset_id, horizon_value)
        current_latest = latest_by_asset_horizon.get(latest_key)
        if current_latest is None or is_newer_view(view, current_latest):
            latest_by_asset_horizon[latest_key] = view
        if view_created >= views_cutoff_24h:
            asset_counts_24h[view.asset_id] += 1
        if view_created >= views_cutoff_7d:
            asset_counts_7d[view.asset_id] += 1
            kol_counts_7d[view.kol_id] += 1
            kol_asset_counts_7d[view.kol_id][view.asset_id] += 1

    dashboard_assets: list[DashboardAssetRead] = []
    order_index = {value: idx for idx, value in enumerate(HORIZON_ORDER)}
    for asset in assets:
        latest_views: list[DashboardAssetLatestViewRead] = []
        for horizon in HORIZON_ORDER:
            key = (asset.id, horizon)
            view = latest_by_asset_horizon.get(key)
            if view is None:
                continue
            kol = kol_map.get(view.kol_id)
            latest_views.append(
                DashboardAssetLatestViewRead(
                    kol_view_id=view.id,
                    horizon=view.horizon,
                    stance=view.stance,
                    confidence=view.confidence,
                    summary=view.summary,
                    as_of=view.as_of,
                    created_at=view.created_at,
                    kol_id=view.kol_id,
                    kol_display_name=kol.display_name if kol is not None else None,
                    kol_handle=kol.handle if kol is not None else None,
                )
            )
        latest_views.sort(key=lambda item: order_index.get(item.horizon, 999))
        dashboard_assets.append(
            DashboardAssetRead(
                id=asset.id,
                symbol=asset.symbol,
                name=asset.name,
                market=asset.market,
                new_views_24h=asset_counts_24h.get(asset.id, 0),
                new_views_7d=asset_counts_7d.get(asset.id, 0),
                latest_views_by_horizon=latest_views,
            )
        )

    active_kols_7d: list[DashboardActiveKolRead] = []
    for kol_id, count in sorted(kol_counts_7d.items(), key=lambda item: (-item[1], item[0]))[:10]:
        kol = kol_map.get(kol_id)
        if kol is None:
            continue
        top_assets = sorted(
            kol_asset_counts_7d[kol_id].items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
        active_kols_7d.append(
            DashboardActiveKolRead(
                kol_id=kol.id,
                display_name=kol.display_name,
                handle=kol.handle,
                platform=kol.platform,
                views_count_7d=count,
                top_assets=[
                    DashboardActiveKolAssetRead(
                        asset_id=asset_id,
                        symbol=asset_map[asset_id].symbol if asset_id in asset_map else str(asset_id),
                        views_count=asset_count,
                    )
                    for asset_id, asset_count in top_assets
                ],
            )
        )

    stats_result = await db.execute(
        select(PostExtraction)
        .order_by(PostExtraction.created_at.desc(), PostExtraction.id.desc())
        .limit(5000)
    )
    all_recent = list(stats_result.scalars().all())
    window_items = [item for item in all_recent if item.created_at and item.created_at >= stats_cutoff]
    dummy_count = sum(1 for item in window_items if item.extractor_name == "dummy")
    openai_count = sum(1 for item in window_items if item.extractor_name.startswith("openai"))
    error_count = sum(1 for item in window_items if item.last_error is not None)

    approved_view_ids: set[int] = {
        item.applied_kol_view_id
        for item in all_recent
        if item.status == ExtractionStatus.approved and item.applied_kol_view_id is not None
    }
    approved_views_in_window = [
        view
        for view in all_views
        if view.id in approved_view_ids and view.created_at is not None and view.created_at >= cutoff
    ]

    top_asset_stats: dict[int, list[int]] = defaultdict(list)
    for view in approved_views_in_window:
        top_asset_stats[view.asset_id].append(view.confidence)
    top_assets = [
        DashboardTopAssetRead(
            asset_id=asset_id,
            symbol=asset_map[asset_id].symbol if asset_id in asset_map else str(asset_id),
            market=asset_map[asset_id].market if asset_id in asset_map else None,
            views_count_7d=len(confidences),
            avg_confidence_7d=(sum(confidences) / len(confidences)) if confidences else 0,
        )
        for asset_id, confidences in top_asset_stats.items()
    ]
    top_assets.sort(key=lambda item: (-item.views_count_7d, -item.avg_confidence_7d, item.asset_id))
    top_assets = top_assets[:20]

    clarity: list[DashboardClarityRead] = []
    top_asset_ids = {item.asset_id for item in top_assets}
    if top_asset_ids:
        grouped_counts: dict[str, dict[Stance, int]] = defaultdict(
            lambda: {Stance.bull: 0, Stance.bear: 0, Stance.neutral: 0}
        )
        for view in approved_views_in_window:
            if view.asset_id not in top_asset_ids:
                continue
            horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
            grouped_counts[horizon_value][view.stance] += 1

        for horizon in sorted(grouped_counts.keys(), key=lambda value: order_index.get(value, 999)):
            counts = grouped_counts[horizon]
            bull_count = counts[Stance.bull]
            bear_count = counts[Stance.bear]
            neutral_count = counts[Stance.neutral]
            clarity.append(
                DashboardClarityRead(
                    horizon=horizon,
                    bull_count=bull_count,
                    bear_count=bear_count,
                    neutral_count=neutral_count,
                    clarity=calc_clarity(bull_count=bull_count, bear_count=bear_count),
                )
            )

    new_views_24h = sum(asset_counts_24h.values())
    new_views_7d = sum(asset_counts_7d.values())
    return DashboardRead(
        pending_extractions_count=int(pending_extractions_count),
        latest_pending_extractions=latest_pending_extractions,
        top_assets=top_assets,
        clarity=clarity,
        extraction_stats=DashboardExtractionStatsRead(
            window_hours=24,
            extraction_count=len(window_items),
            dummy_count=dummy_count,
            openai_count=openai_count,
            error_count=error_count,
        ),
        new_views_24h=new_views_24h,
        new_views_7d=new_views_7d,
        assets=dashboard_assets,
        active_kols_7d=active_kols_7d,
    )


@app.post("/digests/generate", response_model=DailyDigestRead)
async def generate_daily_digest(
    digest_date: date = Query(alias="date"),
    days: int = Query(default=7, ge=1, le=90),
    to_ts: datetime | None = Query(default=None),
    profile_id: int = Query(default=1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await generate_daily_digest_service(
        db,
        digest_date=digest_date,
        days=days,
        to_ts=to_ts,
        profile_id=profile_id,
    )


@app.get("/digests", response_model=DailyDigestRead)
async def get_daily_digest(
    digest_date: date = Query(alias="date"),
    version: int | None = Query(default=None, ge=1),
    profile_id: int = Query(default=1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await get_daily_digest_by_date_service(
        db,
        digest_date=digest_date,
        version=version,
        profile_id=profile_id,
    )


@app.get("/digests/dates", response_model=list[date])
async def list_daily_digest_dates(
    profile_id: int = Query(default=1, ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await list_daily_digest_dates_service(db, profile_id=profile_id)


@app.get("/digests/{digest_id}", response_model=DailyDigestRead)
async def get_daily_digest_by_id(
    digest_id: int = Path(ge=1),
    db: AsyncSession = Depends(get_db),
):
    return await get_daily_digest_by_id_service(db, digest_id=digest_id)


@app.get("/extractions", response_model=list[PostExtractionWithRawPostRead])
async def list_extractions(
    status: ExtractionStatus = Query(default=ExtractionStatus.pending),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(PostExtraction)
        .options(selectinload(PostExtraction.raw_post))
        .where(PostExtraction.status == status)
        .order_by(PostExtraction.created_at.desc(), PostExtraction.id.desc())
        .offset(offset)
        .limit(limit)
    )
    items = list(result.scalars().all())
    for extraction in items:
        _attach_extraction_auto_approve_settings(extraction)
    return items


@app.get("/extractions/{extraction_id}", response_model=PostExtractionWithRawPostRead)
async def get_extraction(extraction_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PostExtraction)
        .options(selectinload(PostExtraction.raw_post))
        .where(PostExtraction.id == extraction_id)
    )
    extraction = result.scalar_one_or_none()
    if extraction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")
    await _attach_auto_applied_metadata(db, extraction)
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


@app.post("/extractions/{extraction_id}/re-extract", response_model=PostExtractionRead, status_code=status.HTTP_201_CREATED)
async def force_reextract_extraction(
    extraction_id: int,
    db: AsyncSession = Depends(get_db),
):
    source = await db.get(PostExtraction, extraction_id)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")

    raw_post = await db.get(RawPost, source.raw_post_id)
    if raw_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="raw post not found")

    _check_reextract_rate_limit(raw_post.id)
    triggered_by = (source.reviewed_by or EXTRACTION_REVIEWER).strip() or EXTRACTION_REVIEWER
    extraction = await create_pending_extraction(
        db,
        raw_post,
        allow_budget_fallback=False,
        force_reextract=True,
        force_reextract_triggered_by=triggered_by,
        source_extraction_id=source.id,
    )
    await db.commit()
    await db.refresh(extraction)
    return extraction


@app.post("/extractions/{extraction_id}/approve", response_model=PostExtractionRead)
async def approve_extraction(
    extraction_id: int,
    payload: ExtractionApproveRequest,
    db: AsyncSession = Depends(get_db),
):
    extraction = await db.get(PostExtraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")
    if extraction.status != ExtractionStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"extraction already {extraction.status.value}",
        )

    kol = await db.get(Kol, payload.kol_id)
    if kol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kol not found")
    asset = await db.get(Asset, payload.asset_id)
    if asset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="asset not found")

    extraction.status = ExtractionStatus.approved
    extraction.reviewed_at = datetime.now(UTC)
    extraction.reviewed_by = EXTRACTION_REVIEWER
    extraction.review_note = None
    raw_post = await db.get(RawPost, extraction.raw_post_id)
    if raw_post is not None:
        raw_post.review_status = ReviewStatus.approved
        raw_post.reviewed_at = extraction.reviewed_at
        raw_post.reviewed_by = extraction.reviewed_by

    inserted_count = 0
    skipped_count = 0
    try:
        view, inserted, _ = await _insert_kol_view_if_absent(
            db,
            kol_id=payload.kol_id,
            asset_id=payload.asset_id,
            stance=payload.stance.value,
            horizon=payload.horizon.value,
            confidence=payload.confidence,
            summary=payload.summary.strip(),
            source_url=payload.source_url.strip(),
            as_of=payload.as_of,
        )
        if inserted:
            inserted_count = 1
        else:
            skipped_count = 1
        extraction.applied_kol_view_id = view.id
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate kol view")

    await db.refresh(extraction)
    setattr(extraction, "approve_inserted_count", inserted_count)
    setattr(extraction, "approve_skipped_count", skipped_count)
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


@app.post("/extractions/{extraction_id}/approve-batch", response_model=PostExtractionRead)
async def approve_extraction_batch(
    extraction_id: int,
    payload: ExtractionApproveBatchRequest,
    db: AsyncSession = Depends(get_db),
):
    extraction = await db.get(PostExtraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")
    if extraction.status != ExtractionStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"extraction already {extraction.status.value}",
        )

    kol = await db.get(Kol, payload.kol_id)
    if kol is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="kol not found")

    inserted_count = 0
    skipped_count = 0
    first_applied_kol_view_id: int | None = None
    try:
        for item in payload.views:
            asset = await db.get(Asset, item.asset_id)
            if asset is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"asset {item.asset_id} not found")
            view, inserted, _ = await _insert_kol_view_if_absent(
                db,
                kol_id=payload.kol_id,
                asset_id=item.asset_id,
                stance=item.stance.value,
                horizon=item.horizon.value,
                confidence=item.confidence,
                summary=item.summary.strip(),
                source_url=item.source_url.strip(),
                as_of=item.as_of,
            )
            if first_applied_kol_view_id is None:
                first_applied_kol_view_id = view.id
            if inserted:
                inserted_count += 1
            else:
                skipped_count += 1
        extraction.status = ExtractionStatus.approved
        extraction.reviewed_at = datetime.now(UTC)
        extraction.reviewed_by = EXTRACTION_REVIEWER
        extraction.review_note = None
        extraction.applied_kol_view_id = first_applied_kol_view_id
        raw_post = await db.get(RawPost, extraction.raw_post_id)
        if raw_post is not None:
            raw_post.review_status = ReviewStatus.approved
            raw_post.reviewed_at = extraction.reviewed_at
            raw_post.reviewed_by = extraction.reviewed_by
        await db.commit()
    except HTTPException:
        await db.rollback()
        raise
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="duplicate kol view")

    await db.refresh(extraction)
    setattr(extraction, "approve_inserted_count", inserted_count)
    setattr(extraction, "approve_skipped_count", skipped_count)
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


@app.post("/extractions/{extraction_id}/reject", response_model=PostExtractionRead)
async def reject_extraction(
    extraction_id: int,
    payload: ExtractionRejectRequest,
    db: AsyncSession = Depends(get_db),
):
    extraction = await db.get(PostExtraction, extraction_id)
    if extraction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="extraction not found")
    if extraction.status != ExtractionStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"extraction already {extraction.status.value}",
        )

    extraction.status = ExtractionStatus.rejected
    extraction.reviewed_at = datetime.now(UTC)
    extraction.reviewed_by = EXTRACTION_REVIEWER
    extraction.review_note = (payload.reason or "").strip() or None
    extraction.applied_kol_view_id = None
    raw_post = await db.get(RawPost, extraction.raw_post_id)
    if raw_post is not None:
        raw_post.review_status = ReviewStatus.rejected
        raw_post.reviewed_at = extraction.reviewed_at
        raw_post.reviewed_by = extraction.reviewed_by
    await db.commit()
    await db.refresh(extraction)
    return extraction
