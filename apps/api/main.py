from collections import defaultdict, deque
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from db import get_db
from enums import HORIZON_ORDER, Stance
from enums import ExtractionStatus
from models import Asset, AssetAlias, Kol, KolView, PostExtraction, RawPost
from schemas import (
    AssetCreate,
    AssetAliasCreate,
    AssetAliasRead,
    AssetViewsGroupRead,
    AssetViewsMetaRead,
    AssetRead,
    AssetViewsRead,
    AutoAppliedViewRead,
    DashboardClarityRead,
    DashboardExtractionStatsRead,
    DashboardPendingExtractionRead,
    DashboardRead,
    DashboardTopAssetRead,
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
    RawPostCreate,
    RawPostRead,
)
from services.extraction import (
    DummyExtractor,
    OpenAIFallbackError,
    OpenAIExtractor,
    default_extracted_json,
    normalize_extracted_json,
    get_openai_call_budget_remaining,
    reset_openai_call_budget_counter,
    select_extractor,
    try_consume_openai_call_budget,
)
from services.prompts import build_extract_prompt
from settings import get_settings

app = FastAPI(title="InvestPulse API")
EXTRACTION_REVIEWER = "human-review"
AUTO_EXTRACTION_REVIEWER = "auto"
REEXTRACT_ATTEMPTS: dict[int, deque[datetime]] = defaultdict(deque)

# 先放开本地前端跨域，后面再收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def reset_reextract_rate_limiter() -> None:
    REEXTRACT_ATTEMPTS.clear()


def reset_runtime_counters() -> None:
    reset_reextract_rate_limiter()
    reset_openai_call_budget_counter()


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


def _apply_alias_symbol_mapping(extracted_json: dict[str, Any], alias_to_symbol: dict[str, str]) -> None:
    if not alias_to_symbol:
        return

    for asset in extracted_json.get("assets", []):
        if not isinstance(asset, dict):
            continue
        symbol = asset.get("symbol")
        if not isinstance(symbol, str):
            continue
        mapped = alias_to_symbol.get(_normalize_alias_key(symbol))
        if mapped:
            asset["symbol"] = mapped

    for view in extracted_json.get("asset_views", []):
        if not isinstance(view, dict):
            continue
        symbol = view.get("symbol")
        if not isinstance(symbol, str):
            continue
        mapped = alias_to_symbol.get(_normalize_alias_key(symbol))
        if mapped:
            view["symbol"] = mapped


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
    extraction_mode = meta.get("extraction_mode") if isinstance(meta, dict) else None
    if extraction_mode == "json_mode" and "openrouter.ai" in settings_base_url.lower():
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
) -> None:
    settings = get_settings()
    extraction.auto_applied_count = 0
    extraction.auto_policy = None
    extraction.auto_applied_kol_view_ids = None
    setattr(extraction, "auto_applied_asset_view_keys", [])
    setattr(extraction, "auto_applied_views", [])
    if not settings.auto_approve_enabled or extraction.status != ExtractionStatus.pending:
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


async def create_pending_extraction(db: AsyncSession, raw_post: RawPost) -> PostExtraction:
    settings = get_settings()
    extraction_input, truncation_meta = _build_extraction_input(raw_post)
    extractor = select_extractor(settings)
    assets_for_prompt = await _load_assets_for_prompt(db, settings.max_assets_in_prompt)
    try:
        aliases_for_prompt = await _load_aliases_for_prompt(db)
    except Exception:  # noqa: BLE001
        aliases_for_prompt = []
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

    if isinstance(extractor, OpenAIExtractor) and not try_consume_openai_call_budget(settings):
        budget_exhausted = True
        extractor = DummyExtractor()
        extractor.set_prompt_bundle(prompt_bundle)
        fallback_reason = "budget_exhausted"

    try:
        extracted_json = extractor.extract(extraction_input)
    except Exception as exc:  # noqa: BLE001
        last_error = _build_last_error(exc)
        if isinstance(exc, OpenAIFallbackError):
            fallback_reason = exc.fallback_reason
        if isinstance(extractor, OpenAIExtractor):
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
    )
    _apply_alias_symbol_mapping(extracted_json, _build_alias_to_symbol_map(aliases_for_prompt))

    base_meta = extracted_json.get("meta") if isinstance(extracted_json, dict) else None
    safe_meta = base_meta if isinstance(base_meta, dict) else {}
    if "extraction_mode" not in safe_meta:
        safe_meta = {**safe_meta, "extraction_mode": "dummy" if isinstance(extractor, DummyExtractor) else "structured"}
    if fallback_reason == "structured_unsupported" and "fallback_reason" not in safe_meta:
        safe_meta = {**safe_meta, "fallback_reason": "structured_unsupported"}
    extracted_json["meta"] = safe_meta

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
    try:
        await _auto_apply_extraction_views(db, extraction=extraction, raw_post=raw_post)
    except Exception as exc:  # noqa: BLE001
        auto_error = _build_last_error(exc)
        extraction.last_error = auto_error if not extraction.last_error else f"{extraction.last_error}; auto_apply={auto_error}"
    _attach_extraction_auto_approve_settings(extraction)
    return extraction


@app.get("/assets", response_model=list[AssetRead])
async def list_assets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Asset).order_by(Asset.id.asc()))
    return list(result.scalars().all())


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

    alias_row = AssetAlias(asset_id=asset_id, alias=alias)
    db.add(alias_row)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="alias already exists")
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


@app.post("/raw-posts/{raw_post_id}/extract", response_model=PostExtractionRead, status_code=status.HTTP_201_CREATED)
async def extract_raw_post(raw_post_id: int, db: AsyncSession = Depends(get_db)):
    raw_post = await db.get(RawPost, raw_post_id)
    if raw_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="raw post not found")
    _check_reextract_rate_limit(raw_post_id)

    extraction = await create_pending_extraction(db, raw_post)
    await db.commit()
    await db.refresh(extraction)
    return extraction


@app.get("/extractor-status", response_model=ExtractorStatusRead)
def get_extractor_status():
    settings = get_settings()
    return ExtractorStatusRead(
        mode=settings.extractor_mode.strip().lower(),
        has_api_key=bool(settings.openai_api_key.strip()),
        default_model=settings.openai_model,
        base_url=settings.openai_base_url,
        call_budget_remaining=get_openai_call_budget_remaining(settings),
        max_output_tokens=max(1, settings.openai_max_output_tokens),
    )


@app.get("/dashboard", response_model=DashboardRead)
async def get_dashboard(
    days: int = Query(default=7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(UTC) - timedelta(days=days)
    stats_cutoff = datetime.now(UTC) - timedelta(hours=24)

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

    top_assets_result = await db.execute(
        select(
            Asset.id.label("asset_id"),
            Asset.symbol.label("symbol"),
            Asset.market.label("market"),
            func.count(KolView.id).label("views_count"),
            func.avg(KolView.confidence).label("avg_confidence"),
        )
        .join(KolView, KolView.asset_id == Asset.id)
        .join(PostExtraction, PostExtraction.applied_kol_view_id == KolView.id)
        .where(
            PostExtraction.status == ExtractionStatus.approved,
            KolView.created_at >= cutoff,
        )
        .group_by(Asset.id, Asset.symbol, Asset.market)
        .order_by(func.count(KolView.id).desc(), func.avg(KolView.confidence).desc(), Asset.id.asc())
        .limit(20)
    )
    top_asset_rows = top_assets_result.all()
    top_assets = [
        DashboardTopAssetRead(
            asset_id=row.asset_id,
            symbol=row.symbol,
            market=row.market,
            views_count_7d=int(row.views_count),
            avg_confidence_7d=float(row.avg_confidence or 0),
        )
        for row in top_asset_rows
    ]

    clarity: list[DashboardClarityRead] = []
    top_asset_ids = [item.asset_id for item in top_assets]
    if top_asset_ids:
        stance_counts_result = await db.execute(
            select(
                KolView.horizon.label("horizon"),
                KolView.stance.label("stance"),
                func.count(KolView.id).label("count"),
            )
            .join(PostExtraction, PostExtraction.applied_kol_view_id == KolView.id)
            .where(
                PostExtraction.status == ExtractionStatus.approved,
                KolView.created_at >= cutoff,
                KolView.asset_id.in_(top_asset_ids),
            )
            .group_by(KolView.horizon, KolView.stance)
        )

        grouped_counts: dict[str, dict[Stance, int]] = defaultdict(
            lambda: {Stance.bull: 0, Stance.bear: 0, Stance.neutral: 0}
        )
        for row in stance_counts_result.all():
            grouped_counts[row.horizon.value][row.stance] = int(row.count)

        order_index = {value: idx for idx, value in enumerate(HORIZON_ORDER)}
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
    )


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
    await db.commit()
    await db.refresh(extraction)
    return extraction
