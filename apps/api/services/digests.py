from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
import json
from typing import Any, Literal

from fastapi import HTTPException, status
import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models import DailyDigest, Kol, PostExtraction, RawPost
from schemas import (
    DailyDigestAIAnalysisRead,
    DailyDigestAuthorSummaryInputRead,
    DailyDigestMetadataRead,
    DailyDigestPostSummaryRead,
    DailyDigestRead,
)
from services.profiles import DEFAULT_PROFILE_ID, load_profile_rules
from services.view_logic import calc_clarity as _calc_clarity
from services.view_logic import select_latest_views as _select_latest_views
from settings import get_settings

TimeFieldUsed = Literal["as_of", "posted_at", "created_at"]


def select_latest_views(views: list[Any]) -> list[Any]:
    return _select_latest_views(views)


def calc_clarity(*, bull_count: int, bear_count: int) -> float:
    return _calc_clarity(bull_count=bull_count, bear_count=bear_count)


def _author_summary_field_guide_text() -> str:
    return (
        "author_summaries.summaries 关键字段值语义（简要）：\n"
        "- symbol: 表示观点对应的具体交易标的（如股票名称/代码、ETF、加密资产、外汇或商品符号）。\n"
        "- market: 表示标的所属市场类型；CRYPTO=加密资产，STOCK=股票，ETF=交易型基金，FOREX=外汇，OTHER=其他。\n"
        "- stance: 表示观点方向；bull=看多，bear=看空，neutral=中性/方向不明确。\n"
        "- horizon: 表示预计影响周期；intraday=日内，1w=一周，1m=一月，3m=三月，1y=一年。"
    )


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_business_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None
    return None


def _resolve_post_business_ts(extraction: PostExtraction, raw_post: RawPost | None) -> tuple[datetime | None, TimeFieldUsed]:
    payload = extraction.extracted_json if isinstance(extraction.extracted_json, dict) else {}
    as_of_date = _parse_business_date(payload.get("as_of"))
    if as_of_date is not None:
        return datetime.combine(as_of_date, time.min, tzinfo=UTC), "as_of"

    if raw_post is not None and raw_post.posted_at is not None:
        return _ensure_utc(raw_post.posted_at), "posted_at"

    if extraction.created_at is not None:
        return _ensure_utc(extraction.created_at), "created_at"

    return None, "created_at"


def _extract_summary_text(extraction: PostExtraction, raw_post: RawPost | None) -> str:
    payload = extraction.extracted_json if isinstance(extraction.extracted_json, dict) else {}

    direct_summary = payload.get("summary")
    if isinstance(direct_summary, str) and direct_summary.strip():
        return direct_summary.strip()[:1024]

    asset_views = payload.get("asset_views")
    if isinstance(asset_views, list):
        parts: list[str] = []
        for item in asset_views:
            if not isinstance(item, dict):
                continue
            summary = item.get("summary")
            if isinstance(summary, str) and summary.strip():
                cleaned = summary.strip()
                if cleaned not in parts:
                    parts.append(cleaned)
            if len(parts) >= 3:
                break
        if parts:
            return "；".join(parts)[:1024]

    if raw_post is not None and isinstance(raw_post.content_text, str) and raw_post.content_text.strip():
        return raw_post.content_text.strip().replace("\n", " ")[:280]

    return "(empty summary)"


def _extract_title(raw_post: RawPost | None) -> str | None:
    if raw_post is None or not isinstance(raw_post.raw_json, dict):
        return None
    for key in ("title", "headline", "subject"):
        value = raw_post.raw_json.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:300]
    return None


def _clean_optional_text(value: Any, *, max_len: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned[:max_len]


def _build_author_summary_rows(
    *,
    row: DailyDigestPostSummaryRead,
    extraction: PostExtraction | None,
) -> list[dict[str, Any]]:
    base_row = {
        "time": (row.posted_at or row.business_ts).isoformat(),
        "symbol": None,
        "market": None,
        "stance": None,
        "horizon": None,
        "summary": row.summary,
        "source_url": row.source_url,
        "title": row.title,
    }
    if extraction is None or not isinstance(extraction.extracted_json, dict):
        return [base_row]

    asset_views = extraction.extracted_json.get("asset_views")
    if not isinstance(asset_views, list):
        return [base_row]

    rows: list[dict[str, Any]] = []
    for item in asset_views:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "time": (row.posted_at or row.business_ts).isoformat(),
                "symbol": _clean_optional_text(item.get("symbol"), max_len=64),
                "market": _clean_optional_text(item.get("market"), max_len=64),
                "stance": _clean_optional_text(item.get("stance"), max_len=32),
                "horizon": _clean_optional_text(item.get("horizon"), max_len=32),
                "summary": _clean_optional_text(item.get("summary"), max_len=1024) or row.summary,
                "source_url": row.source_url,
                "title": row.title,
            }
        )
    return rows or [base_row]


def _coerce_hasview(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return 1 if value == 1 else 0
    if isinstance(value, float):
        return 1 if value == 1.0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes"}:
            return 1
    return 0


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in value]
    return value


def _coerce_post_summaries(value: Any) -> list[DailyDigestPostSummaryRead]:
    if not isinstance(value, list):
        return []
    rows: list[DailyDigestPostSummaryRead] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(DailyDigestPostSummaryRead.model_validate(item))
        except Exception:
            continue
    return rows


def _coerce_author_inputs(value: Any) -> list[DailyDigestAuthorSummaryInputRead]:
    if not isinstance(value, list):
        return []
    rows: list[DailyDigestAuthorSummaryInputRead] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(DailyDigestAuthorSummaryInputRead.model_validate(item))
        except Exception:
            continue
    return rows


def _build_digest_read(*, digest: DailyDigest) -> DailyDigestRead:
    content = digest.content if isinstance(digest.content, dict) else {}
    metadata_raw = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}

    ai_raw = content.get("ai_analysis") if isinstance(content.get("ai_analysis"), dict) else {}
    ai_analysis = DailyDigestAIAnalysisRead.model_validate(
        {
            "market_overview": ai_raw.get("market_overview", ""),
            "market_signals": ai_raw.get("market_signals", ""),
            "focus_points": ai_raw.get("focus_points", []),
            "key_news": ai_raw.get("key_news", []),
            "trading_observations": ai_raw.get("trading_observations"),
        }
    )

    return DailyDigestRead(
        id=digest.id,
        profile_id=digest.profile_id,
        digest_date=digest.digest_date,
        generated_at=digest.generated_at,
        post_summaries=_coerce_post_summaries(content.get("post_summaries")),
        ai_input_by_author=_coerce_author_inputs(content.get("ai_input_by_author")),
        ai_analysis=ai_analysis,
        metadata=DailyDigestMetadataRead(
            generated_at=metadata_raw.get("generated_at", digest.generated_at),
            window_start=metadata_raw.get("window_start", digest.generated_at),
            window_end=metadata_raw.get("window_end", digest.generated_at),
            source_post_count=int(metadata_raw.get("source_post_count", 0)),
            ai_status=str(metadata_raw.get("ai_status", "skipped")),
            ai_error=metadata_raw.get("ai_error") if isinstance(metadata_raw.get("ai_error"), str) else None,
            time_field_priority=["as_of", "posted_at", "created_at"],
        ),
    )


async def _generate_ai_analysis(*, digest_date: date, by_author_payload: list[dict[str, Any]]) -> tuple[dict[str, Any], str, str | None]:
    settings = get_settings()
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return {
            "market_overview": "",
            "market_signals": "",
            "focus_points": [],
            "key_news": [],
            "trading_observations": None,
        }, "skipped_no_api_key", None

    prompt_input = {
        "digest_date": digest_date.isoformat(),
        "sections": ["市场概览", "行情提示", "关注重点", "要闻提炼", "交易观察"],
        "author_summaries": by_author_payload,
        "requirements": {
            "output_schema": {
                "market_overview": "string",
                "market_signals": "string",
                "focus_points": "string[]",
                "key_news": "string[]",
                "trading_observations": "string|null",
            },
            "section_constraints": {
                "market_overview": "用几句话概括当天市场情绪、主要驱动因素和整体节奏。",
                "market_signals": "指出当天盘面最值得注意的方向、风险偏好变化、可能影响接下来走势的信号。",
                "focus_points": "列出接下来最需要盯的事件、板块、资产或宏观变量。",
                "key_news": "提炼最重要的新闻/事件，并说明为什么重要（影响路径/传导逻辑）。",
                "trading_observations": "如果能归纳交易层面的启发则简短给出；信息不足必须输出 null，不得强行下结论。",
            },
        },
    }

    prompt = (
        "你是买方交易研究助手。只基于输入内容生成中文日报分析，禁止编造。"
        "请遵守以下硬性要求："
        "1) 输出语言必须为简体中文；"
        "2) 避免空话和套话，直接给结论与依据；"
        "3) 必须体现观点中的共识与分歧；"
        "4) 若交易观察信息不足，trading_observations 必须输出 null，不得强行下结论。"
        "请严格遵守输入中的 output_schema 与 section_constraints，"
        "输出必须是 JSON 对象，不要输出 markdown，不要输出额外解释。输入数据：\n"
        + json.dumps(prompt_input, ensure_ascii=False)
        + "\n\n"
        + _author_summary_field_guide_text()
    )

    payload = {
        "model": settings.openai_model,
        "temperature": 0.2,
        "max_tokens": max(256, min(2000, int(settings.openai_max_output_tokens))),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if settings.openrouter_site_url.strip():
        headers["HTTP-Referer"] = settings.openrouter_site_url.strip()
    if settings.openrouter_app_name.strip():
        headers["X-Title"] = settings.openrouter_app_name.strip()

    try:
        async with httpx.AsyncClient(timeout=settings.openai_timeout_seconds) as client:
            response = await client.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            return {
                "market_overview": "",
                "market_signals": "",
                "focus_points": [],
                "key_news": [],
                "trading_observations": None,
            }, "failed", response.text[:500]

        body = response.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        content_text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content_text, str) or not content_text.strip():
            raise RuntimeError("empty ai content")

        parsed = json.loads(content_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("ai response is not object")

        result = {
            "market_overview": str(parsed.get("market_overview") or "").strip(),
            "market_signals": str(parsed.get("market_signals") or "").strip(),
            "focus_points": [str(x).strip() for x in (parsed.get("focus_points") or []) if str(x).strip()],
            "key_news": [str(x).strip() for x in (parsed.get("key_news") or []) if str(x).strip()],
            "trading_observations": (
                str(parsed.get("trading_observations")).strip()
                if isinstance(parsed.get("trading_observations"), str) and str(parsed.get("trading_observations")).strip()
                else None
            ),
        }
        return result, "generated", None
    except Exception as exc:
        return {
            "market_overview": "",
            "market_signals": "",
            "focus_points": [],
            "key_news": [],
            "trading_observations": None,
        }, "failed", f"{type(exc).__name__}: {exc}"[:500]


async def generate_daily_digest(
    db: AsyncSession,
    *,
    digest_date: date,
    to_ts: datetime | None = None,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> DailyDigestRead:
    generated_at = _ensure_utc(to_ts) if to_ts is not None else datetime.now(UTC)
    window_start = datetime.combine(digest_date - timedelta(days=1), time.min, tzinfo=UTC)
    window_end = datetime.combine(digest_date + timedelta(days=1), time.min, tzinfo=UTC)

    profile, _weights_map, enabled_kol_ids, _markets = await load_profile_rules(db, profile_id=profile_id)

    extractions_result = await db.execute(
        select(PostExtraction).order_by(PostExtraction.created_at.desc(), PostExtraction.id.desc())
    )
    latest_by_raw_post_id: dict[int, PostExtraction] = {}
    for extraction in extractions_result.scalars().all():
        if extraction.raw_post_id not in latest_by_raw_post_id:
            latest_by_raw_post_id[extraction.raw_post_id] = extraction

    raw_posts_result = await db.execute(select(RawPost).order_by(RawPost.id.asc()))
    raw_post_map = {item.id: item for item in raw_posts_result.scalars().all()}

    kols_result = await db.execute(select(Kol).order_by(Kol.id.asc()))
    kol_map = {item.id: item for item in kols_result.scalars().all()}

    post_summaries: list[DailyDigestPostSummaryRead] = []
    for extraction in latest_by_raw_post_id.values():
        payload = extraction.extracted_json if isinstance(extraction.extracted_json, dict) else {}
        if _coerce_hasview(payload.get("hasview")) != 1:
            continue

        raw_post = raw_post_map.get(extraction.raw_post_id)
        if raw_post is None:
            continue

        if enabled_kol_ids is not None and raw_post.kol_id is not None and raw_post.kol_id not in enabled_kol_ids:
            continue

        business_ts, time_field_used = _resolve_post_business_ts(extraction, raw_post)
        if business_ts is None:
            continue
        if business_ts < window_start or business_ts >= window_end:
            continue

        kol = kol_map.get(raw_post.kol_id) if raw_post.kol_id is not None else None
        source_url = raw_post.url.strip() if isinstance(raw_post.url, str) and raw_post.url.strip() else ""
        if not source_url and isinstance(extraction.extracted_json, dict):
            source = extraction.extracted_json.get("source_url")
            if isinstance(source, str) and source.strip():
                source_url = source.strip()

        post_summaries.append(
            DailyDigestPostSummaryRead(
                raw_post_id=raw_post.id,
                extraction_id=extraction.id,
                kol_id=raw_post.kol_id,
                business_ts=business_ts,
                time_field_used=time_field_used,
                posted_at=_ensure_utc(raw_post.posted_at) if raw_post.posted_at else None,
                author_handle=raw_post.author_handle,
                author_display_name=kol.display_name if kol is not None else None,
                title=_extract_title(raw_post),
                source_url=source_url,
                summary=_extract_summary_text(extraction, raw_post),
            )
        )

    post_summaries.sort(key=lambda item: (item.business_ts, item.raw_post_id, item.extraction_id))

    by_author: dict[str, list[DailyDigestPostSummaryRead]] = defaultdict(list)
    for item in post_summaries:
        key = item.author_handle.strip() if item.author_handle.strip() else "unknown"
        by_author[key].append(item)
    extraction_by_id = {item.id: item for item in latest_by_raw_post_id.values()}

    ai_input_by_author: list[DailyDigestAuthorSummaryInputRead] = []
    ai_author_payload: list[dict[str, Any]] = []
    for author_handle in sorted(by_author.keys()):
        rows = by_author[author_handle]
        display_name = next((row.author_display_name for row in rows if row.author_display_name), None)
        summaries: list[dict[str, Any]] = []
        for row in rows:
            summaries.extend(
                _build_author_summary_rows(
                    row=row,
                    extraction=extraction_by_id.get(row.extraction_id),
                )
            )
        ai_input_by_author.append(
            DailyDigestAuthorSummaryInputRead(
                author_handle=author_handle,
                author_display_name=display_name,
                post_count=len(rows),
                summaries=summaries,
            )
        )
        ai_author_payload.append(
            {
                "author_handle": author_handle,
                "author_display_name": display_name,
                "post_count": len(rows),
                "summaries": summaries,
            }
        )

    ai_analysis, ai_status, ai_error = await _generate_ai_analysis(
        digest_date=digest_date,
        by_author_payload=ai_author_payload,
    )

    content = {
        "post_summaries": [item.model_dump(mode="json") for item in post_summaries],
        "ai_input_by_author": [item.model_dump(mode="json") for item in ai_input_by_author],
        "ai_analysis": ai_analysis,
        "metadata": {
            "generated_at": generated_at,
            "window_start": window_start,
            "window_end": window_end,
            "source_post_count": len(post_summaries),
            "ai_status": ai_status,
            "ai_error": ai_error,
        },
    }
    content = _to_json_safe(content)

    existing_result = await db.execute(
        select(DailyDigest)
        .where(DailyDigest.profile_id == profile.id)
        .where(DailyDigest.digest_date == digest_date)
        .order_by(DailyDigest.id.asc())
    )
    existing_items = list(existing_result.scalars().all())
    digest = existing_items[0] if existing_items else None
    for item in existing_items[1:]:
        await db.delete(item)

    if digest is None:
        digest = DailyDigest(
            profile_id=profile.id,
            digest_date=digest_date,
            version=1,
            days=2,
            content=content,
            generated_at=generated_at,
        )
        db.add(digest)
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            retry_result = await db.execute(
                select(DailyDigest)
                .where(DailyDigest.profile_id == profile.id)
                .where(DailyDigest.digest_date == digest_date)
                .order_by(DailyDigest.id.asc())
            )
            retry_items = list(retry_result.scalars().all())
            if not retry_items:
                raise
            digest = retry_items[0]
            for item in retry_items[1:]:
                await db.delete(item)
            digest.version = 1
            digest.days = 2
            digest.content = content
            digest.generated_at = generated_at
            await db.commit()
    else:
        digest.version = 1
        digest.days = 2
        digest.content = content
        digest.generated_at = generated_at
        await db.commit()

    await db.refresh(digest)
    return _build_digest_read(digest=digest)


async def get_daily_digest_by_date(
    db: AsyncSession,
    *,
    digest_date: date,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> DailyDigestRead:
    await load_profile_rules(db, profile_id=profile_id)

    result = await db.execute(
        select(DailyDigest)
        .where(DailyDigest.profile_id == profile_id)
        .where(DailyDigest.digest_date == digest_date)
        .order_by(DailyDigest.generated_at.desc(), DailyDigest.id.desc())
    )
    items = list(result.scalars().all())
    if not items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")
    return _build_digest_read(digest=items[0])


async def get_daily_digest_by_id(db: AsyncSession, *, digest_id: int) -> DailyDigestRead:
    digest = await db.get(DailyDigest, digest_id)
    if digest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")
    return _build_digest_read(digest=digest)


async def list_daily_digest_dates(db: AsyncSession, *, profile_id: int = DEFAULT_PROFILE_ID) -> list[date]:
    await load_profile_rules(db, profile_id=profile_id)

    result = await db.execute(
        select(DailyDigest)
        .where(DailyDigest.profile_id == profile_id)
        .order_by(DailyDigest.digest_date.desc(), DailyDigest.id.desc())
    )
    items = list(result.scalars().all())
    return sorted({item.digest_date for item in items}, reverse=True)
