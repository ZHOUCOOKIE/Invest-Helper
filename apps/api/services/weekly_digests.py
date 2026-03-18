from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
import json
from typing import Any, Literal

from fastapi import HTTPException, status
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Kol, PostExtraction, RawPost, WeeklyDigest
from schemas import (
    DailyDigestAIAnalysisRead,
    DailyDigestMetadataRead,
    DailyDigestPostSummaryRead,
    WeeklyDigestDaySummaryInputRead,
    WeeklyDigestRead,
)
from services.profiles import DEFAULT_PROFILE_ID, load_profile_rules
from settings import get_settings

TimeFieldUsed = Literal["as_of", "posted_at", "created_at"]
WeeklyDigestKind = Literal["recent_week", "this_week", "last_week"]


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


def _build_day_summary_rows(
    *,
    row: DailyDigestPostSummaryRead,
    extraction: PostExtraction | None,
) -> list[dict[str, Any]]:
    base_row = {
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


def _coerce_day_inputs(value: Any) -> list[WeeklyDigestDaySummaryInputRead]:
    if not isinstance(value, list):
        return []
    rows: list[WeeklyDigestDaySummaryInputRead] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            rows.append(WeeklyDigestDaySummaryInputRead.model_validate(item))
        except Exception:
            continue
    return rows


def _week_start_sunday(target: date) -> date:
    days_since_sunday = (target.weekday() + 1) % 7
    return target - timedelta(days=days_since_sunday)


def _resolve_weekly_window(
    *,
    report_kind: WeeklyDigestKind,
    reference_date: date,
) -> tuple[date, date, date]:
    if report_kind == "recent_week":
        end_date = reference_date
        start_date = reference_date - timedelta(days=6)
        anchor_date = reference_date
        return start_date, end_date, anchor_date
    this_week_start = _week_start_sunday(reference_date)
    if report_kind == "this_week":
        return this_week_start, reference_date, this_week_start
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = this_week_start - timedelta(days=1)
    return last_week_start, last_week_end, last_week_start


def _expected_anchor_date(
    *,
    report_kind: WeeklyDigestKind,
    today: date | None = None,
) -> date:
    reference = today or datetime.now(UTC).date()
    _start_date, _end_date, anchor_date = _resolve_weekly_window(report_kind=report_kind, reference_date=reference)
    return anchor_date


async def _purge_stale_weekly_digests(
    db: AsyncSession,
    *,
    profile_id: int,
    report_kind: WeeklyDigestKind,
    today: date | None = None,
    commit: bool = False,
) -> int:
    expected_anchor = _expected_anchor_date(report_kind=report_kind, today=today)
    result = await db.execute(
        select(WeeklyDigest)
        .where(WeeklyDigest.profile_id == profile_id)
        .where(WeeklyDigest.report_kind == report_kind)
        .order_by(WeeklyDigest.anchor_date.desc(), WeeklyDigest.id.desc())
    )
    items = list(result.scalars().all())
    if not any(item.anchor_date == expected_anchor for item in items):
        return 0
    to_delete = [item for item in items if item.anchor_date != expected_anchor]
    for item in to_delete:
        await db.delete(item)
    if to_delete:
        if commit:
            await db.commit()
        else:
            await db.flush()
    return len(to_delete)


async def _generate_weekly_ai_analysis(
    *,
    report_kind: WeeklyDigestKind,
    anchor_date: date,
    day_payload: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str | None]:
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
        "weekly_report_kind": report_kind,
        "anchor_date": anchor_date.isoformat(),
        "sections": ["市场概览", "行情提示", "关注重点", "要闻提炼", "交易观察"],
        "day_summaries": day_payload,
        "requirements": {
            "output_schema": {
                "market_overview": "string",
                "market_signals": "string",
                "focus_points": "string[]",
                "key_news": "string[]",
                "trading_observations": "string|null",
            },
            "section_constraints": {
                "market_overview": "重点概括帖子所反映的日内及未来一周到一个月市场情绪走向、主线资产或板块、主要驱动因素和整体节奏。若有3个月及以上的中长期线索，只在最后简短概括。",
                "market_signals": "优先指出会影响日内及未来一周到一个月走势的方向、风险偏好变化、强弱切换和关键信号。3个月及以上的中长期判断只在段末简短补充，不要展开。",
                "focus_points": "列出接下来日内、未来一周到一个月最需要盯的事件、板块、资产或宏观变量，按时间优先级组织。若存在3个月及以上主线，只放在最后做简要提醒。",
                "key_news": "提炼最重要的新闻或事件，并说明为什么它会影响日内及未来一周到一个月的走势（影响路径或传导逻辑）。若有3个月及以上影响，只在最后简述。",
                "trading_observations": "如果帖子中存在博主短期交易操作、短期持仓/建仓/减仓建议、或强烈的短线观点，必须按人分类总结：每位博主分别写其短期操作观点、目标资产和主要依据；没有足够信息时必须输出 null，不得强行下结论。",
            },
        },
    }

    prompt = (
        "你是买方交易研究助手。只基于输入内容生成中文周报分析，禁止编造。"
        "请遵守以下硬性要求："
        "1) 输出语言必须为简体中文；"
        "2) 避免空话和套话，直接给结论与依据；"
        "3) 周报重点必须落在帖子对日内及未来一周到一个月走势的影响，3个月及以上的内容只能在各段最后简短概括，不得喧宾夺主；"
        "4) 必须清晰写出观点中的共识与分歧，并严格按以下标准执行：只有当至少2个不同博主对同一资产、同一板块或同一交易方向给出相同或明显一致的方向性判断时，才可写为“共识”；只有当不同博主对同一资产、同一板块或同一交易方向给出明显相反或不一致的方向性判断时，才可写为“分歧”。若不足以构成共识，不要强行写成共识；写分歧时必须明确指出涉及的博主、对应资产或方向，以及判断差异。"
        "5) 若交易观察中存在明确短线操作、短期持仓或建仓建议，trading_observations 必须按博主/作者分类总结，不要混成一段泛化结论；"
        "6) 若交易观察信息不足，trading_observations 必须输出 null，不得强行下结论。"
        "请严格遵守输入中的 output_schema 与 section_constraints，"
        "输出必须是 JSON 对象，不要输出 markdown，不要输出额外解释。输入数据：\n"
        + json.dumps(prompt_input, ensure_ascii=False)
    )

    payload = {
        "model": settings.openai_model,
        "temperature": 0.2,
        "max_tokens": max(256, min(2000, int(settings.openai_max_output_tokens))),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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


async def _build_weekly_digest_content(
    db: AsyncSession,
    *,
    report_kind: WeeklyDigestKind,
    anchor_date: date,
    window_start: datetime,
    window_end: datetime,
    enabled_kol_ids: set[int] | None,
    generated_at: datetime,
) -> dict[str, Any]:
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

    by_day: dict[str, list[DailyDigestPostSummaryRead]] = defaultdict(list)
    for item in post_summaries:
        by_day[item.business_ts.date().isoformat()].append(item)
    extraction_by_id = {item.id: item for item in latest_by_raw_post_id.values()}

    ai_input_by_day: list[WeeklyDigestDaySummaryInputRead] = []
    ai_day_payload: list[dict[str, Any]] = []
    for day_key in sorted(by_day.keys()):
        rows = by_day[day_key]
        summaries: list[dict[str, Any]] = []
        for row in rows:
            summaries.extend(
                _build_day_summary_rows(
                    row=row,
                    extraction=extraction_by_id.get(row.extraction_id),
                )
            )
        ai_input_by_day.append(
            WeeklyDigestDaySummaryInputRead(
                date=date.fromisoformat(day_key),
                post_count=len(rows),
                summaries=summaries,
            )
        )
        ai_day_payload.append(
            {
                "date": day_key,
                "post_count": len(rows),
                "summaries": summaries,
            }
        )

    ai_analysis, ai_status, ai_error = await _generate_weekly_ai_analysis(
        report_kind=report_kind,
        anchor_date=anchor_date,
        day_payload=ai_day_payload,
    )
    return _to_json_safe(
        {
            "post_summaries": [item.model_dump(mode="json") for item in post_summaries],
            "ai_input_by_day": [item.model_dump(mode="json") for item in ai_input_by_day],
            "ai_analysis": ai_analysis,
            "metadata": {
                "generated_at": generated_at,
                "window_start": window_start,
                "window_end": window_end,
                "source_post_count": len(post_summaries),
                "ai_status": ai_status,
                "ai_error": ai_error,
                "report_kind": report_kind,
            },
        }
    )


def _build_weekly_digest_read(*, digest: WeeklyDigest) -> WeeklyDigestRead:
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
    report_kind_raw = str(digest.report_kind or "recent_week")
    report_kind: WeeklyDigestKind = (
        report_kind_raw if report_kind_raw in {"recent_week", "this_week", "last_week"} else "recent_week"
    )
    return WeeklyDigestRead(
        id=digest.id,
        profile_id=digest.profile_id,
        report_kind=report_kind,
        anchor_date=digest.anchor_date,
        generated_at=digest.generated_at,
        post_summaries=_coerce_post_summaries(content.get("post_summaries")),
        ai_input_by_day=_coerce_day_inputs(content.get("ai_input_by_day")),
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


async def generate_weekly_digest(
    db: AsyncSession,
    *,
    report_kind: WeeklyDigestKind,
    reference_date: date | None = None,
    to_ts: datetime | None = None,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> WeeklyDigestRead:
    today = reference_date or datetime.now(UTC).date()
    start_date, end_date, anchor_date = _resolve_weekly_window(report_kind=report_kind, reference_date=today)
    generated_at = _ensure_utc(to_ts) if to_ts is not None else datetime.now(UTC)
    window_start = datetime.combine(start_date, time.min, tzinfo=UTC)
    window_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    profile, _weights_map, enabled_kol_ids, _markets = await load_profile_rules(db, profile_id=profile_id)
    await _purge_stale_weekly_digests(
        db,
        profile_id=profile.id,
        report_kind=report_kind,
        today=today,
        commit=False,
    )
    content = await _build_weekly_digest_content(
        db,
        report_kind=report_kind,
        anchor_date=anchor_date,
        window_start=window_start,
        window_end=window_end,
        enabled_kol_ids=enabled_kol_ids,
        generated_at=generated_at,
    )

    existing_result = await db.execute(
        select(WeeklyDigest)
        .where(WeeklyDigest.profile_id == profile.id)
        .where(WeeklyDigest.report_kind == report_kind)
        .where(WeeklyDigest.anchor_date == anchor_date)
        .order_by(WeeklyDigest.id.asc())
    )
    existing_items = list(existing_result.scalars().all())
    digest = existing_items[0] if existing_items else None
    for item in existing_items[1:]:
        await db.delete(item)
    if digest is None:
        digest = WeeklyDigest(
            profile_id=profile.id,
            report_kind=report_kind,
            anchor_date=anchor_date,
            version=1,
            days=(end_date - start_date).days + 1,
            content=content,
            generated_at=generated_at,
        )
        db.add(digest)
    else:
        digest.version = 1
        digest.days = (end_date - start_date).days + 1
        digest.content = content
        digest.generated_at = generated_at
    await db.commit()
    await db.refresh(digest)
    return _build_weekly_digest_read(digest=digest)


async def get_weekly_digest(
    db: AsyncSession,
    *,
    report_kind: WeeklyDigestKind,
    anchor_date: date,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> WeeklyDigestRead:
    await load_profile_rules(db, profile_id=profile_id)
    await _purge_stale_weekly_digests(
        db,
        profile_id=profile_id,
        report_kind=report_kind,
        commit=True,
    )
    result = await db.execute(
        select(WeeklyDigest)
        .where(WeeklyDigest.profile_id == profile_id)
        .where(WeeklyDigest.report_kind == report_kind)
        .where(WeeklyDigest.anchor_date == anchor_date)
        .order_by(WeeklyDigest.generated_at.desc(), WeeklyDigest.id.desc())
    )
    items = list(result.scalars().all())
    if not items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="weekly digest not found")
    return _build_weekly_digest_read(digest=items[0])


async def list_weekly_digest_anchor_dates(
    db: AsyncSession,
    *,
    report_kind: WeeklyDigestKind,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> list[date]:
    await load_profile_rules(db, profile_id=profile_id)
    await _purge_stale_weekly_digests(
        db,
        profile_id=profile_id,
        report_kind=report_kind,
        commit=True,
    )
    result = await db.execute(
        select(WeeklyDigest)
        .where(WeeklyDigest.profile_id == profile_id)
        .where(WeeklyDigest.report_kind == report_kind)
        .order_by(WeeklyDigest.anchor_date.desc(), WeeklyDigest.id.desc())
    )
    items = list(result.scalars().all())
    return sorted({item.anchor_date for item in items}, reverse=True)
