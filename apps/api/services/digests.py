from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from enums import HORIZON_ORDER, Stance
from models import Asset, DailyDigest, Kol, KolView, RawPost
from schemas import (
    DailyDigestAssetSummaryRead,
    DailyDigestHorizonCountRead,
    DailyDigestMetadataRead,
    DailyDigestRead,
    DailyDigestTopAssetRead,
    DailyDigestTopViewRead,
)
from services.profiles import DEFAULT_PROFILE_ID, load_profile_rules

TimeFieldUsed = Literal["as_of", "posted_at", "created_at"]


def _calc_clarity(bull_count: int, bear_count: int) -> float:
    total = bull_count + bear_count
    return abs(bull_count - bear_count) / max(1, total)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalize_market(value: str | None) -> str:
    return value.strip().upper() if isinstance(value, str) else ""


def _business_ts_from_as_of(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def _choose_time_field(used_fields: set[str]) -> TimeFieldUsed:
    if "as_of" in used_fields:
        return "as_of"
    if "posted_at" in used_fields:
        return "posted_at"
    return "created_at"


def _resolve_view_business_ts(
    view: KolView,
    *,
    posted_at_by_url: dict[str, datetime],
) -> tuple[datetime | None, TimeFieldUsed]:
    if view.as_of is not None:
        return _business_ts_from_as_of(view.as_of), "as_of"

    source_url = (view.source_url or "").strip()
    posted_at = posted_at_by_url.get(source_url)
    if posted_at is not None:
        return _ensure_utc(posted_at), "posted_at"

    if view.created_at is not None:
        return _ensure_utc(view.created_at), "created_at"

    return None, "created_at"


def _is_newer_view(candidate: KolView, current: KolView) -> bool:
    return (candidate.as_of, candidate.created_at, candidate.id) > (
        current.as_of,
        current.created_at,
        current.id,
    )


def _select_latest_views(views: list[KolView]) -> list[KolView]:
    latest_by_key: dict[tuple[int, int, str], KolView] = {}
    for view in views:
        horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
        key = (view.kol_id, view.asset_id, horizon_value)
        prev = latest_by_key.get(key)
        if prev is None or _is_newer_view(view, prev):
            latest_by_key[key] = view
    return list(latest_by_key.values())


async def _load_posted_at_by_url(db: AsyncSession) -> dict[str, datetime]:
    result = await db.execute(select(RawPost).order_by(RawPost.posted_at.desc(), RawPost.id.desc()))
    mapping: dict[str, datetime] = {}
    for post in result.scalars().all():
        url = (post.url or "").strip()
        if not url or url in mapping:
            continue
        mapping[url] = post.posted_at
    return mapping


def _build_daily_digest_read_from_content(
    *,
    digest_id: int,
    profile_id: int,
    digest_date: date,
    version: int,
    days: int,
    generated_at: datetime,
    content: dict[str, Any],
) -> DailyDigestRead:
    metadata = content.get("metadata") if isinstance(content.get("metadata"), dict) else {}
    top_assets = content.get("top_assets") if isinstance(content.get("top_assets"), list) else []
    per_asset_summary = (
        content.get("per_asset_summary") if isinstance(content.get("per_asset_summary"), list) else []
    )

    summary_window_start = metadata.get("summary_window_start", generated_at)
    summary_window_end = metadata.get("summary_window_end", generated_at)
    generated_from_ts = metadata.get("generated_from_ts", summary_window_start)
    generated_to_ts = metadata.get("generated_to_ts", summary_window_end)

    used_field = metadata.get("time_field_used")
    time_field_used: TimeFieldUsed = used_field if used_field in {"as_of", "posted_at", "created_at"} else "as_of"

    return DailyDigestRead(
        id=digest_id,
        profile_id=profile_id,
        digest_date=digest_date,
        version=version,
        generated_at=generated_at,
        top_assets=top_assets,
        per_asset_summary=per_asset_summary,
        metadata=DailyDigestMetadataRead(
            generated_at=metadata.get("generated_at", generated_at),
            days=int(metadata.get("days", days)),
            summary_window_start=summary_window_start,
            summary_window_end=summary_window_end,
            generated_from_ts=generated_from_ts,
            generated_to_ts=generated_to_ts,
            time_field_used=time_field_used,
        ),
    )


def _build_daily_digest_content(
    *,
    days: int,
    generated_from_ts: datetime,
    generated_to_ts: datetime,
    assets: list[Asset],
    kols: list[Kol],
    views: list[KolView],
    posted_at_by_url: dict[str, datetime],
    weights_map: dict[int, float],
    enabled_kol_ids: set[int] | None,
    markets: set[str] | None,
    top_n_per_stance: int = 3,
) -> dict[str, Any]:
    asset_map = {item.id: item for item in assets}
    kol_map = {item.id: item for item in kols}

    used_fields: set[str] = set()
    filtered_views_by_time: list[KolView] = []
    for view in views:
        business_ts, field_used = _resolve_view_business_ts(view, posted_at_by_url=posted_at_by_url)
        if business_ts is None:
            continue
        if business_ts < generated_from_ts or business_ts > generated_to_ts:
            continue
        used_fields.add(field_used)
        filtered_views_by_time.append(view)

    profile_filtered: list[KolView] = []
    for view in filtered_views_by_time:
        if enabled_kol_ids is not None and view.kol_id not in enabled_kol_ids:
            continue
        asset = asset_map.get(view.asset_id)
        if markets is not None:
            market_value = _normalize_market(asset.market if asset is not None else None)
            if market_value not in markets:
                continue
        profile_filtered.append(view)

    latest_views = _select_latest_views(profile_filtered)

    views_cutoff_24h = generated_to_ts - timedelta(hours=24)
    views_cutoff_7d = generated_to_ts - timedelta(days=7)

    asset_counts_24h: dict[int, int] = defaultdict(int)
    asset_counts_7d: dict[int, int] = defaultdict(int)
    asset_weighted_24h: dict[int, float] = defaultdict(float)
    asset_weighted_7d: dict[int, float] = defaultdict(float)
    for view in latest_views:
        business_ts, _ = _resolve_view_business_ts(view, posted_at_by_url=posted_at_by_url)
        if business_ts is None:
            continue
        kol_weight = float(weights_map.get(view.kol_id, 1.0))
        if business_ts >= views_cutoff_24h:
            asset_counts_24h[view.asset_id] += 1
            asset_weighted_24h[view.asset_id] += kol_weight
        if business_ts >= views_cutoff_7d:
            asset_counts_7d[view.asset_id] += 1
            asset_weighted_7d[view.asset_id] += kol_weight

    top_assets: list[DailyDigestTopAssetRead] = []
    top_asset_ids = sorted(
        {asset_id for asset_id in set(asset_counts_24h) | set(asset_counts_7d)},
        key=lambda asset_id: (
            -asset_weighted_24h.get(asset_id, 0.0),
            -asset_weighted_7d.get(asset_id, 0.0),
            -asset_counts_24h.get(asset_id, 0),
            -asset_counts_7d.get(asset_id, 0),
            asset_map[asset_id].symbol if asset_id in asset_map else str(asset_id),
        ),
    )
    for asset_id in top_asset_ids:
        asset = asset_map.get(asset_id)
        if asset is None:
            continue
        top_assets.append(
            DailyDigestTopAssetRead(
                asset_id=asset.id,
                symbol=asset.symbol,
                name=asset.name,
                market=asset.market,
                new_views_24h=asset_counts_24h.get(asset.id, 0),
                new_views_7d=asset_counts_7d.get(asset.id, 0),
                weighted_views_24h=round(asset_weighted_24h.get(asset.id, 0.0), 4),
                weighted_views_7d=round(asset_weighted_7d.get(asset.id, 0.0), 4),
            )
        )

    per_asset_views: dict[int, list[KolView]] = defaultdict(list)
    for view in latest_views:
        per_asset_views[view.asset_id].append(view)

    stance_order = {Stance.bull.value: 0, Stance.bear.value: 1, Stance.neutral.value: 2}
    horizon_order = {value: idx for idx, value in enumerate(HORIZON_ORDER)}
    summary_asset_ids = sorted(
        per_asset_views.keys(),
        key=lambda asset_id: (
            -asset_weighted_24h.get(asset_id, 0.0),
            -asset_weighted_7d.get(asset_id, 0.0),
            -asset_counts_24h.get(asset_id, 0),
            -asset_counts_7d.get(asset_id, 0),
            asset_map[asset_id].symbol if asset_id in asset_map else str(asset_id),
        ),
    )

    per_asset_summary: list[DailyDigestAssetSummaryRead] = []
    for asset_id in summary_asset_ids:
        asset = asset_map.get(asset_id)
        if asset is None:
            continue
        asset_views = per_asset_views.get(asset_id, [])
        grouped_counts: dict[str, dict[Stance, int]] = defaultdict(
            lambda: {Stance.bull: 0, Stance.bear: 0, Stance.neutral: 0}
        )
        for view in asset_views:
            horizon_value = view.horizon.value if hasattr(view.horizon, "value") else str(view.horizon)
            stance_value = view.stance.value if hasattr(view.stance, "value") else str(view.stance)
            stance = Stance(stance_value) if stance_value in {"bull", "bear", "neutral"} else Stance.neutral
            grouped_counts[horizon_value][stance] += 1

        horizon_counts: list[DailyDigestHorizonCountRead] = []
        for horizon in sorted(grouped_counts.keys(), key=lambda value: horizon_order.get(value, 999)):
            counts = grouped_counts[horizon]
            horizon_counts.append(
                DailyDigestHorizonCountRead(
                    horizon=horizon,
                    bull_count=counts[Stance.bull],
                    bear_count=counts[Stance.bear],
                    neutral_count=counts[Stance.neutral],
                )
            )

        bull_total = sum(item.bull_count for item in horizon_counts)
        bear_total = sum(item.bear_count for item in horizon_counts)

        top_views_by_stance: dict[str, list[DailyDigestTopViewRead]] = {
            Stance.bull.value: [],
            Stance.bear.value: [],
            Stance.neutral.value: [],
        }
        sorted_views = sorted(
            asset_views,
            key=lambda item: (
                stance_order.get(item.stance.value if hasattr(item.stance, "value") else str(item.stance), 999),
                -(float(weights_map.get(item.kol_id, 1.0)) * float(item.confidence or 0)),
                -(item.confidence or 0),
                -(item.id or 0),
            ),
        )
        for view in sorted_views:
            stance_value = view.stance.value if hasattr(view.stance, "value") else str(view.stance)
            if stance_value not in top_views_by_stance:
                continue
            if len(top_views_by_stance[stance_value]) >= top_n_per_stance:
                continue
            kol = kol_map.get(view.kol_id)
            kol_weight = float(weights_map.get(view.kol_id, 1.0))
            top_views_by_stance[stance_value].append(
                DailyDigestTopViewRead(
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
                    kol_weight=kol_weight,
                    weighted_score=round(kol_weight * float(view.confidence or 0), 4),
                )
            )

        per_asset_summary.append(
            DailyDigestAssetSummaryRead(
                asset_id=asset.id,
                symbol=asset.symbol,
                name=asset.name,
                market=asset.market,
                horizon_counts=horizon_counts,
                clarity=_calc_clarity(bull_count=bull_total, bear_count=bear_total),
                top_views_bull=top_views_by_stance[Stance.bull.value],
                top_views_bear=top_views_by_stance[Stance.bear.value],
                top_views_neutral=top_views_by_stance[Stance.neutral.value],
            )
        )

    metadata = DailyDigestMetadataRead(
        generated_at=generated_to_ts,
        days=days,
        summary_window_start=generated_from_ts,
        summary_window_end=generated_to_ts,
        generated_from_ts=generated_from_ts,
        generated_to_ts=generated_to_ts,
        time_field_used=_choose_time_field(used_fields or {"as_of"}),
    )
    return {
        "top_assets": [item.model_dump(mode="json") for item in top_assets],
        "per_asset_summary": [item.model_dump(mode="json") for item in per_asset_summary],
        "metadata": metadata.model_dump(mode="json"),
    }


async def generate_daily_digest(
    db: AsyncSession,
    *,
    digest_date: date,
    days: int,
    to_ts: datetime | None = None,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> DailyDigestRead:
    generated_at = datetime.now(UTC)
    generated_to_ts = _ensure_utc(to_ts) if to_ts is not None else generated_at
    generated_from_ts = generated_to_ts - timedelta(days=days)

    profile, weights_map, enabled_kol_ids, markets = await load_profile_rules(db, profile_id=profile_id)

    assets_result = await db.execute(select(Asset).order_by(Asset.id.asc()))
    assets = list(assets_result.scalars().all())

    kols_result = await db.execute(select(Kol).order_by(Kol.id.asc()))
    kols = list(kols_result.scalars().all())

    views_result = await db.execute(select(KolView).order_by(KolView.created_at.desc(), KolView.id.desc()))
    views = list(views_result.scalars().all())

    posted_at_by_url = await _load_posted_at_by_url(db)

    existing_result = await db.execute(
        select(DailyDigest).where(DailyDigest.profile_id == profile.id).where(DailyDigest.digest_date == digest_date)
    )
    existing_items = list(existing_result.scalars().all())
    next_version = max((item.version for item in existing_items), default=0) + 1

    content = _build_daily_digest_content(
        days=days,
        generated_from_ts=generated_from_ts,
        generated_to_ts=generated_to_ts,
        assets=assets,
        kols=kols,
        views=views,
        posted_at_by_url=posted_at_by_url,
        weights_map=weights_map,
        enabled_kol_ids=enabled_kol_ids,
        markets=markets,
    )
    digest = DailyDigest(
        profile_id=profile.id,
        digest_date=digest_date,
        version=next_version,
        days=days,
        content=content,
        generated_at=generated_at,
    )
    db.add(digest)
    await db.commit()
    await db.refresh(digest)
    return _build_daily_digest_read_from_content(
        digest_id=digest.id,
        profile_id=digest.profile_id,
        digest_date=digest.digest_date,
        version=digest.version,
        days=digest.days,
        generated_at=digest.generated_at,
        content=digest.content if isinstance(digest.content, dict) else {},
    )


async def get_daily_digest_by_date(
    db: AsyncSession,
    *,
    digest_date: date,
    version: int | None = None,
    profile_id: int = DEFAULT_PROFILE_ID,
) -> DailyDigestRead:
    await load_profile_rules(db, profile_id=profile_id)

    result = await db.execute(
        select(DailyDigest)
        .where(DailyDigest.profile_id == profile_id)
        .where(DailyDigest.digest_date == digest_date)
    )
    items = list(result.scalars().all())
    if not items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")

    if version is not None:
        digest = next((item for item in items if item.version == version), None)
        if digest is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")
    else:
        digest = max(items, key=lambda item: (item.version, item.id))

    return _build_daily_digest_read_from_content(
        digest_id=digest.id,
        profile_id=digest.profile_id,
        digest_date=digest.digest_date,
        version=digest.version,
        days=digest.days,
        generated_at=digest.generated_at,
        content=digest.content if isinstance(digest.content, dict) else {},
    )


async def get_daily_digest_by_id(db: AsyncSession, *, digest_id: int) -> DailyDigestRead:
    digest = await db.get(DailyDigest, digest_id)
    if digest is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="digest not found")

    return _build_daily_digest_read_from_content(
        digest_id=digest.id,
        profile_id=digest.profile_id,
        digest_date=digest.digest_date,
        version=digest.version,
        days=digest.days,
        generated_at=digest.generated_at,
        content=digest.content if isinstance(digest.content, dict) else {},
    )


async def list_daily_digest_dates(db: AsyncSession, *, profile_id: int = DEFAULT_PROFILE_ID) -> list[date]:
    await load_profile_rules(db, profile_id=profile_id)

    result = await db.execute(
        select(DailyDigest)
        .where(DailyDigest.profile_id == profile_id)
        .order_by(DailyDigest.digest_date.desc(), DailyDigest.version.desc())
    )
    items = list(result.scalars().all())
    return sorted({item.digest_date for item in items}, reverse=True)
