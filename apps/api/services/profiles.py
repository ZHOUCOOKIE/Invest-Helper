from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Kol, ProfileKolWeight, ProfileMarket, UserProfile
from schemas import (
    ProfileKolsUpdateRequest,
    ProfileKolWeightRead,
    ProfileMarketsUpdateRequest,
    ProfileRead,
    ProfileSummaryRead,
)

DEFAULT_PROFILE_ID = 1
DEFAULT_PROFILE_NAME = "default"


def _normalize_market(value: str) -> str:
    return value.strip().upper()


async def ensure_default_profile(db: AsyncSession) -> UserProfile:
    profile = await db.get(UserProfile, DEFAULT_PROFILE_ID)
    if profile is not None:
        return profile

    profile = UserProfile(id=DEFAULT_PROFILE_ID, name=DEFAULT_PROFILE_NAME)
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return profile


async def _build_profile_read(db: AsyncSession, profile: UserProfile) -> ProfileRead:
    kols_result = await db.execute(select(Kol).order_by(Kol.id.asc()))
    kols = list(kols_result.scalars().all())

    weight_result = await db.execute(
        select(ProfileKolWeight)
        .where(ProfileKolWeight.profile_id == profile.id)
        .order_by(ProfileKolWeight.kol_id.asc())
    )
    weights = list(weight_result.scalars().all())
    weight_map = {item.kol_id: item for item in weights}

    markets_result = await db.execute(
        select(ProfileMarket)
        .where(ProfileMarket.profile_id == profile.id)
        .order_by(ProfileMarket.market.asc())
    )
    markets = [item.market for item in markets_result.scalars().all()]

    kols_read: list[ProfileKolWeightRead] = []
    for kol in kols:
        weight_row = weight_map.get(kol.id)
        kols_read.append(
            ProfileKolWeightRead(
                kol_id=kol.id,
                weight=float(weight_row.weight) if weight_row is not None else 1.0,
                enabled=bool(weight_row.enabled) if weight_row is not None else True,
                kol_display_name=kol.display_name,
                kol_handle=kol.handle,
                kol_platform=kol.platform,
            )
        )

    return ProfileRead(
        id=profile.id,
        name=profile.name,
        created_at=profile.created_at,
        kols=kols_read,
        markets=markets,
    )


async def list_profiles(db: AsyncSession) -> list[ProfileSummaryRead]:
    await ensure_default_profile(db)
    result = await db.execute(select(UserProfile).order_by(UserProfile.id.asc()))
    return [ProfileSummaryRead.model_validate(item) for item in result.scalars().all()]


async def get_profile(db: AsyncSession, *, profile_id: int) -> ProfileRead:
    await ensure_default_profile(db)
    profile = await db.get(UserProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile not found")
    return await _build_profile_read(db, profile)


async def update_profile_kols(
    db: AsyncSession,
    *,
    profile_id: int,
    payload: ProfileKolsUpdateRequest,
) -> ProfileRead:
    await ensure_default_profile(db)
    profile = await db.get(UserProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile not found")

    kols_result = await db.execute(select(Kol).order_by(Kol.id.asc()))
    kol_ids = {item.id for item in kols_result.scalars().all()}

    existing_result = await db.execute(
        select(ProfileKolWeight).where(ProfileKolWeight.profile_id == profile_id)
    )
    existing = {item.kol_id: item for item in existing_result.scalars().all()}

    for item in payload.items:
        if item.kol_id not in kol_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"kol {item.kol_id} not found")
        row = existing.get(item.kol_id)
        if row is None:
            db.add(
                ProfileKolWeight(
                    profile_id=profile_id,
                    kol_id=item.kol_id,
                    weight=float(item.weight),
                    enabled=bool(item.enabled),
                )
            )
        else:
            row.weight = float(item.weight)
            row.enabled = bool(item.enabled)
            if hasattr(row, "created_at") and row.created_at is None:
                row.created_at = datetime.now(UTC)

    await db.commit()
    return await _build_profile_read(db, profile)


async def update_profile_markets(
    db: AsyncSession,
    *,
    profile_id: int,
    payload: ProfileMarketsUpdateRequest,
) -> ProfileRead:
    await ensure_default_profile(db)
    profile = await db.get(UserProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile not found")

    normalized_markets = sorted({_normalize_market(item) for item in payload.markets if _normalize_market(item)})

    existing_result = await db.execute(select(ProfileMarket).where(ProfileMarket.profile_id == profile_id))
    for row in existing_result.scalars().all():
        await db.delete(row)

    for market in normalized_markets:
        db.add(ProfileMarket(profile_id=profile_id, market=market))

    await db.commit()
    return await _build_profile_read(db, profile)


async def load_profile_rules(
    db: AsyncSession,
    *,
    profile_id: int,
) -> tuple[UserProfile, dict[int, float], set[int] | None, set[str] | None]:
    profile = await ensure_default_profile(db)
    if profile.id != profile_id:
        profile = await db.get(UserProfile, profile_id)
        if profile is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="profile not found")

    weights_result = await db.execute(
        select(ProfileKolWeight).where(ProfileKolWeight.profile_id == profile_id)
    )
    weights_rows = list(weights_result.scalars().all())
    weights_map: dict[int, float] = {item.kol_id: float(item.weight) for item in weights_rows if item.enabled}
    enabled_kol_ids = {item.kol_id for item in weights_rows if item.enabled}
    has_explicit_kols = len(weights_rows) > 0

    markets_result = await db.execute(select(ProfileMarket).where(ProfileMarket.profile_id == profile_id))
    markets_rows = list(markets_result.scalars().all())
    markets = {_normalize_market(item.market) for item in markets_rows if _normalize_market(item.market)}

    return (
        profile,
        weights_map,
        enabled_kol_ids if has_explicit_kols else None,
        markets if markets else None,
    )
