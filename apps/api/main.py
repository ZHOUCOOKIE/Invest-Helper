from collections import defaultdict
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from enums import HORIZON_ORDER, Stance
from enums import ExtractionStatus
from models import Asset, Kol, KolView, PostExtraction, RawPost
from schemas import (
    AssetCreate,
    AssetViewsGroupRead,
    AssetViewsMetaRead,
    AssetRead,
    AssetViewsRead,
    KolCreate,
    KolRead,
    KolViewCreate,
    KolViewRead,
    PostExtractionRead,
    RawPostCreate,
    RawPostRead,
)
from services.extraction import DummyExtractor

app = FastAPI(title="InvestPulse API")

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
        key = (view.kol_id, view.asset_id, str(view.horizon.value))
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
        grouped[view.horizon.value][view.stance].append(KolViewRead.model_validate(view))

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


@app.post("/raw-posts/{raw_post_id}/extract", response_model=PostExtractionRead, status_code=status.HTTP_201_CREATED)
async def extract_raw_post(raw_post_id: int, db: AsyncSession = Depends(get_db)):
    raw_post = await db.get(RawPost, raw_post_id)
    if raw_post is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="raw post not found")

    extractor = DummyExtractor()
    extraction = PostExtraction(
        raw_post_id=raw_post.id,
        status=ExtractionStatus.pending,
        extracted_json=extractor.extract(raw_post),
        model_name=extractor.model_name,
    )
    db.add(extraction)
    await db.commit()
    await db.refresh(extraction)
    return extraction
