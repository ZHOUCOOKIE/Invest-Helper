from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from enums import ExtractionStatus, Horizon, Stance


class AssetCreate(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    name: str | None = Field(default=None, max_length=255)
    market: str | None = Field(default=None, max_length=32)


class AssetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    name: str | None
    market: str | None
    created_at: datetime


class AssetAliasCreate(BaseModel):
    alias: str = Field(min_length=1, max_length=255)


class AssetAliasRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    asset_id: int
    alias: str
    created_at: datetime


class KolCreate(BaseModel):
    platform: str = Field(min_length=1, max_length=32)
    handle: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=255)


class KolRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    platform: str
    handle: str
    display_name: str | None
    enabled: bool
    created_at: datetime


class KolViewCreate(BaseModel):
    kol_id: int
    asset_id: int
    stance: Stance
    horizon: Horizon
    confidence: int = Field(ge=0, le=100)
    summary: str | None = Field(default=None, max_length=1024)
    source_url: str | None = Field(default=None, max_length=1024)
    as_of: date | None = None


class KolViewRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kol_id: int
    asset_id: int
    stance: Stance
    horizon: Horizon
    confidence: int
    summary: str
    source_url: str
    as_of: date
    created_at: datetime


class AssetViewsGroupRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    horizon: Horizon
    bull: list[KolViewRead]
    bear: list[KolViewRead]
    neutral: list[KolViewRead]


class AssetViewsMetaRead(BaseModel):
    sort: str
    generated_at: datetime
    version_policy: str


class AssetViewsRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    asset_id: int
    groups: list[AssetViewsGroupRead]
    meta: AssetViewsMetaRead


class RawPostCreate(BaseModel):
    platform: str = Field(min_length=1, max_length=32)
    author_handle: str = Field(min_length=1, max_length=128)
    external_id: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=1024)
    content_text: str = Field(min_length=1, max_length=8192)
    posted_at: datetime
    raw_json: dict | None = None


class ManualIngestCreate(BaseModel):
    platform: str = Field(min_length=1, max_length=32)
    author_handle: str = Field(min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=1024)
    external_id: str | None = Field(default=None, max_length=128)
    content_text: str = Field(min_length=1, max_length=8192)
    posted_at: datetime | None = None
    raw_json: dict | None = None


class RawPostRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    platform: str
    author_handle: str
    external_id: str
    url: str
    content_text: str
    posted_at: datetime
    fetched_at: datetime
    raw_json: dict | None


class PostExtractionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: int
    raw_post_id: int
    status: ExtractionStatus
    extracted_json: dict
    model_name: str
    extractor_name: str
    prompt_version: str | None
    prompt_text: str | None
    prompt_hash: str | None
    raw_model_output: str | None
    parsed_model_output: dict | None
    model_latency_ms: int | None
    model_input_tokens: int | None
    model_output_tokens: int | None
    last_error: str | None
    reviewed_at: datetime | None
    reviewed_by: str | None
    review_note: str | None
    applied_kol_view_id: int | None
    auto_applied_count: int | None
    auto_policy: str | None
    auto_applied_kol_view_ids: list[int] | None
    auto_approve_confidence_threshold: int | None = None
    auto_approve_min_display_confidence: int | None = None
    approve_inserted_count: int | None = None
    approve_skipped_count: int | None = None
    auto_applied_asset_view_keys: list[str] | None = None
    auto_applied_views: list["AutoAppliedViewRead"] | None = None
    created_at: datetime


class AutoAppliedViewRead(BaseModel):
    kol_view_id: int
    symbol: str
    asset_id: int
    stance: Stance
    horizon: Horizon
    as_of: date
    confidence: int


class ExtractionApproveRequest(BaseModel):
    kol_id: int
    asset_id: int
    stance: Stance
    horizon: Horizon
    confidence: int = Field(ge=0, le=100)
    summary: str = Field(min_length=1, max_length=1024)
    source_url: str = Field(min_length=1, max_length=1024)
    as_of: date


class ExtractionApproveBatchViewRequest(BaseModel):
    asset_id: int
    stance: Stance
    horizon: Horizon
    confidence: int = Field(ge=0, le=100)
    summary: str = Field(min_length=1, max_length=1024)
    source_url: str = Field(min_length=1, max_length=1024)
    as_of: date


class ExtractionApproveBatchRequest(BaseModel):
    kol_id: int
    views: list[ExtractionApproveBatchViewRequest] = Field(min_length=1, max_length=50)


class ExtractionRejectRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=1024)


class PostExtractionWithRawPostRead(PostExtractionRead):
    raw_post: RawPostRead


class ManualIngestRead(BaseModel):
    raw_post: RawPostRead
    extraction: PostExtractionRead
    extraction_id: int


class DashboardPendingExtractionRead(BaseModel):
    id: int
    platform: str
    author_handle: str
    url: str
    posted_at: datetime
    created_at: datetime


class DashboardTopAssetRead(BaseModel):
    asset_id: int
    symbol: str
    market: str | None
    views_count_7d: int
    avg_confidence_7d: float


class DashboardClarityRead(BaseModel):
    horizon: Horizon
    bull_count: int
    bear_count: int
    neutral_count: int
    clarity: float


class DashboardExtractionStatsRead(BaseModel):
    window_hours: int
    extraction_count: int
    dummy_count: int
    openai_count: int
    error_count: int


class DashboardRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    pending_extractions_count: int
    latest_pending_extractions: list[DashboardPendingExtractionRead]
    top_assets: list[DashboardTopAssetRead]
    clarity: list[DashboardClarityRead]
    extraction_stats: DashboardExtractionStatsRead


class ExtractorStatusRead(BaseModel):
    mode: str
    has_api_key: bool
    default_model: str
    base_url: str
    call_budget_remaining: int | None
    max_output_tokens: int
