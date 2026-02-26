from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from enums import ExtractionStatus, Horizon, Stance

ExtractionListStatus = Literal["pending", "approved", "rejected", "all", ""]


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


class AssetAliasMapRead(BaseModel):
    alias: str
    symbol: str


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
    kol_id: int | None
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
    auto_reject_confidence_threshold: int | None = None
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


class XImportItemCreate(BaseModel):
    kol_id: int | None = Field(default=None, ge=1)
    external_id: str = Field(min_length=1, max_length=128)
    author_handle: str = Field(min_length=1, max_length=128)
    resolved_author_handle: str | None = None
    url: str = Field(min_length=1, max_length=1024)
    posted_at: datetime
    content_text: str = Field(min_length=1, max_length=8192)
    raw_json: dict | None = None


class XConvertErrorRead(BaseModel):
    row_index: int
    external_id: str | None = None
    url: str | None = None
    reason: str


class XSkippedNotFollowedRead(BaseModel):
    row_index: int
    author_handle: str | None = None
    external_id: str | None = None
    reason: str


class XConvertResponseRead(BaseModel):
    converted_rows: int
    converted_ok: int
    converted_failed: int
    errors: list[XConvertErrorRead]
    items: list[XImportItemCreate]
    handles_summary: list["XHandleSummaryRead"] = Field(default_factory=list)
    resolved_author_handle: str | None = None
    resolved_kol_id: int | None = None
    kol_created: bool = False
    skipped_not_followed_count: int = 0
    skipped_not_followed_samples: list[XSkippedNotFollowedRead] = Field(default_factory=list)


class XHandleSummaryRead(BaseModel):
    author_handle: str
    count: int
    earliest_posted_at: datetime | None = None
    latest_posted_at: datetime | None = None
    will_create_kol: bool = False


class XImportedByHandleRead(BaseModel):
    received: int = 0
    inserted: int = 0
    dedup: int = 0
    warnings: int = 0
    raw_post_ids: list[int] = Field(default_factory=list)
    extract_success: int = 0
    extract_failed: int = 0
    skipped_already_extracted: int = 0


class XCreatedKolRead(BaseModel):
    id: int
    handle: str
    name: str | None = None


class XImportStatsRead(BaseModel):
    received_count: int
    inserted_raw_posts_count: int
    inserted_raw_post_ids: list[int]
    dedup_existing_raw_post_ids: list[int]
    dedup_skipped_count: int
    extract_success_count: int
    extract_failed_count: int
    skipped_already_extracted_count: int
    warnings_count: int
    warnings: list[str]
    imported_by_handle: dict[str, XImportedByHandleRead] = Field(default_factory=dict)
    created_kols: list[XCreatedKolRead] = Field(default_factory=list)
    resolved_author_handle: str | None = None
    resolved_kol_id: int | None = None
    kol_created: bool = False
    skipped_not_followed_count: int = 0
    skipped_not_followed_samples: list[XSkippedNotFollowedRead] = Field(default_factory=list)


class XFollowingImportKolRead(BaseModel):
    id: int
    handle: str


class XFollowingImportErrorRead(BaseModel):
    row_index: int
    reason: str
    raw_snippet: str


class XFollowingImportStatsRead(BaseModel):
    received_rows: int
    following_true_rows: int
    created_kols_count: int
    updated_kols_count: int
    skipped_count: int
    created_kols: list[XFollowingImportKolRead] = Field(default_factory=list)
    updated_kols: list[XFollowingImportKolRead] = Field(default_factory=list)
    errors: list[XFollowingImportErrorRead] = Field(default_factory=list)


class AdminDeletePendingExtractionsRead(BaseModel):
    deleted_extractions_count: int
    deleted_raw_posts_count: int
    scoped_author_handle: str | None


class AdminCleanupDuplicatePendingRead(BaseModel):
    scanned: int
    duplicates_found: int
    dry_run: bool
    would_delete_ids: list[int] = Field(default_factory=list)
    would_keep_ids: list[int] = Field(default_factory=list)
    deleted_count: int = 0
    errors: list[str] = Field(default_factory=list)


class AutoReviewBackfillErrorRead(BaseModel):
    extraction_id: int | None = None
    raw_post_id: int | None = None
    error: str


class AdminBackfillAutoReviewRead(BaseModel):
    scanned: int
    approved_count: int
    rejected_count: int
    skipped_no_result_count: int
    skipped_no_confidence_count: int
    skipped_already_terminal_count: int
    errors: list[AutoReviewBackfillErrorRead] = Field(default_factory=list)


class AdminRefreshWrongExtractedJsonRead(BaseModel):
    scanned: int
    updated: int
    dry_run: bool
    updated_ids: list[int] = Field(default_factory=list)


class AdminHardDeleteRead(BaseModel):
    operation: str
    target: str
    derived_only: bool
    enable_cascade: bool
    also_delete_raw_posts: bool
    counts: dict[str, int]


class XImportTemplateRead(BaseModel):
    required_fields: list[str]
    optional_fields: list[str]
    notes: list[str]
    example: list[XImportItemCreate]


class RawPostsExtractBatchRequest(BaseModel):
    raw_post_ids: list[int] = Field(min_length=1, max_length=500)
    mode: Literal["pending_only", "pending_or_failed", "force"] = "pending_only"


class RawPostsExtractBatchRead(BaseModel):
    requested_count: int
    success_count: int
    skipped_count: int
    skipped_already_extracted_count: int
    skipped_already_pending_count: int = 0
    skipped_already_success_count: int = 0
    skipped_already_has_result_count: int = 0
    skipped_already_rejected_count: int = 0
    skipped_already_approved_count: int = 0
    skipped_not_followed_count: int = 0
    failed_count: int
    auto_approved_count: int = 0
    auto_rejected_count: int = 0
    resumed_requested_count: int = 0
    resumed_success: int = 0
    resumed_failed: int = 0
    resumed_skipped: int = 0


class ExtractJobCreateRequest(BaseModel):
    raw_post_ids: list[int] = Field(min_length=1, max_length=5000)
    mode: Literal["pending_only", "pending_or_failed", "force"] = "pending_or_failed"
    batch_size: int = Field(default=50, ge=1, le=500)
    batch_sleep_ms: int = Field(default=200, ge=0, le=60_000)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)


class ExtractJobCreateRead(BaseModel):
    job_id: str


class ExtractJobRead(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    mode: Literal["pending_only", "pending_or_failed", "force"]
    batch_size: int
    batch_sleep_ms: int
    requested_count: int
    success_count: int
    skipped_count: int
    skipped_already_extracted_count: int
    skipped_already_pending_count: int = 0
    skipped_already_success_count: int = 0
    skipped_already_has_result_count: int = 0
    skipped_already_rejected_count: int = 0
    skipped_already_approved_count: int = 0
    skipped_not_followed_count: int = 0
    failed_count: int
    auto_approved_count: int = 0
    auto_rejected_count: int = 0
    resumed_requested_count: int = 0
    resumed_success: int = 0
    resumed_failed: int = 0
    resumed_skipped: int = 0
    last_error_summary: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class XIngestProgressRead(BaseModel):
    scope: str
    author_handle: str | None
    total_raw_posts: int
    extracted_success_count: int
    pending_count: int
    failed_count: int
    no_extraction_count: int
    latest_error_summary: str | None
    latest_extraction_at: datetime | None


class XRetryFailedRead(BaseModel):
    author_handle: str | None
    requested_limit: int
    retried_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    failure_reasons: dict[str, int]


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


class DashboardClarityContributorRead(BaseModel):
    handle: str
    contribution: float


class DashboardClarityRankingRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    asset_id: int
    symbol: str
    name: str | None
    market: str | None
    direction: Stance
    s_raw: float
    clarity_score: float
    n: int
    k: int
    top_contributors: list[DashboardClarityContributorRead] = []


class DashboardExtractionStatsRead(BaseModel):
    window_hours: int
    extraction_count: int
    dummy_count: int
    openai_count: int
    error_count: int


class DashboardAssetLatestViewRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    kol_view_id: int
    horizon: Horizon
    stance: Stance
    confidence: int
    summary: str
    as_of: date
    created_at: datetime
    kol_id: int
    kol_display_name: str | None
    kol_handle: str | None


class DashboardAssetRead(BaseModel):
    id: int
    symbol: str
    name: str | None
    market: str | None
    new_views_24h: int
    new_views_7d: int
    latest_views_by_horizon: list[DashboardAssetLatestViewRead]


class DashboardActiveKolAssetRead(BaseModel):
    asset_id: int
    symbol: str
    views_count: int


class DashboardActiveKolRead(BaseModel):
    kol_id: int
    display_name: str | None
    handle: str
    platform: str
    views_count_7d: int
    top_assets: list[DashboardActiveKolAssetRead]


class DashboardRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    pending_extractions_count: int
    latest_pending_extractions: list[DashboardPendingExtractionRead]
    top_assets: list[DashboardTopAssetRead]
    clarity: list[DashboardClarityRead]
    clarity_ranking: list[DashboardClarityRankingRead]
    extraction_stats: DashboardExtractionStatsRead
    new_views_24h: int
    new_views_7d: int
    assets: list[DashboardAssetRead]
    active_kols_7d: list[DashboardActiveKolRead]


class AssetViewFeedItemRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: int
    kol_id: int
    kol_display_name: str | None
    kol_handle: str | None
    stance: Stance
    horizon: Horizon
    confidence: int
    summary: str
    source_url: str
    as_of: date
    created_at: datetime


class AssetViewsFeedRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    asset_id: int
    horizon: Horizon | None
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[AssetViewFeedItemRead]


class AssetViewsTimelineItemRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: int
    kol_id: int
    kol_display_name: str | None
    kol_handle: str | None
    stance: Stance
    horizon: Horizon
    confidence: int
    summary: str
    source_url: str
    as_of: date
    created_at: datetime


class AssetViewsTimelineRead(BaseModel):
    asset_id: int
    days: int
    since_date: date
    generated_at: datetime
    items: list[AssetViewsTimelineItemRead]


class ExtractionsStatsRead(BaseModel):
    bad_count: int
    total_count: int


class DailyDigestTopAssetRead(BaseModel):
    asset_id: int
    symbol: str
    name: str | None
    market: str | None
    new_views_24h: int
    new_views_7d: int
    weighted_views_24h: float = 0.0
    weighted_views_7d: float = 0.0


class DailyDigestHorizonCountRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    horizon: Horizon
    bull_count: int
    bear_count: int
    neutral_count: int


class DailyDigestTopViewRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    kol_id: int
    kol_display_name: str | None
    kol_handle: str | None
    stance: Stance
    horizon: Horizon
    confidence: int
    summary: str
    source_url: str
    as_of: date
    created_at: datetime
    kol_weight: float = 1.0
    weighted_score: float = 0.0


class DailyDigestAssetSummaryRead(BaseModel):
    asset_id: int
    symbol: str
    name: str | None
    market: str | None
    horizon_counts: list[DailyDigestHorizonCountRead]
    clarity: float
    top_views_bull: list[DailyDigestTopViewRead]
    top_views_bear: list[DailyDigestTopViewRead]
    top_views_neutral: list[DailyDigestTopViewRead]


class DailyDigestMetadataRead(BaseModel):
    generated_at: datetime
    days: int
    summary_window_start: datetime
    summary_window_end: datetime
    generated_from_ts: datetime
    generated_to_ts: datetime
    time_field_used: Literal["as_of", "posted_at", "created_at"]


class DailyDigestRead(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: int
    profile_id: int = 1
    digest_date: date
    version: int
    generated_at: datetime
    top_assets: list[DailyDigestTopAssetRead]
    per_asset_summary: list[DailyDigestAssetSummaryRead]
    metadata: DailyDigestMetadataRead


class ProfileSummaryRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime


class ProfileKolWeightRead(BaseModel):
    kol_id: int
    weight: float
    enabled: bool
    kol_display_name: str | None
    kol_handle: str
    kol_platform: str


class ProfileRead(BaseModel):
    id: int
    name: str
    created_at: datetime
    kols: list[ProfileKolWeightRead]
    markets: list[str]


class ProfileKolWeightUpdateItem(BaseModel):
    kol_id: int
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    enabled: bool = True


class ProfileKolsUpdateRequest(BaseModel):
    items: list[ProfileKolWeightUpdateItem] = Field(default_factory=list, max_length=500)


class ProfileMarketsUpdateRequest(BaseModel):
    markets: list[str] = Field(default_factory=list, max_length=50)


class ExtractorStatusRead(BaseModel):
    mode: str
    has_api_key: bool
    default_model: str
    base_url: str
    call_budget_remaining: int | None
    max_output_tokens: int


class RuntimeSettingsRead(BaseModel):
    extractor_mode: str
    provider_detected: str
    extraction_output_mode: str
    model: str
    has_api_key: bool
    base_url: str
    budget_remaining: int | None
    budget_total: int
    default_budget_total: int
    call_budget_override_enabled: bool
    call_budget_override_value: int | None
    override_value: int | None = None
    window_start: datetime
    window_end: datetime
    max_output_tokens: int
    auto_reject_confidence_threshold: int
    throttle: dict[str, int]
    effective_throttle: dict[str, int]
    burst: dict[str, bool | datetime | str | None]
    runtime_overrides: dict[str, bool]
    adaptive_throttle: dict[str, bool | int | str | datetime | None]


class RuntimeCallBudgetUpdateRequest(BaseModel):
    call_budget: int = Field(ge=0, le=100000)


class RuntimeBurstUpdateRequest(BaseModel):
    enabled: bool
    mode: Literal["normal", "unlimited_safe"] = "normal"
    call_budget: int = Field(default=0, ge=0, le=100000)
    duration_minutes: int = Field(default=0, ge=1, le=120)


class RuntimeThrottleUpdateRequest(BaseModel):
    max_concurrency: int = Field(ge=1, le=1000)
    max_rpm: int = Field(ge=1, le=100000)
    batch_size: int = Field(ge=1, le=1000)
    batch_sleep_ms: int = Field(ge=0, le=600000)
