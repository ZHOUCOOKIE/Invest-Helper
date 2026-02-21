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
    created_at: datetime
