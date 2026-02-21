from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from enums import ExtractionStatus, Horizon, Stance

enum_values = lambda enum_cls: [item.value for item in enum_cls]


class Base(DeclarativeBase):
    pass


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    market: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    kol_views: Mapped[list["KolView"]] = relationship(back_populates="asset", cascade="all, delete-orphan")


class Kol(Base):
    __tablename__ = "kols"
    __table_args__ = (
        UniqueConstraint("platform", "handle", name="uq_kols_platform_handle"),
        Index("ix_kols_enabled", "enabled"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    handle: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    views: Mapped[list["KolView"]] = relationship(back_populates="kol", cascade="all, delete-orphan")


class KolView(Base):
    __tablename__ = "kol_views"
    __table_args__ = (
        UniqueConstraint(
            "kol_id",
            "asset_id",
            "horizon",
            "as_of",
            "source_url",
            name="uq_kol_views_dedup",
        ),
        Index("ix_kol_views_asset_horizon_confidence", "asset_id", "horizon", "confidence"),
        Index("ix_kol_views_kol_asset", "kol_id", "asset_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    kol_id: Mapped[int] = mapped_column(ForeignKey("kols.id", ondelete="CASCADE"), nullable=False)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"), nullable=False)
    stance: Mapped[Stance] = mapped_column(
        SQLEnum(
            Stance,
            name="stance_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    horizon: Mapped[Horizon] = mapped_column(
        SQLEnum(
            Horizon,
            name="horizon_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            values_callable=enum_values,
        ),
        nullable=False,
    )
    confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    kol: Mapped["Kol"] = relationship(back_populates="views")
    asset: Mapped["Asset"] = relationship(back_populates="kol_views")


class RawPost(Base):
    __tablename__ = "raw_posts"
    __table_args__ = (
        UniqueConstraint("platform", "external_id", name="uq_raw_posts_platform_external_id"),
        Index("ix_raw_posts_platform_posted_at", "platform", "posted_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    platform: Mapped[str] = mapped_column(String(32), nullable=False)
    author_handle: Mapped[str] = mapped_column(String(128), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    url: Mapped[str] = mapped_column(String(1024), nullable=False)
    content_text: Mapped[str] = mapped_column(String(8192), nullable=False)
    posted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    extractions: Mapped[list["PostExtraction"]] = relationship(
        back_populates="raw_post", cascade="all, delete-orphan"
    )


class PostExtraction(Base):
    __tablename__ = "post_extractions"
    __table_args__ = (
        Index("ix_post_extractions_raw_post_created", "raw_post_id", "created_at"),
        Index("ix_post_extractions_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    raw_post_id: Mapped[int] = mapped_column(ForeignKey("raw_posts.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[ExtractionStatus] = mapped_column(
        SQLEnum(
            ExtractionStatus,
            name="extraction_status_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            values_callable=enum_values,
        ),
        nullable=False,
        default=ExtractionStatus.pending,
        server_default=text("'pending'"),
    )
    extracted_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    extractor_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="dummy",
        server_default=text("'dummy'"),
    )
    last_error: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_note: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    applied_kol_view_id: Mapped[int | None] = mapped_column(
        ForeignKey("kol_views.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    raw_post: Mapped["RawPost"] = relationship(back_populates="extractions")
