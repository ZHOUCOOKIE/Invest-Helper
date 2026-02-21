"""add kols and kol_views

Revision ID: 8d9f20b4e4be
Revises: 9d013009192b
Create Date: 2026-02-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8d9f20b4e4be"
down_revision: Union[str, Sequence[str], None] = "9d013009192b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kols",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("handle", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("platform", "handle", name="uq_kols_platform_handle"),
    )
    op.create_index(op.f("ix_kols_id"), "kols", ["id"], unique=False)
    op.create_index("ix_kols_enabled", "kols", ["enabled"], unique=False)

    op.create_table(
        "kol_views",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("kol_id", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column("stance", sa.String(length=16), nullable=False),
        sa.Column("horizon", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Integer(), nullable=False),
        sa.Column("summary", sa.String(length=1024), nullable=False),
        sa.Column("source_url", sa.String(length=1024), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["kol_id"], ["kols.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kol_id",
            "asset_id",
            "horizon",
            "as_of",
            "source_url",
            name="uq_kol_views_dedup",
        ),
    )
    op.create_index(op.f("ix_kol_views_id"), "kol_views", ["id"], unique=False)
    op.create_index(
        "ix_kol_views_asset_horizon_confidence",
        "kol_views",
        ["asset_id", "horizon", "confidence"],
        unique=False,
    )
    op.create_index("ix_kol_views_kol_asset", "kol_views", ["kol_id", "asset_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_kol_views_kol_asset", table_name="kol_views")
    op.drop_index("ix_kol_views_asset_horizon_confidence", table_name="kol_views")
    op.drop_index(op.f("ix_kol_views_id"), table_name="kol_views")
    op.drop_table("kol_views")

    op.drop_index("ix_kols_enabled", table_name="kols")
    op.drop_index(op.f("ix_kols_id"), table_name="kols")
    op.drop_table("kols")
