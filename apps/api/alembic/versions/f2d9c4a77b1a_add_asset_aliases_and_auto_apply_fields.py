"""add asset aliases and auto apply fields

Revision ID: f2d9c4a77b1a
Revises: c4b1f8d7a9e2
Create Date: 2026-02-21 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f2d9c4a77b1a"
down_revision: Union[str, Sequence[str], None] = "c4b1f8d7a9e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "asset_aliases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("asset_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["asset_id"], ["assets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_asset_aliases_asset_id", "asset_aliases", ["asset_id"], unique=False)
    op.create_index("ix_asset_aliases_id", "asset_aliases", ["id"], unique=False)
    op.create_index(
        "uq_asset_aliases_alias_norm",
        "asset_aliases",
        [sa.text("lower(trim(alias))")],
        unique=True,
    )

    op.add_column(
        "post_extractions",
        sa.Column("auto_applied_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column("post_extractions", sa.Column("auto_policy", sa.String(length=32), nullable=True))
    op.add_column(
        "post_extractions",
        sa.Column("auto_applied_kol_view_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("post_extractions", "auto_applied_kol_view_ids")
    op.drop_column("post_extractions", "auto_policy")
    op.drop_column("post_extractions", "auto_applied_count")

    op.drop_index("uq_asset_aliases_alias_norm", table_name="asset_aliases")
    op.drop_index("ix_asset_aliases_id", table_name="asset_aliases")
    op.drop_index("ix_asset_aliases_asset_id", table_name="asset_aliases")
    op.drop_table("asset_aliases")
