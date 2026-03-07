"""add weekly digests

Revision ID: 2b7f8d9c1a2e
Revises: f9b8c7d6e5a4
Create Date: 2026-03-07 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "2b7f8d9c1a2e"
down_revision: Union[str, None] = "f9b8c7d6e5a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "weekly_digests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("report_kind", sa.String(length=32), nullable=False),
        sa.Column("anchor_date", sa.Date(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("days", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["user_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "report_kind", "anchor_date", name="uq_weekly_digests_profile_kind_anchor"),
    )
    op.create_index("ix_weekly_digests_id", "weekly_digests", ["id"], unique=False)
    op.create_index("ix_weekly_digests_profile_id", "weekly_digests", ["profile_id"], unique=False)
    op.create_index("ix_weekly_digests_anchor_date", "weekly_digests", ["anchor_date"], unique=False)
    op.create_index(
        "ix_weekly_digests_profile_kind_anchor",
        "weekly_digests",
        ["profile_id", "report_kind", "anchor_date"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_weekly_digests_profile_kind_anchor", table_name="weekly_digests")
    op.drop_index("ix_weekly_digests_anchor_date", table_name="weekly_digests")
    op.drop_index("ix_weekly_digests_profile_id", table_name="weekly_digests")
    op.drop_index("ix_weekly_digests_id", table_name="weekly_digests")
    op.drop_table("weekly_digests")
