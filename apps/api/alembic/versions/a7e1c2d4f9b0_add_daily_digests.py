"""add daily digests

Revision ID: a7e1c2d4f9b0
Revises: f2d9c4a77b1a
Create Date: 2026-02-23 15:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "a7e1c2d4f9b0"
down_revision: Union[str, Sequence[str], None] = "f2d9c4a77b1a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "daily_digests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("digest_date", sa.Date(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("days", sa.Integer(), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("digest_date", "version", name="uq_daily_digests_date_version"),
    )
    op.create_index("ix_daily_digests_id", "daily_digests", ["id"], unique=False)
    op.create_index("ix_daily_digests_digest_date", "daily_digests", ["digest_date"], unique=False)
    op.create_index(
        "ix_daily_digests_digest_date_version",
        "daily_digests",
        ["digest_date", "version"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_daily_digests_digest_date_version", table_name="daily_digests")
    op.drop_index("ix_daily_digests_digest_date", table_name="daily_digests")
    op.drop_index("ix_daily_digests_id", table_name="daily_digests")
    op.drop_table("daily_digests")
