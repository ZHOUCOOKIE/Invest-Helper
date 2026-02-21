"""add extraction review fields

Revision ID: 5a2e3c1b9d4f
Revises: e13c9b44a2de
Create Date: 2026-02-21 03:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5a2e3c1b9d4f"
down_revision: Union[str, Sequence[str], None] = "e13c9b44a2de"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("post_extractions", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("post_extractions", sa.Column("reviewed_by", sa.String(length=128), nullable=True))
    op.add_column("post_extractions", sa.Column("review_note", sa.String(length=1024), nullable=True))
    op.add_column("post_extractions", sa.Column("applied_kol_view_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_post_extractions_applied_kol_view_id",
        "post_extractions",
        "kol_views",
        ["applied_kol_view_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_post_extractions_status_created",
        "post_extractions",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_post_extractions_status_created", table_name="post_extractions")
    op.drop_constraint("fk_post_extractions_applied_kol_view_id", "post_extractions", type_="foreignkey")
    op.drop_column("post_extractions", "applied_kol_view_id")
    op.drop_column("post_extractions", "review_note")
    op.drop_column("post_extractions", "reviewed_by")
    op.drop_column("post_extractions", "reviewed_at")
