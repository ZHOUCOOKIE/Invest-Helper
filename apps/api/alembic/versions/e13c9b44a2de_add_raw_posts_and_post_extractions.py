"""add raw_posts and post_extractions

Revision ID: e13c9b44a2de
Revises: d71a6b9a5f2c
Create Date: 2026-02-21 01:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e13c9b44a2de"
down_revision: Union[str, Sequence[str], None] = "d71a6b9a5f2c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "raw_posts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("platform", sa.String(length=32), nullable=False),
        sa.Column("author_handle", sa.String(length=128), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("url", sa.String(length=1024), nullable=False),
        sa.Column("content_text", sa.String(length=8192), nullable=False),
        sa.Column("posted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("platform", "external_id", name="uq_raw_posts_platform_external_id"),
    )
    op.create_index(op.f("ix_raw_posts_id"), "raw_posts", ["id"], unique=False)
    op.create_index("ix_raw_posts_platform_posted_at", "raw_posts", ["platform", "posted_at"], unique=False)

    op.create_table(
        "post_extractions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("raw_post_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("extracted_json", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["raw_post_id"], ["raw_posts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_check_constraint(
        "ck_post_extractions_status_enum",
        "post_extractions",
        "status IN ('pending', 'approved', 'rejected')",
    )
    op.create_index(op.f("ix_post_extractions_id"), "post_extractions", ["id"], unique=False)
    op.create_index(
        "ix_post_extractions_raw_post_created",
        "post_extractions",
        ["raw_post_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_post_extractions_raw_post_created", table_name="post_extractions")
    op.drop_index(op.f("ix_post_extractions_id"), table_name="post_extractions")
    op.drop_constraint("ck_post_extractions_status_enum", "post_extractions", type_="check")
    op.drop_table("post_extractions")

    op.drop_index("ix_raw_posts_platform_posted_at", table_name="raw_posts")
    op.drop_index(op.f("ix_raw_posts_id"), table_name="raw_posts")
    op.drop_table("raw_posts")
