"""add extraction search indexes

Revision ID: c8b4f0e5a123
Revises: f34c577c633e
Create Date: 2026-02-25 16:20:00.000000
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "c8b4f0e5a123"
down_revision = "f34c577c633e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_posts_content_text_trgm "
        "ON raw_posts USING gin (content_text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_posts_author_handle_trgm "
        "ON raw_posts USING gin (author_handle gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_raw_posts_url_trgm "
        "ON raw_posts USING gin (url gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_post_extractions_summary_trgm "
        "ON post_extractions USING gin ((coalesce(extracted_json->>'summary', '')) gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_post_extractions_reasoning_trgm "
        "ON post_extractions USING gin ((coalesce(extracted_json->>'reasoning', '')) gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_post_extractions_reasoning_trgm")
    op.execute("DROP INDEX IF EXISTS ix_post_extractions_summary_trgm")
    op.execute("DROP INDEX IF EXISTS ix_raw_posts_url_trgm")
    op.execute("DROP INDEX IF EXISTS ix_raw_posts_author_handle_trgm")
    op.execute("DROP INDEX IF EXISTS ix_raw_posts_content_text_trgm")
