"""add partial unique index for active extractions

Revision ID: b81f4d2d9d2f
Revises: c9f3e1a7b2d0
Create Date: 2026-02-23 18:20:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b81f4d2d9d2f"
down_revision: Union[str, Sequence[str], None] = "c9f3e1a7b2d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_post_extractions_active_raw_post_id",
        "post_extractions",
        ["raw_post_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending' AND (last_error IS NULL OR btrim(last_error) = '')"),
    )


def downgrade() -> None:
    op.drop_index("uq_post_extractions_active_raw_post_id", table_name="post_extractions")
