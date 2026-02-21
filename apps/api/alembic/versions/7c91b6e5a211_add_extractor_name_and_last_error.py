"""add extractor_name and last_error to post_extractions

Revision ID: 7c91b6e5a211
Revises: 5a2e3c1b9d4f
Create Date: 2026-02-21 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7c91b6e5a211"
down_revision: Union[str, Sequence[str], None] = "5a2e3c1b9d4f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "post_extractions",
        sa.Column("extractor_name", sa.String(length=64), nullable=False, server_default=sa.text("'dummy'")),
    )
    op.add_column("post_extractions", sa.Column("last_error", sa.String(length=2048), nullable=True))


def downgrade() -> None:
    op.drop_column("post_extractions", "last_error")
    op.drop_column("post_extractions", "extractor_name")
