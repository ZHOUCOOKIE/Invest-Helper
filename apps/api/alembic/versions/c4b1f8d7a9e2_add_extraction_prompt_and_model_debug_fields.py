"""add extraction prompt and model debug fields

Revision ID: c4b1f8d7a9e2
Revises: 7c91b6e5a211
Create Date: 2026-02-21 09:05:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "c4b1f8d7a9e2"
down_revision: Union[str, Sequence[str], None] = "7c91b6e5a211"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "post_extractions",
        sa.Column("prompt_version", sa.Text(), nullable=False, server_default=sa.text("'extract_v1'")),
    )
    op.add_column("post_extractions", sa.Column("prompt_text", sa.Text(), nullable=True))
    op.add_column("post_extractions", sa.Column("prompt_hash", sa.Text(), nullable=True))
    op.add_column("post_extractions", sa.Column("raw_model_output", sa.Text(), nullable=True))
    op.add_column("post_extractions", sa.Column("parsed_model_output", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column("post_extractions", sa.Column("model_latency_ms", sa.Integer(), nullable=True))
    op.add_column("post_extractions", sa.Column("model_input_tokens", sa.Integer(), nullable=True))
    op.add_column("post_extractions", sa.Column("model_output_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("post_extractions", "model_output_tokens")
    op.drop_column("post_extractions", "model_input_tokens")
    op.drop_column("post_extractions", "model_latency_ms")
    op.drop_column("post_extractions", "parsed_model_output")
    op.drop_column("post_extractions", "raw_model_output")
    op.drop_column("post_extractions", "prompt_hash")
    op.drop_column("post_extractions", "prompt_text")
    op.drop_column("post_extractions", "prompt_version")
