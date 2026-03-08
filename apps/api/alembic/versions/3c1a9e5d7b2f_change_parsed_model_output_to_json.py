"""change parsed_model_output to json

Revision ID: 3c1a9e5d7b2f
Revises: 2b7f8d9c1a2e
Create Date: 2026-03-08 03:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "3c1a9e5d7b2f"
down_revision: Union[str, Sequence[str], None] = "2b7f8d9c1a2e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "post_extractions",
        "parsed_model_output",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        postgresql_using="parsed_model_output::json",
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        "post_extractions",
        "parsed_model_output",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="parsed_model_output::jsonb",
        existing_nullable=True,
    )
