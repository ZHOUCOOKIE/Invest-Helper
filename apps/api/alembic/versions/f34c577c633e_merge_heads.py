"""merge heads

Revision ID: f34c577c633e
Revises: 1f3e2d4c5b6a, b81f4d2d9d2f
Create Date: 2026-02-24 01:15:45.503694

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f34c577c633e'
down_revision: Union[str, Sequence[str], None] = ('1f3e2d4c5b6a', 'b81f4d2d9d2f')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
