"""merge heads 6e4f8a9b2c1d and c8b4f0e5a123

Revision ID: 708527c13f5f
Revises: 6e4f8a9b2c1d, c8b4f0e5a123
Create Date: 2026-02-26 06:00:29.572243

"""
from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = '708527c13f5f'
down_revision: Union[str, Sequence[str], None] = ('6e4f8a9b2c1d', 'c8b4f0e5a123')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
