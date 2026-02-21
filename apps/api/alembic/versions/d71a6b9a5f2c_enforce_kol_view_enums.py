"""enforce kol view enums

Revision ID: d71a6b9a5f2c
Revises: 8d9f20b4e4be
Create Date: 2026-02-21 00:30:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "d71a6b9a5f2c"
down_revision: Union[str, Sequence[str], None] = "8d9f20b4e4be"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_kol_views_stance_enum",
        "kol_views",
        "stance IN ('bull', 'bear', 'neutral')",
    )
    op.create_check_constraint(
        "ck_kol_views_horizon_enum",
        "kol_views",
        "horizon IN ('intraday', '1w', '1m', '3m', '1y')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_kol_views_horizon_enum", "kol_views", type_="check")
    op.drop_constraint("ck_kol_views_stance_enum", "kol_views", type_="check")
