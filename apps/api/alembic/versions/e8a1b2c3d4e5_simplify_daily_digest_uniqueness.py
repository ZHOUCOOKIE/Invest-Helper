"""simplify daily digest uniqueness

Revision ID: e8a1b2c3d4e5
Revises: f34c577c633e
Create Date: 2026-03-06 00:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "e8a1b2c3d4e5"
down_revision = "f34c577c633e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("uq_daily_digests_profile_date_version", "daily_digests", type_="unique")
    op.drop_index("ix_daily_digests_profile_date_version", table_name="daily_digests")
    op.create_index("ix_daily_digests_profile_date", "daily_digests", ["profile_id", "digest_date"], unique=False)
    op.create_unique_constraint(
        "uq_daily_digests_profile_date",
        "daily_digests",
        ["profile_id", "digest_date"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_daily_digests_profile_date", "daily_digests", type_="unique")
    op.drop_index("ix_daily_digests_profile_date", table_name="daily_digests")
    op.create_index(
        "ix_daily_digests_profile_date_version",
        "daily_digests",
        ["profile_id", "digest_date", "version"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_daily_digests_profile_date_version",
        "daily_digests",
        ["profile_id", "digest_date", "version"],
    )
