"""add profiles and digest profile scope

Revision ID: c9f3e1a7b2d0
Revises: a7e1c2d4f9b0
Create Date: 2026-02-23 18:30:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c9f3e1a7b2d0"
down_revision: Union[str, Sequence[str], None] = "a7e1c2d4f9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "user_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_user_profiles_name"),
    )
    op.create_index("ix_user_profiles_id", "user_profiles", ["id"], unique=False)

    op.execute(
        sa.text(
            "INSERT INTO user_profiles (id, name) VALUES (1, 'default') "
            "ON CONFLICT (id) DO NOTHING"
        )
    )

    op.create_table(
        "profile_kol_weights",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("kol_id", sa.Integer(), nullable=False),
        sa.Column("weight", sa.Float(), server_default=sa.text("1.0"), nullable=False),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["kol_id"], ["kols.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["profile_id"], ["user_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "kol_id", name="uq_profile_kol_weights_profile_kol"),
    )
    op.create_index("ix_profile_kol_weights_id", "profile_kol_weights", ["id"], unique=False)
    op.create_index("ix_profile_kol_weights_profile_id", "profile_kol_weights", ["profile_id"], unique=False)

    op.create_table(
        "profile_markets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("profile_id", sa.Integer(), nullable=False),
        sa.Column("market", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["profile_id"], ["user_profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("profile_id", "market", name="uq_profile_markets_profile_market"),
    )
    op.create_index("ix_profile_markets_id", "profile_markets", ["id"], unique=False)
    op.create_index("ix_profile_markets_profile_id", "profile_markets", ["profile_id"], unique=False)

    op.add_column(
        "daily_digests",
        sa.Column("profile_id", sa.Integer(), nullable=True, server_default=sa.text("1")),
    )
    op.execute(sa.text("UPDATE daily_digests SET profile_id = 1 WHERE profile_id IS NULL"))
    op.create_foreign_key(
        "fk_daily_digests_profile_id",
        "daily_digests",
        "user_profiles",
        ["profile_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.alter_column("daily_digests", "profile_id", nullable=False)

    op.drop_constraint("uq_daily_digests_date_version", "daily_digests", type_="unique")
    op.drop_index("ix_daily_digests_digest_date_version", table_name="daily_digests")

    op.create_index("ix_daily_digests_profile_id", "daily_digests", ["profile_id"], unique=False)
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


def downgrade() -> None:
    op.drop_constraint("uq_daily_digests_profile_date_version", "daily_digests", type_="unique")
    op.drop_index("ix_daily_digests_profile_date_version", table_name="daily_digests")
    op.drop_index("ix_daily_digests_profile_id", table_name="daily_digests")

    op.create_index(
        "ix_daily_digests_digest_date_version",
        "daily_digests",
        ["digest_date", "version"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_daily_digests_date_version",
        "daily_digests",
        ["digest_date", "version"],
    )

    op.drop_constraint("fk_daily_digests_profile_id", "daily_digests", type_="foreignkey")
    op.drop_column("daily_digests", "profile_id")

    op.drop_index("ix_profile_markets_profile_id", table_name="profile_markets")
    op.drop_index("ix_profile_markets_id", table_name="profile_markets")
    op.drop_table("profile_markets")

    op.drop_index("ix_profile_kol_weights_profile_id", table_name="profile_kol_weights")
    op.drop_index("ix_profile_kol_weights_id", table_name="profile_kol_weights")
    op.drop_table("profile_kol_weights")

    op.drop_index("ix_user_profiles_id", table_name="user_profiles")
    op.drop_table("user_profiles")
