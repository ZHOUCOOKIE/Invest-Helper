"""add raw_posts.kol_id

Revision ID: 1f3e2d4c5b6a
Revises: d71a6b9a5f2c
Create Date: 2026-02-24 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1f3e2d4c5b6a"
down_revision = "d71a6b9a5f2c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("raw_posts", sa.Column("kol_id", sa.Integer(), nullable=True))
    op.create_index("ix_raw_posts_kol_id", "raw_posts", ["kol_id"], unique=False)
    op.create_foreign_key(
        "fk_raw_posts_kol_id_kols",
        "raw_posts",
        "kols",
        ["kol_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_raw_posts_kol_id_kols", "raw_posts", type_="foreignkey")
    op.drop_index("ix_raw_posts_kol_id", table_name="raw_posts")
    op.drop_column("raw_posts", "kol_id")
