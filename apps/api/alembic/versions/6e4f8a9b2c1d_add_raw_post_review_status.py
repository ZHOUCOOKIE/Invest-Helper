"""add raw post review status

Revision ID: 6e4f8a9b2c1d
Revises: f34c577c633e
Create Date: 2026-02-24 12:10:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6e4f8a9b2c1d"
down_revision: Union[str, Sequence[str], None] = "f34c577c633e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    review_status_enum = sa.Enum(
        "unreviewed",
        "approved",
        "rejected",
        name="review_status_enum",
        native_enum=False,
        create_constraint=True,
    )
    review_status_enum.create(op.get_bind(), checkfirst=True)

    op.add_column(
        "raw_posts",
        sa.Column(
            "review_status",
            review_status_enum,
            nullable=False,
            server_default=sa.text("'unreviewed'"),
        ),
    )
    op.add_column("raw_posts", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("raw_posts", sa.Column("reviewed_by", sa.String(length=128), nullable=True))

    op.execute(
        sa.text(
            """
            WITH latest_terminal AS (
                SELECT
                    pe.raw_post_id,
                    pe.status,
                    pe.reviewed_at,
                    pe.reviewed_by,
                    ROW_NUMBER() OVER (
                        PARTITION BY pe.raw_post_id
                        ORDER BY pe.created_at DESC NULLS LAST, pe.id DESC
                    ) AS rn
                FROM post_extractions pe
                WHERE pe.status IN ('approved', 'rejected')
            )
            UPDATE raw_posts rp
            SET
                review_status = latest_terminal.status,
                reviewed_at = latest_terminal.reviewed_at,
                reviewed_by = latest_terminal.reviewed_by
            FROM latest_terminal
            WHERE latest_terminal.rn = 1
              AND latest_terminal.raw_post_id = rp.id
            """
        )
    )

    op.alter_column("raw_posts", "review_status", server_default=None)


def downgrade() -> None:
    op.drop_column("raw_posts", "reviewed_by")
    op.drop_column("raw_posts", "reviewed_at")
    op.drop_column("raw_posts", "review_status")
