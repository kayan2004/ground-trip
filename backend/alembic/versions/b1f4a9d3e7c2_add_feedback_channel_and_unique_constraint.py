"""add channel column and unique constraint to feedback

Revision ID: b1f4a9d3e7c2
Revises: a7c3e5f19d02
Create Date: 2026-07-06 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b1f4a9d3e7c2'
down_revision: Union[str, Sequence[str], None] = 'a7c3e5f19d02'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "feedback",
        sa.Column(
            "channel",
            sa.String(length=50),
            nullable=False,
            server_default="web",
        ),
    )
    op.create_unique_constraint(
        "uq_feedback_recommendation_session",
        "feedback",
        ["recommendation_id", "session_uuid"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_feedback_recommendation_session", "feedback", type_="unique"
    )
    op.drop_column("feedback", "channel")
