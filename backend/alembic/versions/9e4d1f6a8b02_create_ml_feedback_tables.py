"""create ml feedback schema (recommendations, feedback, tag_definitions)

Revision ID: 9e4d1f6a8b02
Revises: 5f2b7a3d9c14
Create Date: 2026-07-04 00:00:01.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = '9e4d1f6a8b02'
down_revision: Union[str, Sequence[str], None] = '5f2b7a3d9c14'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("destinations", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("agent_runs", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    op.create_table(
        "recommendations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("agent_run_id", sa.Integer(), sa.ForeignKey("agent_runs.id"), nullable=False),
        # destinations.id is a UUID PK (see 0e2bdbc1cc5a), not an int.
        sa.Column(
            "destination_id",
            UUID(as_uuid=True),
            sa.ForeignKey("destinations.id"),
            nullable=False,
        ),
        sa.Column("rank_position", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("features", JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_recommendations_agent_run_id", "recommendations", ["agent_run_id"])
    op.create_index("ix_recommendations_destination_id", "recommendations", ["destination_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "recommendation_id", sa.Integer(), sa.ForeignKey("recommendations.id"), nullable=False
        ),
        sa.Column("session_uuid", UUID(as_uuid=True), nullable=False),
        sa.Column("verdict", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_feedback_recommendation_id", "feedback", ["recommendation_id"])
    op.create_index("ix_feedback_session_uuid", "feedback", ["session_uuid"])
    # Spec-mandated partial index. verdict is NOT NULL at the column level,
    # so this condition is currently always true (functionally a full
    # index) - kept as specified in case verdict is ever relaxed to
    # nullable (e.g. a withdrawn-feedback state) without a follow-up
    # migration to add the partial index retroactively.
    op.create_index(
        "ix_feedback_recommendation_id_verdict_not_null",
        "feedback",
        ["recommendation_id"],
        postgresql_where=sa.text("verdict IS NOT NULL"),
    )

    op.create_table(
        "tag_definitions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cluster_id", sa.Integer(), nullable=False),
        sa.Column("tag_name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("quality_metrics", JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("cluster_id", name="uq_tag_definitions_cluster_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("tag_definitions")

    op.drop_index("ix_feedback_recommendation_id_verdict_not_null", table_name="feedback")
    op.drop_index("ix_feedback_session_uuid", table_name="feedback")
    op.drop_index("ix_feedback_recommendation_id", table_name="feedback")
    op.drop_table("feedback")

    op.drop_index("ix_recommendations_destination_id", table_name="recommendations")
    op.drop_index("ix_recommendations_agent_run_id", table_name="recommendations")
    op.drop_table("recommendations")

    op.drop_column("agent_runs", "deleted_at")
    op.drop_column("destinations", "deleted_at")
