"""create baseline tables (users, agent_runs, tool_logs, destination_documents)

Revision ID: 5f2b7a3d9c14
Revises: 0e2bdbc1cc5a
Create Date: 2026-07-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '5f2b7a3d9c14'
down_revision: Union[str, Sequence[str], None] = '0e2bdbc1cc5a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 1024


def upgrade() -> None:
    """Upgrade schema.

    Reflects the schema as it already exists on any DB where these tables
    were created via Base.metadata.create_all() at startup (i.e. every DB
    this project has run against so far). Existing populated DBs should
    `alembic stamp 5f2b7a3d9c14` rather than run this upgrade - see
    backend/README.md's migration section.
    """
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=50), server_default=sa.text("'completed'"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])

    op.create_table(
        "tool_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "agent_run_id", sa.Integer(), sa.ForeignKey("agent_runs.id"), nullable=False
        ),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("input_payload", sa.Text(), nullable=False),
        sa.Column("output_payload", sa.Text(), nullable=False),
        sa.Column(
            "status", sa.String(length=50), server_default=sa.text("'completed'"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_tool_logs_agent_run_id", "tool_logs", ["agent_run_id"])

    # Vector extension already created by 0e2bdbc1cc5a (this migration's
    # parent in the chain), so it's guaranteed to exist here.
    op.create_table(
        "destination_documents",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("destination_name", sa.String(length=255), nullable=False),
        sa.Column("travel_style", sa.String(length=50), nullable=True),
        sa.Column("source_type", sa.String(length=100), nullable=False),
        sa.Column("source_title", sa.String(length=255), nullable=False),
        sa.Column("source_url", sa.String(length=1000), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_destination_documents_destination_name", "destination_documents", ["destination_name"]
    )
    op.create_index(
        "ix_destination_documents_travel_style", "destination_documents", ["travel_style"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_destination_documents_travel_style", table_name="destination_documents"
    )
    op.drop_index(
        "ix_destination_documents_destination_name", table_name="destination_documents"
    )
    op.drop_table("destination_documents")

    op.drop_index("ix_tool_logs_agent_run_id", table_name="tool_logs")
    op.drop_table("tool_logs")

    op.drop_index("ix_agent_runs_user_id", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
