"""create destinations table

Revision ID: 0e2bdbc1cc5a
Revises:
Create Date: 2026-07-04 13:27:23.152172

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

# revision identifiers, used by Alembic.
revision: str = '0e2bdbc1cc5a'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIMENSIONS = 1024


def upgrade() -> None:
    """Upgrade schema."""
    # Shared with destination_documents (the existing RAG table); other
    # tables already depend on this extension, so downgrade() must not drop
    # it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "destinations",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=False),
        sa.Column("region", sa.Text(), nullable=True),
        sa.Column("budget_level", sa.String(length=10), nullable=True),
        sa.Column("details", sa.Text(), nullable=False),
        sa.Column(
            "raw_sources",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "source_provenance",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS), nullable=True),
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
        sa.Column("embedding_version", sa.String(length=50), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("name", "country", name="uq_destinations_name_country"),
    )

    # ANN index for cosine similarity search over `embedding`. Alembic
    # autogenerate cannot produce this (pgvector index types/ops aren't
    # reflected), hence it is hand-written here.
    op.create_index(
        "ix_destinations_embedding_hnsw",
        "destinations",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_destinations_region", "destinations", ["region"])
    op.create_index("ix_destinations_budget_level", "destinations", ["budget_level"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_destinations_budget_level", table_name="destinations")
    op.drop_index("ix_destinations_region", table_name="destinations")
    op.drop_index("ix_destinations_embedding_hnsw", table_name="destinations")
    op.drop_table("destinations")
    # Intentionally not dropping the `vector` extension: destination_documents
    # (the existing RAG table) also depends on it.
