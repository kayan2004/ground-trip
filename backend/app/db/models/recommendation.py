import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    # References destinations.id (UUID PK on the separate DestinationCorpusBase
    # metadata - see app/db/models/destination.py). No ORM relationship() to
    # Destination: that class lives on its own declarative registry, so
    # relationship() couldn't resolve it by name anyway, and coupling the two
    # bases would undo the deliberate Alembic-only isolation for `destinations`.
    destination_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("destinations.id"), nullable=False, index=True
    )
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    agent_run = relationship("AgentRun", back_populates="recommendations")
    feedback = relationship("Feedback", back_populates="recommendation", cascade="all, delete-orphan")
