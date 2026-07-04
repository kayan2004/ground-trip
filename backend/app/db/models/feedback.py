import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, SmallInteger, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    recommendation_id: Mapped[int] = mapped_column(
        ForeignKey("recommendations.id"), nullable=False, index=True
    )
    # Anonymous client UUID, not a users FK - feedback can be submitted
    # without an authenticated session.
    session_uuid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    verdict: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # +1 / -1
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    recommendation = relationship("Recommendation", back_populates="feedback")
