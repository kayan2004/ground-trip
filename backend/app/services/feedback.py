from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import Feedback
from app.schemas.feedback import FeedbackCreate, FeedbackRead


class RecommendationNotFoundError(Exception):
    """Raised when FeedbackCreate.recommendation_id does not exist."""


async def submit_feedback(
    session: AsyncSession,
    payload: FeedbackCreate,
    *,
    channel: str = "web",
) -> FeedbackRead:
    statement = (
        insert(Feedback)
        .values(
            recommendation_id=payload.recommendation_id,
            session_uuid=payload.session_uuid,
            verdict=payload.verdict,
            channel=channel,
        )
        .on_conflict_do_update(
            constraint="uq_feedback_recommendation_session",
            set_={"verdict": payload.verdict, "channel": channel},
        )
        .returning(
            Feedback.id,
            Feedback.recommendation_id,
            Feedback.session_uuid,
            Feedback.verdict,
            Feedback.channel,
            Feedback.created_at,
        )
    )

    try:
        result = await session.execute(statement)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise RecommendationNotFoundError(
            f"Recommendation {payload.recommendation_id} does not exist."
        ) from exc

    row = result.one()
    return FeedbackRead(
        id=row.id,
        recommendation_id=row.recommendation_id,
        session_uuid=row.session_uuid,
        verdict=row.verdict,
        channel=row.channel,
        created_at=row.created_at,
    )
