from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.schemas.feedback import FeedbackCreate, FeedbackRead
from app.services.feedback import RecommendationNotFoundError, submit_feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackRead, status_code=status.HTTP_200_OK)
async def submit_feedback_route(
    payload: FeedbackCreate,
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackRead:
    try:
        return await submit_feedback(session, payload)
    except RecommendationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
