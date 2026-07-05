from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.recommendations import (
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)
from app.services.destination_recommendations import recommend_destinations

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/recommend-destinations",
    response_model=DestinationRecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def recommend_destinations_route(
    payload: DestinationRecommendationRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    _current_user: User = Depends(get_current_user),
) -> DestinationRecommendationResponse:
    http_client = request.app.state.resources.get("http_client")
    if http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HTTP client is not available.",
        )

    return await recommend_destinations(
        session,
        http_client,
        request.app.state.settings,
        payload,
    )
