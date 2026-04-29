from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.schemas.recommendations import (
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)
from app.services.recommendations import recommend_destinations

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/recommend-destinations",
    response_model=DestinationRecommendationResponse,
    status_code=status.HTTP_200_OK,
)
async def recommend_destinations_route(
    payload: DestinationRecommendationRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> DestinationRecommendationResponse:
    catalog = request.app.state.resources.get("destination_catalog")
    if catalog is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Destination catalog is not loaded.",
        )

    return recommend_destinations(catalog, payload)
