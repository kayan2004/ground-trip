from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies.auth import get_current_user
from app.db.models.user import User
from app.schemas.classifier import (
    TravelStylePredictionRequest,
    TravelStylePredictionResponse,
)
from app.services.classifier import predict_travel_style

router = APIRouter(prefix="/tools", tags=["tools"])


@router.post(
    "/classify-travel-style",
    response_model=TravelStylePredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def classify_travel_style_route(
    payload: TravelStylePredictionRequest,
    request: Request,
    _current_user: User = Depends(get_current_user),
) -> TravelStylePredictionResponse:
    model = request.app.state.resources.get("travel_style_model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Travel style model is not loaded.",
        )

    return predict_travel_style(model, payload)
