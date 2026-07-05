import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class RecommendationRead(BaseModel):
    id: int
    destination_id: uuid.UUID
    destination_name: str
    country: str
    rank_position: int
    score: float
    features: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
