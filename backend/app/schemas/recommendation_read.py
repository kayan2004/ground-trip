import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.recommendations import DestinationFeatureSnapshot


class RecommendationRead(BaseModel):
    id: int
    destination_id: uuid.UUID
    destination_name: str
    country: str
    rank_position: int
    score: float
    # Optional despite the DB column being NOT NULL - a malformed/legacy-shaped
    # JSON blob (predating this schema, or hand-edited) shouldn't 500 the whole
    # recommendations list; get_recommendations_for_agent_run() falls back to
    # None rather than let Pydantic validation raise.
    features: DestinationFeatureSnapshot | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
