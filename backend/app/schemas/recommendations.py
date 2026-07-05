import uuid
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]


class DestinationRecommendationRequest(BaseModel):
    query_text: str = Field(min_length=1, max_length=4000)
    budget_level: Level | None = None
    region: str | None = Field(default=None, min_length=1, max_length=100)
    required_tags: list[str] = Field(default_factory=list)
    tag_weight_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=5, ge=1, le=20)
    min_candidates: int = Field(default=10, ge=1, le=50)


class DestinationFeatureSnapshot(BaseModel):
    cosine_sim: float
    tag_match_count: int
    budget_delta: int | None
    region_match: bool


class DestinationRecommendationItem(BaseModel):
    destination_id: uuid.UUID
    destination: str
    country: str
    region: str | None
    budget_level: Level | None
    score: float
    rank_position: int
    features: DestinationFeatureSnapshot


class DestinationRecommendationResponse(BaseModel):
    query_text: str
    count: int
    used_relaxed_constraints: bool
    results: list[DestinationRecommendationItem]
