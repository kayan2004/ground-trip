from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]
TravelStyle = Literal[
    "Adventure",
    "Relaxation",
    "Culture",
    "Budget",
    "Luxury",
    "Family",
]


class DestinationRecommendationRequest(BaseModel):
    travel_style: TravelStyle
    budget_level: Level | None = None
    region: str | None = Field(default=None, min_length=1, max_length=100)
    has_hiking: bool | None = None
    has_beach: bool | None = None
    limit: int = Field(default=5, ge=1, le=20)


class DestinationRecommendationItem(BaseModel):
    destination: str
    country: str
    region: str
    budget_level: Level
    tourism_level: Level
    travel_style: TravelStyle
    has_hiking: bool
    has_beach: bool
    culture_score: float
    luxury_score: float
    family_friendly: float
    nightlife_level: float
    avg_temp_peak: float
    match_score: float


class DestinationRecommendationResponse(BaseModel):
    travel_style: TravelStyle
    count: int
    results: list[DestinationRecommendationItem]
