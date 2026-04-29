from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]


class TravelStylePredictionRequest(BaseModel):
    region: str = Field(min_length=1, max_length=100)
    budget_level: Level
    tourism_level: Level
    has_hiking: bool
    has_beach: bool
    culture_score: float = Field(ge=0, le=10)
    luxury_score: float = Field(ge=0, le=10)
    family_friendly: float = Field(ge=0, le=10)
    nightlife_level: float = Field(ge=0, le=10)
    avg_temp_peak: float = Field(ge=-50, le=70)


class TravelStylePredictionResponse(BaseModel):
    predicted_style: str
    probabilities: dict[str, float]
