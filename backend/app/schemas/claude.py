from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "medium", "high"]


class TravelProfile(BaseModel):
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


class ExtractedRequestFields(BaseModel):
    destination_name: str | None = Field(default=None, max_length=120)
    location_query: str | None = Field(default=None, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    travel_profile: TravelProfile | None = None
