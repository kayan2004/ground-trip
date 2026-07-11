from pydantic import BaseModel, Field

from app.schemas.classifier import TravelStylePredictionRequest


class ExtractedRequestFields(BaseModel):
    destination_name: str | None = Field(default=None, max_length=120)
    location_query: str | None = Field(default=None, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    travel_profile: TravelStylePredictionRequest | None = None
