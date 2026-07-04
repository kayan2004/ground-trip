from pydantic import BaseModel, Field


class DestinationSeedEntry(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    country: str = Field(min_length=1, max_length=255)
    region: str | None = Field(default=None, max_length=100)
    wikivoyage_url: str = Field(min_length=1, max_length=1000)


class DestinationSeedManifest(BaseModel):
    version: int
    generated_at: str
    note: str | None = None
    destinations: list[DestinationSeedEntry]
