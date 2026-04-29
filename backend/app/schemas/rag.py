from pydantic import BaseModel, Field


class RagSourceDocument(BaseModel):
    destination_name: str = Field(min_length=1, max_length=255)
    travel_style: str | None = Field(default=None, min_length=1, max_length=50)
    source_type: str = Field(min_length=1, max_length=100)
    source_title: str = Field(min_length=1, max_length=255)
    source_url: str | None = Field(default=None, min_length=1, max_length=1000)
    content: str = Field(min_length=1)


class RagDocumentChunk(BaseModel):
    destination_name: str
    travel_style: str | None
    source_type: str
    source_title: str
    source_url: str | None
    chunk_index: int
    content: str
