from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.claude import TravelProfile
from app.schemas.recommendation_read import RecommendationRead
from app.schemas.tool_logs import ToolLogRead


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    travel_profile: TravelProfile | None = None
    destination_name: str | None = Field(default=None, min_length=2, max_length=120)
    location_query: str | None = Field(default=None, min_length=2, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    retrieval_top_k: int = Field(default=3, ge=1, le=8)
    thread_id: str | None = None
    clarification_answer: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _validate_resume_pairing(self) -> "AgentRunCreate":
        if self.thread_id is not None and self.clarification_answer is None:
            raise ValueError("clarification_answer is required when thread_id is set.")
        if self.thread_id is None and self.clarification_answer is not None:
            raise ValueError("clarification_answer must not be set without a thread_id.")
        return self


class AgentRunNeedsInput(BaseModel):
    status: Literal["needs_input"] = "needs_input"
    thread_id: str
    question: str
    turn: int


class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []
    recommendations: list[RecommendationRead] = []

    model_config = ConfigDict(from_attributes=True)
