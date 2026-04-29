from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.classifier import TravelStylePredictionRequest
from app.schemas.tool_logs import ToolLogRead


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    travel_profile: TravelStylePredictionRequest | None = None


class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []

    model_config = ConfigDict(from_attributes=True)
