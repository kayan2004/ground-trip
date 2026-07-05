import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class FeedbackCreate(BaseModel):
    recommendation_id: int
    session_uuid: uuid.UUID
    verdict: Literal[1, -1]


class FeedbackRead(BaseModel):
    id: int
    recommendation_id: int
    session_uuid: uuid.UUID
    verdict: int
    channel: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
