from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings


@dataclass(slots=True)
class ToolContext:
    settings: Settings
    resources: Mapping[str, Any]
    session: AsyncSession | None = None
    http_client: httpx.AsyncClient | None = None
    # True when `settings` is a per-request BYOK copy (see app/core/byok.py)
    # rather than the shared server-default Settings singleton - controls
    # whether an LLM auth failure propagates as a 4xx (BYOK) or degrades
    # gracefully like any other tool failure (server key), see
    # app/agent/graph.py's extract_request_fields_node/synthesize_response_node.
    is_byok: bool = False


class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]

    @abstractmethod
    async def arun(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        raise NotImplementedError

