from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
import httpx

from app.agent.tools.registry import build_default_tool_registry
from .config import get_settings
from app.db.session import create_db_engine, create_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    db_engine = create_db_engine(settings)
    db_session_factory = create_session_factory(db_engine)
    http_client = httpx.AsyncClient(follow_redirects=True)
    tool_registry = build_default_tool_registry()

    app.state.settings = settings

    app.state.resources = {
        "db_engine": db_engine,
        "db_session_factory": db_session_factory,
        "http_client": http_client,
        "tool_registry": tool_registry,
    }

    yield

    await http_client.aclose()
    await db_engine.dispose()
    app.state.resources.clear()
