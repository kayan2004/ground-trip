from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.byok import BYOKOverride, BYOKValidationError, build_byok_settings
from app.core.rate_limit import agent_run_ip_rate_limiter, agent_run_user_rate_limiter
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate, AgentRunRead
from app.schemas.tool_logs import ToolLogRead
from app.services.agent_runs import create_agent_run
from app.services.llm_providers import LLMAuthenticationError
from app.services.recommendation_persistence import get_recommendations_for_agent_run
from app.agent.tools.base import ToolContext

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


@router.post("", response_model=AgentRunRead, status_code=status.HTTP_201_CREATED)
async def create_agent_run_route(
    payload: AgentRunCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
    x_llm_api_key: str | None = Header(default=None, alias="X-LLM-API-Key"),
) -> AgentRunRead:
    client_ip = request.client.host if request.client else "unknown"
    if not await agent_run_ip_rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests from this address - please slow down.")
    if not await agent_run_user_rate_limiter.check(current_user.id):
        raise HTTPException(status_code=429, detail="Too many requests - please slow down.")

    tool_registry = request.app.state.resources.get("tool_registry")
    http_client = request.app.state.resources.get("http_client")

    is_byok = bool(x_llm_api_key)
    if x_llm_api_key:
        if not payload.llm_provider or not payload.llm_model:
            raise HTTPException(
                status_code=400,
                detail="llm_provider and llm_model are required when X-LLM-API-Key is set.",
            )
        try:
            trip_settings = build_byok_settings(
                request.app.state.settings,
                BYOKOverride(
                    provider=payload.llm_provider,
                    model=payload.llm_model,
                    api_key=x_llm_api_key,
                ),
            )
        except BYOKValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        trip_settings = request.app.state.settings

    tool_context = ToolContext(
        settings=trip_settings,
        resources=request.app.state.resources,
        session=session,
        http_client=http_client,
        is_byok=is_byok,
    )

    try:
        agent_run = await create_agent_run(
            session,
            current_user,
            payload,
            tool_registry=tool_registry,
            tool_context=tool_context,
        )
    except LLMAuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail="The provided API key was rejected by the provider.",
        ) from exc

    recommendations = await get_recommendations_for_agent_run(session, agent_run.id)
    return AgentRunRead(
        id=agent_run.id,
        user_id=agent_run.user_id,
        prompt=agent_run.prompt,
        response=agent_run.response,
        status=agent_run.status,
        created_at=agent_run.created_at,
        tool_logs=[ToolLogRead.model_validate(log) for log in agent_run.tool_logs],
        recommendations=recommendations,
    )
