import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.agent_run import AgentRun
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate
from app.services.classifier import predict_travel_style
from app.services.tool_logs import create_tool_log


async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
    *,
    travel_style_model: Any | None = None,
) -> AgentRun:
    prompt = payload.prompt.strip()
    response = f"Saved prompt for later planning: {prompt}"
    run_status = "completed"
    tool_input_payload = prompt
    tool_output_payload = "Classifier skipped because no structured travel profile was provided."
    tool_status = "skipped"

    if payload.travel_profile is not None:
        tool_input_payload = json.dumps(payload.travel_profile.model_dump())
        if travel_style_model is None:
            response = "Travel profile was saved, but the classifier model is unavailable right now."
            run_status = "partial"
            tool_output_payload = "Classifier could not run because the model was not loaded."
            tool_status = "failed"
        else:
            prediction = predict_travel_style(travel_style_model, payload.travel_profile)
            response = (
                f"Predicted travel style: {prediction.predicted_style} "
                f"for prompt: {prompt}"
            )
            tool_output_payload = json.dumps(prediction.model_dump())
            tool_status = "completed"

    agent_run = AgentRun(
        user_id=current_user.id,
        prompt=prompt,
        response=response,
        status=run_status,
    )
    session.add(agent_run)
    await session.commit()
    await session.refresh(agent_run)
    await create_tool_log(
        session,
        agent_run,
        tool_name="travel_style_classifier",
        input_payload=tool_input_payload,
        output_payload=tool_output_payload,
        status=tool_status,
    )
    await session.refresh(agent_run, attribute_names=["tool_logs"])
    return agent_run
