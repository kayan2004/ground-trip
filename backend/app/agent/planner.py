import uuid
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from app.agent.graph import TripPlannerRuntime, build_trip_planner_graph
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.schemas.agent_runs import AgentRunCreate


@dataclass(slots=True)
class ToolExecutionRecord:
    tool_name: str
    input_payload: str
    output_payload: str
    status: str


@dataclass(slots=True)
class PlannerResult:
    status: str
    response: str
    tool_logs: list[ToolExecutionRecord]
    recommended_destinations: list[dict[str, Any]]


@dataclass(slots=True)
class PlannerNeedsInput:
    thread_id: str
    question: str
    turn: int


async def run_trip_planner(
    payload: AgentRunCreate,
    *,
    user_id: int,
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> PlannerResult | PlannerNeedsInput:
    graph = build_trip_planner_graph()
    thread_id = payload.thread_id or str(uuid.uuid4())
    # Namespace the checkpointer's thread key by user so a resume can only ever
    # find a paused thread that belongs to the same user. The plain `thread_id`
    # (unprefixed) is what's returned to/round-tripped by the frontend - callers
    # never see or send the composite key.
    composite_thread_id = f"{user_id}:{thread_id}"
    config = {"configurable": {"thread_id": composite_thread_id}}
    runtime_context = TripPlannerRuntime(tool_registry=tool_registry, tool_context=tool_context)

    graph_input: Any
    if payload.thread_id is not None:
        graph_input = Command(resume=payload.clarification_answer)
    else:
        graph_input = {
            "prompt": payload.prompt,
            "travel_profile": payload.travel_profile,
            "destination_name": payload.destination_name,
            "location_query": payload.location_query,
            "location_country_code": payload.location_country_code,
            "retrieval_top_k": payload.retrieval_top_k,
        }

    final_state = await graph.ainvoke(graph_input, config=config, context=runtime_context)

    if "__interrupt__" in final_state:
        interrupt_payload = final_state["__interrupt__"][0].value
        return PlannerNeedsInput(
            thread_id=thread_id,
            question=interrupt_payload["question"],
            turn=interrupt_payload["turn"],
        )

    # The run reached a terminal state (no interrupt) - the caller persists
    # the resulting AgentRun row, and this thread will never be resumed
    # again, so drop its checkpoint chain from the in-memory checkpointer to
    # bound memory growth across completed runs.
    if graph.checkpointer is not None:
        await graph.checkpointer.adelete_thread(composite_thread_id)

    return PlannerResult(
        status=str(final_state["status"]),
        response=str(
            final_state.get("final_response") or "\n".join(final_state["response_sections"])
        ),
        tool_logs=[
            ToolExecutionRecord(
                tool_name=tool_log["tool_name"],
                input_payload=tool_log["input_payload"],
                output_payload=tool_log["output_payload"],
                status=tool_log["status"],
            )
            for tool_log in final_state["tool_logs"]
        ],
        recommended_destinations=list(final_state.get("recommended_destinations") or []),
    )
