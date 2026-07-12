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
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> PlannerResult | PlannerNeedsInput:
    graph = build_trip_planner_graph()
    thread_id = payload.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
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
