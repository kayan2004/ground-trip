"""Priority coverage for run_trip_planner's resume contract: a fresh call
with an unmet required bar returns PlannerNeedsInput; resuming with the
same thread_id + an answer drives the run to completion."""

import httpx
import pytest

from app.agent import graph as graph_module
from app.agent.planner import PlannerNeedsInput, PlannerResult, run_trip_planner
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.schemas.agent_runs import AgentRunCreate
from app.schemas.claude import ExtractedRequestFields, TravelProfile


def _profile(region: str = "Flexible") -> TravelProfile:
    return TravelProfile(
        region=region,
        budget_level="medium",
        tourism_level="medium",
        has_hiking=False,
        has_beach=False,
        culture_score=5.0,
        luxury_score=5.0,
        family_friendly=5.0,
        nightlife_level=5.0,
        avg_temp_peak=20.0,
    )


def _sequential_extractor(*results: ExtractedRequestFields):
    calls = {"n": 0}

    async def _fake(http_client, settings, *, prompt):
        index = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        return results[index]

    return _fake


async def _fake_synthesize(*args, **kwargs) -> str:
    return "synthesized response"


@pytest.mark.asyncio(loop_scope="session")
async def test_run_trip_planner_returns_needs_input_then_completes_on_resume(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "extract_request_fields",
        _sequential_extractor(
            ExtractedRequestFields(
                destination_name="Lisbon",
                location_query="Lisbon, Portugal",
                location_country_code="PT",
                travel_profile=_profile("Flexible"),
            ),
            ExtractedRequestFields(
                destination_name="Lisbon",
                location_query="Lisbon, Portugal",
                location_country_code="PT",
                travel_profile=_profile("Europe"),
            ),
        ),
    )
    monkeypatch.setattr(graph_module, "synthesize_trip_response", _fake_synthesize)

    async with httpx.AsyncClient() as client:
        tool_context = ToolContext(
            settings=get_settings(), resources={}, session=None, http_client=client
        )
        registry = ToolRegistry()

        first = await run_trip_planner(
            AgentRunCreate(prompt="A trip to Lisbon"),
            user_id=1,
            tool_registry=registry,
            tool_context=tool_context,
        )
        assert isinstance(first, PlannerNeedsInput)
        assert "region" in first.question.lower()

        second = await run_trip_planner(
            AgentRunCreate(
                prompt="A trip to Lisbon",
                thread_id=first.thread_id,
                clarification_answer="Somewhere in Europe, ideally Portugal",
            ),
            user_id=1,
            tool_registry=registry,
            tool_context=tool_context,
        )

    assert isinstance(second, PlannerResult)
    assert second.status in {"completed", "partial"}
