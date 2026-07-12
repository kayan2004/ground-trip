"""Priority coverage for the multi-turn clarification loop's graph-level
mechanics: which question gets asked and in what priority, that resuming
an answer actually overwrites the stale heuristic-guessed profile (not
just re-asks forever), and that the 3-round cap falls back to best-effort.
See docs/superpowers/specs/2026-07-12-clarification-loop-design.md.
"""

import uuid

import httpx
import pytest
from langgraph.types import Command

from app.agent import graph as graph_module
from app.agent.graph import TripPlannerRuntime, build_trip_planner_graph
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
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


def _initial_state(prompt: str) -> dict:
    return {
        "prompt": prompt,
        "travel_profile": None,
        "destination_name": None,
        "location_query": None,
        "location_country_code": None,
        "retrieval_top_k": 3,
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_asks_about_destination_first_when_both_signals_missing(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "extract_request_fields",
        _sequential_extractor(
            ExtractedRequestFields(
                destination_name=None,
                location_query=None,
                location_country_code=None,
                travel_profile=_profile("Flexible"),
            )
        ),
    )
    monkeypatch.setattr(graph_module, "synthesize_trip_response", _fake_synthesize)

    graph = build_trip_planner_graph()
    thread_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        runtime_context = TripPlannerRuntime(
            tool_registry=ToolRegistry(),
            tool_context=ToolContext(
                settings=get_settings(), resources={}, session=None, http_client=client
            ),
        )
        result = await graph.ainvoke(
            _initial_state("I want a relaxing trip"),
            config={"configurable": {"thread_id": thread_id}},
            context=runtime_context,
        )

    assert "__interrupt__" in result
    question = result["__interrupt__"][0].value["question"]
    assert "destination" in question.lower()


@pytest.mark.asyncio(loop_scope="session")
async def test_resume_answer_overwrites_stale_flexible_profile(monkeypatch):
    """Regression test for the merge-semantics bug: extract_request_fields_node
    merges `state.get("travel_profile") or extracted.travel_profile`
    (existing-state-wins), which is correct for a one-shot run but would
    silently discard every clarification answer on the profile/region path
    if clarify_missing_fields_node didn't clear the stale profile before
    looping back to re-extraction.
    """
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

    graph = build_trip_planner_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    async with httpx.AsyncClient() as client:
        runtime_context = TripPlannerRuntime(
            tool_registry=ToolRegistry(),
            tool_context=ToolContext(
                settings=get_settings(), resources={}, session=None, http_client=client
            ),
        )

        first = await graph.ainvoke(
            _initial_state("A trip to Lisbon"), config=config, context=runtime_context
        )
        assert "__interrupt__" in first
        assert "region" in first["__interrupt__"][0].value["question"].lower()

        final = await graph.ainvoke(
            Command(resume="Somewhere in Europe, ideally Portugal"),
            config=config,
            context=runtime_context,
        )

    assert "__interrupt__" not in final
    assert final["travel_profile"].region == "Europe"


@pytest.mark.asyncio(loop_scope="session")
async def test_max_turns_cap_proceeds_best_effort(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "extract_request_fields",
        _sequential_extractor(
            ExtractedRequestFields(
                destination_name=None,
                location_query=None,
                location_country_code=None,
                travel_profile=_profile("Flexible"),
            )
        ),
    )
    monkeypatch.setattr(graph_module, "synthesize_trip_response", _fake_synthesize)

    graph = build_trip_planner_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    async with httpx.AsyncClient() as client:
        runtime_context = TripPlannerRuntime(
            tool_registry=ToolRegistry(),
            tool_context=ToolContext(
                settings=get_settings(), resources={}, session=None, http_client=client
            ),
        )

        result = await graph.ainvoke(
            _initial_state("Take me somewhere"), config=config, context=runtime_context
        )
        for _ in range(3):
            assert "__interrupt__" in result
            result = await graph.ainvoke(
                Command(resume="still not sure"), config=config, context=runtime_context
            )

    assert "__interrupt__" not in result
    assert result["status"] in {"completed", "partial"}
    cap_logs = [
        log
        for log in result["tool_logs"]
        if log["tool_name"] == "clarification_loop" and "cap" in log["output_payload"].lower()
    ]
    assert len(cap_logs) == 1
