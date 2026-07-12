# Multi-turn Clarification Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trip-planning LangGraph pipeline pause and ask the user a targeted follow-up question (via LangGraph `interrupt`/`Command.resume`) when a destination anchor or a real region preference is missing, instead of silently proceeding on a guessed/incomplete profile.

**Architecture:** Split `TripPlannerState` (checkpointed) from tool/DB runtime (LangGraph's `Runtime[Context]`, injected fresh per call, never checkpointed). Add a `clarify_missing_fields` node that checks a required bar (destination anchor OR non-"Flexible" region), asks one question via `interrupt()`, and loops back to re-extraction on resume. Compile the graph with an in-memory checkpointer. `run_trip_planner`/`create_agent_run`/the `/agent-runs` route thread a `thread_id` through so a paused run returns `{status: "needs_input", ...}` with no DB row written until the run actually completes.

**Tech Stack:** FastAPI, LangGraph 1.1.10 (`langgraph.types.interrupt`/`Command`, `langgraph.runtime.Runtime`, `langgraph.checkpoint.memory.InMemorySaver`), Pydantic v2, pytest + pytest-asyncio, React 19 + TypeScript, Vitest.

## Global Constraints

- Backend commands run from `backend/` via `uv run ...`. Frontend commands run from `frontend/` via `npm run ...`.
- Every external HTTP boundary (LLM provider, Voyage embeddings) must be mocked in tests — never a live call. Use `monkeypatch.setattr` on the `app.agent.graph` module's imported names (`extract_request_fields`, `synthesize_trip_response`), since `graph.py` imports them via `from app.services.llm import ...` — patching `app.services.llm.extract_request_fields` directly would not affect the already-bound name inside `graph.py`.
- Tests requiring Postgres use the `db_session`/`engine`/`test_user`/`auth_headers`/`seeded_destinations` fixtures already defined in `backend/tests/conftest.py`. Do not invent new DB fixtures.
- `TravelProfile` requires all ten fields whenever it's constructed (no partial profiles) — see `backend/app/schemas/claude.py`.
- Cap clarification rounds at exactly 3 (`REQUIRED_SIGNAL_MAX_TURNS = 3` in `app/agent/graph.py`).
- No schema/migration changes — no DB row is persisted for a run until it actually completes (spec: `docs/superpowers/specs/2026-07-12-clarification-loop-design.md`).
- Follow this repo's existing "tool failures are data, not exceptions" convention (CLAUDE.md) — do not raise from graph nodes for expected-missing-data cases.

---

### Task 1: Split runtime (tool/DB) dependencies out of checkpointed graph state

**Files:**
- Modify: `backend/app/agent/graph.py`
- Modify: `backend/app/agent/planner.py`
- Modify: `backend/tests/agent/test_graph_tool_failure.py`

**Interfaces:**
- Produces: `TripPlannerRuntime` dataclass (`tool_registry: ToolRegistry | None`, `tool_context: ToolContext | None`) in `app/agent/graph.py`, used by every node's `runtime: Runtime[TripPlannerRuntime]` parameter and by `planner.py`'s `graph.ainvoke(..., context=TripPlannerRuntime(...))`.
- Produces: `TripPlannerState` with `tool_registry`/`tool_context` keys **removed** (no longer part of graph state).

This is a mechanical refactor: no new behavior yet, all five existing nodes must keep working exactly as before, just reading their tool dependencies from `runtime.context` instead of from `state`.

- [ ] **Step 1: Run the existing test suite to confirm the starting point is green**

Run: `uv run pytest tests/agent/test_graph_tool_failure.py -v`
Expected: 2 passed

- [ ] **Step 2: Add `TripPlannerRuntime` and remove runtime fields from `TripPlannerState`**

In `backend/app/agent/graph.py`, add near the top (after existing imports) and edit the `TypedDict`:

```python
from dataclasses import dataclass


@dataclass
class TripPlannerRuntime:
    tool_registry: ToolRegistry | None
    tool_context: ToolContext | None


class TripPlannerState(TypedDict):
    prompt: str
    travel_profile: NotRequired[TravelProfile | None]
    destination_name: NotRequired[str | None]
    location_query: NotRequired[str | None]
    location_country_code: NotRequired[str | None]
    retrieval_top_k: int
    status: str
    response_sections: list[str]
    recommended_destinations: NotRequired[list[dict[str, Any]]]
    final_response: NotRequired[str | None]
    tool_logs: list[dict[str, str]]
```

(Removes the old `tool_registry: NotRequired[ToolRegistry | None]` and `tool_context: NotRequired[ToolContext | None]` lines.)

- [ ] **Step 3: Update every node signature to accept `runtime: Runtime[TripPlannerRuntime]` and read from it**

Add `from langgraph.runtime import Runtime` to the imports.

In `extract_request_fields_node`, change the signature and the two `tool_context = state.get("tool_context")` style reads:

```python
async def extract_request_fields_node(
    state: TripPlannerState, runtime: Runtime[TripPlannerRuntime]
) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    response_sections = list(state["response_sections"])
    tool_context = runtime.context.tool_context
```

Apply the same pattern to `retrieve_context_node`, `recommend_destinations_node`, and `live_conditions_node`: add the `runtime: Runtime[TripPlannerRuntime]` parameter, and replace every `state.get("tool_registry")` with `runtime.context.tool_registry` and every `state.get("tool_context")` with `runtime.context.tool_context`.

`synthesize_response_node` also reads `tool_context = state.get("tool_context")` — same change: add the `runtime` parameter, read `runtime.context.tool_context`.

`initialize_trip_state` takes no tool dependencies — leave its signature as `(state: TripPlannerState) -> TripPlannerState` (LangGraph introspects each node's signature independently; a `runtime` parameter is optional per-node).

- [ ] **Step 4: Wire `context_schema` and a checkpointer into `build_trip_planner_graph`**

```python
from langgraph.checkpoint.memory import InMemorySaver


@lru_cache(maxsize=1)
def build_trip_planner_graph():
    checkpointer = InMemorySaver()
    graph = StateGraph(TripPlannerState, context_schema=TripPlannerRuntime)
    graph.add_node("initialize", initialize_trip_state)
    graph.add_node("extract_request_fields", extract_request_fields_node)
    graph.add_node("recommend_destinations", recommend_destinations_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("live_conditions", live_conditions_node)
    graph.add_node("synthesize_response", synthesize_response_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "extract_request_fields")
    graph.add_edge("extract_request_fields", "recommend_destinations")
    graph.add_edge("recommend_destinations", "retrieve_context")
    graph.add_edge("retrieve_context", "live_conditions")
    graph.add_edge("live_conditions", "synthesize_response")
    graph.add_edge("synthesize_response", END)

    return graph.compile(checkpointer=checkpointer)
```

(`clarify_missing_fields` is added in Task 2 — this step only adds the checkpointer/context_schema plumbing so Task 1 stays independently testable.)

- [ ] **Step 5: Update `planner.py` to pass `context=` and a `thread_id` config**

In `backend/app/agent/planner.py`, change the `run_trip_planner` body:

```python
import uuid

from app.agent.graph import TripPlannerRuntime, build_trip_planner_graph
...

async def run_trip_planner(
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None,
    tool_context: ToolContext | None,
) -> PlannerResult:
    graph = build_trip_planner_graph()
    thread_id = str(uuid.uuid4())
    final_state = await graph.ainvoke(
        {
            "prompt": payload.prompt,
            "travel_profile": payload.travel_profile,
            "destination_name": payload.destination_name,
            "location_query": payload.location_query,
            "location_country_code": payload.location_country_code,
            "retrieval_top_k": payload.retrieval_top_k,
        },
        config={"configurable": {"thread_id": thread_id}},
        context=TripPlannerRuntime(tool_registry=tool_registry, tool_context=tool_context),
    )
    ... (rest of the function body — the PlannerResult construction — is unchanged)
```

(This is a placeholder `thread_id` for now — Task 5 replaces it with real resume support once `AgentRunCreate.thread_id` exists.)

- [ ] **Step 6: Update the existing test to construct a `Runtime` directly**

Replace the full contents of `backend/tests/agent/test_graph_tool_failure.py`:

```python
"""Priority 6 coverage: a failing tool degrades a node to
tool_logs status=failed + graph status=partial, without raising -
app/agent/graph.py's core "tool failures are data, not exceptions" pattern
(see CLAUDE.md's "Conventions to follow when editing").
"""

import pytest
from langgraph.runtime import Runtime
from pydantic import BaseModel

from app.agent.graph import TripPlannerRuntime, retrieve_context_node
from app.agent.tools.base import BaseTool, ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings


class _AlwaysFailsTool(BaseTool):
    name = "destination_context_retriever"
    description = "Test double that always raises."
    input_model = BaseModel

    async def arun(self, payload: BaseModel, context: ToolContext) -> BaseModel:
        raise RuntimeError("simulated RAG retrieval failure")


@pytest.mark.asyncio(loop_scope="session")
async def test_tool_failure_produces_failed_tool_log_and_partial_status():
    registry = ToolRegistry()
    registry.register(_AlwaysFailsTool())
    context = ToolContext(settings=get_settings(), resources={}, session=None, http_client=None)
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=registry, tool_context=context))

    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    result = await retrieve_context_node(state, runtime)

    assert result["status"] == "partial"
    failed_logs = [log for log in result["tool_logs"] if log["status"] == "failed"]
    assert len(failed_logs) == 1
    assert failed_logs[0]["tool_name"] == "destination_context_retriever"
    assert "simulated RAG retrieval failure" in failed_logs[0]["output_payload"]


@pytest.mark.asyncio(loop_scope="session")
async def test_missing_tool_runtime_also_degrades_gracefully():
    """tool_registry=None (shared services unavailable) is the other real
    "can't run this tool" path this node handles - also data, not an
    exception.
    """
    runtime = Runtime(context=TripPlannerRuntime(tool_registry=None, tool_context=None))
    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
    }

    result = await retrieve_context_node(state, runtime)

    assert result["status"] == "partial"
    assert result["tool_logs"][0]["status"] == "failed"
```

- [ ] **Step 7: Run tests to verify the refactor is behavior-preserving**

Run: `uv run pytest tests/agent/test_graph_tool_failure.py -v`
Expected: 2 passed

- [ ] **Step 8: Commit**

```bash
git add app/agent/graph.py app/agent/planner.py tests/agent/test_graph_tool_failure.py
git commit -m "refactor(agent): move tool/DB deps into LangGraph Runtime context

Splits checkpointable graph state from non-serializable runtime
dependencies (httpx client, DB session) ahead of adding a checkpointer
for the clarification loop."
```

---

### Task 2: Add the `clarify_missing_fields` node and interrupt/resume loop

**Files:**
- Modify: `backend/app/agent/graph.py`
- Create: `backend/tests/agent/test_clarify_missing_fields.py`

**Interfaces:**
- Consumes: `TripPlannerState`, `TripPlannerRuntime`, `build_trip_planner_graph()` from Task 1.
- Produces: `TripPlannerState` gains `clarification_turn: NotRequired[int]`, `clarification_qa: NotRequired[list[dict[str, str]]]`, `clarification_outcome: NotRequired[str]`. New node `clarify_missing_fields_node`. `REQUIRED_SIGNAL_MAX_TURNS = 3` module constant, used by Task 7's tests indirectly (not imported directly elsewhere).

- [ ] **Step 1: Add the new state fields and constant**

In `backend/app/agent/graph.py`, add to `TripPlannerState`:

```python
    clarification_turn: NotRequired[int]
    clarification_qa: NotRequired[list[dict[str, str]]]
    clarification_outcome: NotRequired[str]
```

And seed them in `initialize_trip_state`'s return dict:

```python
def initialize_trip_state(state: TripPlannerState) -> TripPlannerState:
    prompt = state["prompt"].strip()
    return {
        **state,
        "prompt": prompt,
        "status": "completed",
        "response_sections": [f"Prompt: {prompt}"],
        "recommended_destinations": [],
        "final_response": None,
        "tool_logs": [],
        "clarification_turn": 0,
        "clarification_qa": [],
    }
```

Add the constant near the top of the module (after imports):

```python
REQUIRED_SIGNAL_MAX_TURNS = 3
```

- [ ] **Step 2: Write the failing tests**

Create `backend/tests/agent/test_clarify_missing_fields.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/agent/test_clarify_missing_fields.py -v`
Expected: FAIL — `ImportError` or `KeyError: 'clarify_missing_fields'`/no interrupt raised (the node doesn't exist yet, graph isn't wired to it).

- [ ] **Step 4: Implement `clarify_missing_fields_node` and wire it into the graph**

In `backend/app/agent/graph.py`, add `from langgraph.types import interrupt` to imports, then add these module-level functions (place after `extract_request_fields_node`, before `retrieve_context_node`):

```python
def _needs_destination_signal(state: TripPlannerState) -> bool:
    return state.get("destination_name") is None and state.get("location_query") is None


def _needs_region_signal(state: TripPlannerState) -> bool:
    travel_profile = state.get("travel_profile")
    return travel_profile is None or travel_profile.region == "Flexible"


async def clarify_missing_fields_node(state: TripPlannerState) -> TripPlannerState:
    tool_logs = list(state["tool_logs"])
    turn = state.get("clarification_turn", 0)

    needs_destination = _needs_destination_signal(state)
    needs_region = _needs_region_signal(state)

    if not (needs_destination or needs_region):
        return {"tool_logs": tool_logs, "clarification_outcome": "satisfied"}

    if turn >= REQUIRED_SIGNAL_MAX_TURNS:
        tool_logs.append(
            {
                "tool_name": "clarification_loop",
                "input_payload": state["prompt"],
                "output_payload": (
                    f"Clarification cap ({REQUIRED_SIGNAL_MAX_TURNS} rounds) reached; "
                    "proceeding with best-effort fields."
                ),
                "status": "skipped",
            }
        )
        return {"tool_logs": tool_logs, "clarification_outcome": "cap_reached"}

    question = (
        "Which destination or region are you considering, or would you like me to "
        "recommend one based on your trip style?"
        if needs_destination
        else "Do you have a specific region or country in mind, or should I keep the search worldwide?"
    )

    tool_logs.append(
        {
            "tool_name": "clarification_loop",
            "input_payload": state["prompt"],
            "output_payload": question,
            "status": "needs_input",
        }
    )

    answer = interrupt({"question": question, "turn": turn})

    tool_logs.append(
        {
            "tool_name": "clarification_loop",
            "input_payload": question,
            "output_payload": f"User answered: {answer}",
            "status": "completed",
        }
    )

    updates: dict[str, Any] = {
        "prompt": f"{state['prompt']}\n\nAdditional detail from the traveler: {answer}",
        "clarification_turn": turn + 1,
        "clarification_qa": [
            *state.get("clarification_qa", []),
            {"question": question, "answer": answer},
        ],
        "tool_logs": tool_logs,
        "clarification_outcome": "answered",
    }
    if needs_region:
        updates["travel_profile"] = None

    return updates


def _route_after_clarification(state: TripPlannerState) -> str:
    return "retry" if state.get("clarification_outcome") == "answered" else "proceed"
```

Then update `build_trip_planner_graph()` to register the node and rewire edges:

```python
    graph.add_node("initialize", initialize_trip_state)
    graph.add_node("extract_request_fields", extract_request_fields_node)
    graph.add_node("clarify_missing_fields", clarify_missing_fields_node)
    graph.add_node("recommend_destinations", recommend_destinations_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("live_conditions", live_conditions_node)
    graph.add_node("synthesize_response", synthesize_response_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "extract_request_fields")
    graph.add_edge("extract_request_fields", "clarify_missing_fields")
    graph.add_conditional_edges(
        "clarify_missing_fields",
        _route_after_clarification,
        {"retry": "extract_request_fields", "proceed": "recommend_destinations"},
    )
    graph.add_edge("recommend_destinations", "retrieve_context")
    graph.add_edge("retrieve_context", "live_conditions")
    graph.add_edge("live_conditions", "synthesize_response")
    graph.add_edge("synthesize_response", END)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/agent/test_clarify_missing_fields.py -v`
Expected: 3 passed

- [ ] **Step 6: Run the full agent test directory to check for regressions**

Run: `uv run pytest tests/agent/ -v`
Expected: 5 passed (2 from Task 1 + 3 new)

- [ ] **Step 7: Commit**

```bash
git add app/agent/graph.py tests/agent/test_clarify_missing_fields.py
git commit -m "feat(agent): add interrupt/resume clarification loop

Asks one targeted question (destination anchor first, then region) via
LangGraph interrupt() when the required signal bar isn't met, capped at
3 rounds. Clears the stale heuristic-guessed travel_profile before
looping back to re-extraction so resumed answers actually take effect."
```

---

### Task 3: Enrich the recommender query with clarified travel-style signals

**Files:**
- Modify: `backend/app/agent/graph.py`
- Create: `backend/tests/agent/test_query_enrichment.py`

**Interfaces:**
- Produces: `_build_enriched_query_text(prompt: str, travel_profile: TravelProfile | None) -> str` in `app/agent/graph.py`, used by `recommend_destinations_node`.

This makes the activity/style fields (`has_hiking`, `has_beach`, `culture_score`, `luxury_score`, `family_friendly`, `nightlife_level`) — which the clarification loop works to fill in — actually change the recommendation, since `required_tags` is inert against the current 5-tag corpus (see spec Section 3).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/agent/test_query_enrichment.py`:

```python
"""Unit coverage for _build_enriched_query_text: turns travel_profile
activity/style signals into phrases appended to the recommender's
query_text, since destinations were embedded from Wikivoyage prose and
the query is embedded directly (destination_recommendations.py:28-31).
"""

from app.agent.graph import _build_enriched_query_text
from app.schemas.claude import TravelProfile


def _profile(**overrides) -> TravelProfile:
    base = dict(
        region="Flexible",
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
    base.update(overrides)
    return TravelProfile(**base)


def test_enrichment_adds_phrases_for_strong_signals():
    profile = _profile(has_hiking=True, culture_score=9.0)
    enriched = _build_enriched_query_text("a relaxing trip", profile)
    assert enriched.startswith("a relaxing trip")
    assert "enjoys hiking and outdoor trails" in enriched
    assert "seeks rich cultural and historical experiences" in enriched


def test_enrichment_leaves_prompt_unchanged_without_profile():
    assert _build_enriched_query_text("a relaxing trip", None) == "a relaxing trip"


def test_enrichment_leaves_prompt_unchanged_when_no_signals_are_strong():
    profile = _profile()
    assert _build_enriched_query_text("a relaxing trip", profile) == "a relaxing trip"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agent/test_query_enrichment.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_enriched_query_text'`

- [ ] **Step 3: Implement `_build_enriched_query_text` and use it in `recommend_destinations_node`**

In `backend/app/agent/graph.py`, add (near the top, after `REQUIRED_SIGNAL_MAX_TURNS`):

```python
_ENRICHMENT_PHRASES: list[tuple[str, "Callable[[TravelProfile], bool]"]] = [
    ("enjoys hiking and outdoor trails", lambda profile: profile.has_hiking),
    ("wants beach and coastal experiences", lambda profile: profile.has_beach),
    ("seeks rich cultural and historical experiences", lambda profile: profile.culture_score >= 7.0),
    ("prefers luxury and upscale travel", lambda profile: profile.luxury_score >= 7.0),
    ("traveling family-friendly", lambda profile: profile.family_friendly >= 7.0),
    ("wants vibrant nightlife", lambda profile: profile.nightlife_level >= 7.0),
]


def _build_enriched_query_text(prompt: str, travel_profile: TravelProfile | None) -> str:
    if travel_profile is None:
        return prompt
    phrases = [phrase for phrase, predicate in _ENRICHMENT_PHRASES if predicate(travel_profile)]
    if not phrases:
        return prompt
    return f"{prompt} ({'; '.join(phrases)})"
```

Add `from collections.abc import Callable` to imports (the string-quoted type hint above avoids needing it at runtime, but importing keeps it valid for tooling — add it alongside the existing `from typing import Any, NotRequired, TypedDict`).

In `recommend_destinations_node`, change:

```python
    recommendation_input = DestinationRecommendationRequest(
        query_text=state["prompt"],
```

to:

```python
    recommendation_input = DestinationRecommendationRequest(
        query_text=_build_enriched_query_text(state["prompt"], travel_profile),
```

(`travel_profile` is already a local variable in this function — `travel_profile = state.get("travel_profile")`, a few lines above.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agent/test_query_enrichment.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full agent test directory to check for regressions**

Run: `uv run pytest tests/agent/ -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add app/agent/graph.py tests/agent/test_query_enrichment.py
git commit -m "feat(agent): enrich recommender query with clarified travel-style signals

Appends activity/style phrases (hiking, beach, culture, luxury, family,
nightlife) to the embedded query text so information gathered by the
clarification loop actually changes the cosine ranking - required_tags
is inert against the current 5-tag regional-cluster corpus."
```

---

### Task 4: Extend `AgentRunCreate`/add `AgentRunNeedsInput` schema

**Files:**
- Modify: `backend/app/schemas/agent_runs.py`
- Create: `backend/tests/schemas/__init__.py`
- Create: `backend/tests/schemas/test_agent_runs_schema.py`

**Interfaces:**
- Produces: `AgentRunCreate.thread_id: str | None`, `AgentRunCreate.clarification_answer: str | None`, with paired-validation. `AgentRunNeedsInput {status: Literal["needs_input"], thread_id: str, question: str, turn: int}`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/schemas/__init__.py` (empty file).

Create `backend/tests/schemas/test_agent_runs_schema.py`:

```python
"""Priority coverage for AgentRunCreate's fresh-vs-resume validation:
thread_id and clarification_answer must be set together or not at all."""

import pytest
from pydantic import ValidationError

from app.schemas.agent_runs import AgentRunCreate


def test_resume_requires_clarification_answer():
    with pytest.raises(ValidationError):
        AgentRunCreate(prompt="a trip", thread_id="abc")


def test_clarification_answer_requires_thread_id():
    with pytest.raises(ValidationError):
        AgentRunCreate(prompt="a trip", clarification_answer="Europe")


def test_fresh_request_without_thread_id_is_valid():
    payload = AgentRunCreate(prompt="a trip")
    assert payload.thread_id is None
    assert payload.clarification_answer is None


def test_resume_request_with_both_fields_is_valid():
    payload = AgentRunCreate(prompt="a trip", thread_id="abc", clarification_answer="Europe")
    assert payload.thread_id == "abc"
    assert payload.clarification_answer == "Europe"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/schemas/test_agent_runs_schema.py -v`
Expected: FAIL — `TypeError: AgentRunCreate() got an unexpected keyword argument 'thread_id'`

- [ ] **Step 3: Implement the schema changes**

Replace the contents of `backend/app/schemas/agent_runs.py`:

```python
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.claude import TravelProfile
from app.schemas.recommendation_read import RecommendationRead
from app.schemas.tool_logs import ToolLogRead


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    travel_profile: TravelProfile | None = None
    destination_name: str | None = Field(default=None, min_length=2, max_length=120)
    location_query: str | None = Field(default=None, min_length=2, max_length=120)
    location_country_code: str | None = Field(default=None, min_length=2, max_length=2)
    retrieval_top_k: int = Field(default=3, ge=1, le=8)
    thread_id: str | None = None
    clarification_answer: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _validate_resume_pairing(self) -> "AgentRunCreate":
        if self.thread_id is not None and self.clarification_answer is None:
            raise ValueError("clarification_answer is required when thread_id is set.")
        if self.thread_id is None and self.clarification_answer is not None:
            raise ValueError("clarification_answer must not be set without a thread_id.")
        return self


class AgentRunNeedsInput(BaseModel):
    status: Literal["needs_input"] = "needs_input"
    thread_id: str
    question: str
    turn: int


class AgentRunRead(BaseModel):
    id: int
    user_id: int
    prompt: str
    response: str
    status: str
    created_at: datetime
    tool_logs: list[ToolLogRead] = []
    recommendations: list[RecommendationRead] = []

    model_config = ConfigDict(from_attributes=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/schemas/test_agent_runs_schema.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add app/schemas/agent_runs.py tests/schemas/__init__.py tests/schemas/test_agent_runs_schema.py
git commit -m "feat(schemas): add resume fields to AgentRunCreate + AgentRunNeedsInput

thread_id/clarification_answer must be set together (resume call) or
neither (fresh call). AgentRunNeedsInput is the response shape for a
paused clarification-loop run."
```

---

### Task 5: Wire `run_trip_planner` to support resume via `Command`

**Files:**
- Modify: `backend/app/agent/planner.py`
- Create: `backend/tests/agent/test_planner_clarification_loop.py`

**Interfaces:**
- Consumes: `AgentRunCreate.thread_id`/`.clarification_answer` (Task 4), `TripPlannerRuntime`/`build_trip_planner_graph()` (Task 1), the `__interrupt__` key convention from Task 2's graph.
- Produces: `PlannerNeedsInput {thread_id: str, question: str, turn: int}`. `run_trip_planner(...) -> PlannerResult | PlannerNeedsInput`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/agent/test_planner_clarification_loop.py`:

```python
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
            tool_registry=registry,
            tool_context=tool_context,
        )

    assert isinstance(second, PlannerResult)
    assert second.status in {"completed", "partial"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/agent/test_planner_clarification_loop.py -v`
Expected: FAIL — `AttributeError`/`ImportError: cannot import name 'PlannerNeedsInput'`

- [ ] **Step 3: Implement the planner changes**

Replace the contents of `backend/app/agent/planner.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/agent/test_planner_clarification_loop.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the full agent test directory to check for regressions**

Run: `uv run pytest tests/agent/ -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add app/agent/planner.py tests/agent/test_planner_clarification_loop.py
git commit -m "feat(agent): support resuming a paused run via Command(resume=...)

run_trip_planner now returns PlannerNeedsInput when the graph interrupts,
and resumes an existing thread via Command when the caller supplies
thread_id + clarification_answer."
```

---

### Task 6: `create_agent_run` — no DB row until the run completes

**Files:**
- Modify: `backend/app/services/agent_runs.py`
- Create: `backend/tests/services/test_agent_runs_clarification.py`

**Interfaces:**
- Consumes: `PlannerNeedsInput`/`run_trip_planner` (Task 5).
- Produces: `create_agent_run(...) -> AgentRun | PlannerNeedsInput`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/services/test_agent_runs_clarification.py`:

```python
"""Priority coverage: an interrupted run persists nothing (no AgentRun
row, no schema/migration change needed) until it actually completes -
see docs/superpowers/specs/2026-07-12-clarification-loop-design.md
Section 4."""

import httpx
import pytest
from sqlalchemy import select

from app.agent import graph as graph_module
from app.agent.planner import PlannerNeedsInput
from app.agent.tools.base import ToolContext
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.db.models.agent_run import AgentRun
from app.schemas.agent_runs import AgentRunCreate
from app.schemas.claude import ExtractedRequestFields, TravelProfile
from app.services.agent_runs import create_agent_run


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
async def test_interrupted_run_persists_no_agent_run_row_until_completion(
    monkeypatch, db_session, test_user
):
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
            settings=get_settings(), resources={}, session=db_session, http_client=client
        )
        registry = ToolRegistry()

        first = await create_agent_run(
            db_session,
            test_user,
            AgentRunCreate(prompt="A trip to Lisbon"),
            tool_registry=registry,
            tool_context=tool_context,
        )
        assert isinstance(first, PlannerNeedsInput)

        no_rows_yet = (await db_session.execute(select(AgentRun))).scalars().all()
        assert no_rows_yet == []

        second = await create_agent_run(
            db_session,
            test_user,
            AgentRunCreate(
                prompt="A trip to Lisbon",
                thread_id=first.thread_id,
                clarification_answer="Somewhere in Europe, ideally Portugal",
            ),
            tool_registry=registry,
            tool_context=tool_context,
        )

    assert isinstance(second, AgentRun)
    persisted_rows = (await db_session.execute(select(AgentRun))).scalars().all()
    assert len(persisted_rows) == 1
    assert persisted_rows[0].id == second.id
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/services/test_agent_runs_clarification.py -v`
Expected: FAIL — `AssertionError` (a row gets persisted on the first, interrupted call) or `AttributeError` on `PlannerNeedsInput` handling.

- [ ] **Step 3: Implement the branching in `create_agent_run`**

In `backend/app/services/agent_runs.py`, add the import and the early-return branch:

```python
from app.agent.planner import PlannerNeedsInput, run_trip_planner
...

async def create_agent_run(
    session: AsyncSession,
    current_user: User,
    payload: AgentRunCreate,
    *,
    tool_registry: ToolRegistry | None = None,
    tool_context: ToolContext | None = None,
) -> AgentRun | PlannerNeedsInput:
    planner_result = await run_trip_planner(
        payload,
        tool_registry=tool_registry,
        tool_context=tool_context,
    )

    if isinstance(planner_result, PlannerNeedsInput):
        return planner_result

    agent_run = AgentRun(
        user_id=current_user.id,
        prompt=payload.prompt.strip(),
        response=planner_result.response,
        status=planner_result.status,
    )
    session.add(agent_run)
    await session.commit()
    await session.refresh(agent_run)
    # ... rest of the function body (tool_logs loop, recommendation
    # persistence, Discord delivery, final refresh/return) is unchanged.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/services/test_agent_runs_clarification.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the full services test directory to check for regressions**

Run: `uv run pytest tests/services/ -v`
Expected: all passed (existing suite + 1 new)

- [ ] **Step 6: Commit**

```bash
git add app/services/agent_runs.py tests/services/test_agent_runs_clarification.py
git commit -m "feat(services): short-circuit agent-run persistence on interrupt

create_agent_run now returns PlannerNeedsInput without touching the DB
when the graph pauses for clarification - no AgentRun row, tool_logs, or
Discord delivery happen until the run actually completes."
```

---

### Task 7: Route wiring — `needs_input`/resume HTTP contract

**Files:**
- Modify: `backend/app/api/routes/agent_runs.py`
- Create: `backend/tests/api/test_agent_runs_clarification.py`

**Interfaces:**
- Consumes: `create_agent_run` (Task 6), `AgentRunNeedsInput`/`AgentRunCreate` (Task 4).
- Produces: `POST /agent-runs` returns `AgentRunRead` (201) or `AgentRunNeedsInput` (200).

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/api/test_agent_runs_clarification.py`:

```python
"""Priority coverage for the /agent-runs HTTP contract added by the
clarification loop: interrupt -> resume -> complete, and the 3-round cap
falling back to a completed run. Uses httpx.ASGITransport directly (same
reasoning as test_auth.py: the real lifespan builds live DB/HTTP
resources this test doesn't want) but manually populates
app.state.resources/app.state.settings, since this route (unlike auth)
reads them.
"""

import httpx
import pytest
import pytest_asyncio

from app.agent import graph as graph_module
from app.agent.tools.registry import ToolRegistry
from app.core.config import get_settings
from app.db.dependencies import get_db_session
from app.schemas.claude import ExtractedRequestFields, TravelProfile
from main import app


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


@pytest_asyncio.fixture(scope="function", loop_scope="session")
async def api_client(engine):
    from app.db.session import create_session_factory

    factory = create_session_factory(engine)

    async def override_get_db_session():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.state.settings = get_settings()
    async with httpx.AsyncClient() as http_client:
        app.state.resources = {"tool_registry": ToolRegistry(), "http_client": http_client}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    app.dependency_overrides.clear()
    app.state.resources = {}


@pytest.mark.asyncio(loop_scope="session")
async def test_clarification_interrupt_resume_complete_round_trip(
    monkeypatch, api_client, auth_headers
):
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

    first = await api_client.post(
        "/agent-runs", json={"prompt": "A trip to Lisbon"}, headers=auth_headers
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["status"] == "needs_input"
    assert "region" in first_body["question"].lower()
    thread_id = first_body["thread_id"]

    second = await api_client.post(
        "/agent-runs",
        json={
            "prompt": "A trip to Lisbon",
            "thread_id": thread_id,
            "clarification_answer": "Somewhere in Europe, ideally Portugal",
        },
        headers=auth_headers,
    )
    assert second.status_code == 201
    second_body = second.json()
    assert second_body["status"] in {"completed", "partial"}
    assert second_body["id"]


@pytest.mark.asyncio(loop_scope="session")
async def test_clarification_max_turns_cap_falls_back_to_completed(
    monkeypatch, api_client, auth_headers
):
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

    response = await api_client.post(
        "/agent-runs", json={"prompt": "Take me somewhere"}, headers=auth_headers
    )
    assert response.status_code == 200
    thread_id = response.json()["thread_id"]

    for _ in range(2):
        response = await api_client.post(
            "/agent-runs",
            json={
                "prompt": "Take me somewhere",
                "thread_id": thread_id,
                "clarification_answer": "still not sure",
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "needs_input"

    final = await api_client.post(
        "/agent-runs",
        json={
            "prompt": "Take me somewhere",
            "thread_id": thread_id,
            "clarification_answer": "still not sure",
        },
        headers=auth_headers,
    )
    assert final.status_code == 201
    final_body = final.json()
    assert final_body["status"] in {"completed", "partial"}
    cap_logs = [
        log
        for log in final_body["tool_logs"]
        if log["tool_name"] == "clarification_loop" and "cap" in log["output_payload"].lower()
    ]
    assert len(cap_logs) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/api/test_agent_runs_clarification.py -v`
Expected: FAIL — 500/422 (route doesn't yet accept `thread_id`/return `needs_input`)

- [ ] **Step 3: Implement the route changes**

Replace the contents of `backend/app/api/routes/agent_runs.py`:

```python
from fastapi import APIRouter, Depends, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.planner import PlannerNeedsInput
from app.agent.tools.base import ToolContext
from app.api.dependencies.auth import get_current_user
from app.db.dependencies import get_db_session
from app.db.models.user import User
from app.schemas.agent_runs import AgentRunCreate, AgentRunNeedsInput, AgentRunRead
from app.schemas.tool_logs import ToolLogRead
from app.services.agent_runs import create_agent_run
from app.services.recommendation_persistence import get_recommendations_for_agent_run

router = APIRouter(prefix="/agent-runs", tags=["agent-runs"])


@router.post("", response_model=AgentRunRead | AgentRunNeedsInput)
async def create_agent_run_route(
    payload: AgentRunCreate,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> AgentRunRead | AgentRunNeedsInput:
    tool_registry = request.app.state.resources.get("tool_registry")
    http_client = request.app.state.resources.get("http_client")
    tool_context = ToolContext(
        settings=request.app.state.settings,
        resources=request.app.state.resources,
        session=session,
        http_client=http_client,
    )
    result = await create_agent_run(
        session,
        current_user,
        payload,
        tool_registry=tool_registry,
        tool_context=tool_context,
    )

    if isinstance(result, PlannerNeedsInput):
        response.status_code = status.HTTP_200_OK
        return AgentRunNeedsInput(
            thread_id=result.thread_id, question=result.question, turn=result.turn
        )

    agent_run = result
    response.status_code = status.HTTP_201_CREATED
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/api/test_agent_runs_clarification.py -v`
Expected: 2 passed

- [ ] **Step 5: Run the full backend test suite to check for regressions**

Run: `uv run pytest -v`
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add app/api/routes/agent_runs.py tests/api/test_agent_runs_clarification.py
git commit -m "feat(api): needs_input/resume contract on POST /agent-runs

Same endpoint handles both fresh requests and resumes (discriminated by
thread_id/clarification_answer in the body). Returns 200 + needs_input
while the clarification loop is still asking, 201 + the normal
AgentRunRead once it completes."
```

---

### Task 8: Frontend types and API client

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/lib/api.ts`

**Interfaces:**
- Produces: `AgentRunNeedsInput` type. `PlannerRequest.thread_id?: string`, `.clarification_answer?: string`. `createAgentRun(...): Promise<AgentRunRead | AgentRunNeedsInput>`.

- [ ] **Step 1: Update `types.ts`**

In `frontend/src/types.ts`, change `PlannerRequest` and add `AgentRunNeedsInput`:

```typescript
export interface PlannerRequest {
  prompt: string
  retrieval_top_k: number
  thread_id?: string
  clarification_answer?: string
}

export interface AgentRunNeedsInput {
  status: 'needs_input'
  thread_id: string
  question: string
  turn: number
}
```

(Add `AgentRunNeedsInput` after the existing `AgentRunRead` interface.)

- [ ] **Step 2: Update `api.ts`'s `createAgentRun` return type**

In `frontend/src/lib/api.ts`, change the import and function signature:

```typescript
import type {
  AgentRunNeedsInput,
  AgentRunRead,
  FeedbackRead,
  FeedbackVerdict,
  PlannerRequest,
  TokenResponse,
  UserRead,
} from '../types'
...

export async function createAgentRun(
  token: string,
  payload: PlannerRequest,
): Promise<AgentRunRead | AgentRunNeedsInput> {
  return request<AgentRunRead | AgentRunNeedsInput>('/agent-runs', {
    method: 'POST',
    token,
    body: payload,
  })
}
```

- [ ] **Step 3: Run the frontend typecheck to verify it compiles**

Run: `npm run build` (from `frontend/`)
Expected: build succeeds with no type errors (nothing calls `createAgentRun` with the new fields yet, and the wider return type is compatible with existing usage until Task 9 narrows it)

- [ ] **Step 4: Commit**

```bash
git add src/types.ts src/lib/api.ts
git commit -m "feat(frontend): add clarification-loop types to the API client

PlannerRequest gains optional thread_id/clarification_answer for resume
calls; createAgentRun's return type now includes the needs_input variant."
```

---

### Task 9: Frontend chat-style clarification UI

**Files:**
- Modify: `frontend/src/App.tsx`

**Interfaces:**
- Consumes: `AgentRunNeedsInput`, updated `PlannerRequest`/`createAgentRun` (Task 8).

- [ ] **Step 1: Add clarification state**

In `frontend/src/App.tsx`, alongside the existing `useState` calls (near line 92, after `const [result, setResult] = useState<AgentRunRead | null>(null)`):

```typescript
  const [clarification, setClarification] = useState<
    { threadId: string; question: string; turn: number } | null
  >(null)
  const [clarificationAnswer, setClarificationAnswer] = useState('')
```

Update the import line to include the new type:

```typescript
import type { AgentRunRead, AuthMode, FeedbackVerdict, SessionState } from './types'
```
becomes
```typescript
import type {
  AgentRunNeedsInput,
  AgentRunRead,
  AuthMode,
  FeedbackVerdict,
  SessionState,
} from './types'
```

- [ ] **Step 2: Update `handlePlanSubmit` to branch on `needs_input`**

Replace the body of `handlePlanSubmit` (around line 192):

```typescript
  async function handlePlanSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!session) {
      setPlannerError('Please log in first.')
      navigateTo('login')
      setView('login')
      return
    }

    setPlannerPending(true)
    setPlannerError('')

    try {
      const agentRun = await createAgentRun(session.token, {
        prompt,
        retrieval_top_k: retrievalTopK,
      })
      if (agentRun.status === 'needs_input') {
        const needsInput = agentRun as AgentRunNeedsInput
        setClarification({
          threadId: needsInput.thread_id,
          question: needsInput.question,
          turn: needsInput.turn,
        })
        setClarificationAnswer('')
      } else {
        setResult(agentRun)
        setClarification(null)
      }
    } catch (error) {
      setPlannerError(
        error instanceof ApiError ? error.message : 'Trip planning failed.',
      )
    } finally {
      setPlannerPending(false)
    }
  }

  async function handleClarificationSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!session || !clarification) {
      return
    }

    setPlannerPending(true)
    setPlannerError('')

    try {
      const agentRun = await createAgentRun(session.token, {
        prompt,
        retrieval_top_k: retrievalTopK,
        thread_id: clarification.threadId,
        clarification_answer: clarificationAnswer,
      })
      if (agentRun.status === 'needs_input') {
        const needsInput = agentRun as AgentRunNeedsInput
        setClarification({
          threadId: needsInput.thread_id,
          question: needsInput.question,
          turn: needsInput.turn,
        })
        setClarificationAnswer('')
      } else {
        setResult(agentRun)
        setClarification(null)
      }
    } catch (error) {
      setPlannerError(
        error instanceof ApiError ? error.message : 'Trip planning failed.',
      )
    } finally {
      setPlannerPending(false)
    }
  }
```

- [ ] **Step 3: Render the clarification block, hiding the form while it's active**

In the JSX (around line 358-398), wrap the existing `<form className="form-grid" onSubmit={handlePlanSubmit}>...</form>` with a conditional, adding the clarification block as the alternate branch:

```tsx
        <div className="gt-panel planner-inner">
          <div className="gt-panel-header">
            <div>
              <p className="gt-eyebrow">Planner</p>
              <h2>Ask for a trip recommendation</h2>
            </div>
            <span className="gt-pill gt-pill--positive">ready</span>
          </div>

          {clarification ? (
            <div className="form-grid">
              <span className="gt-pill gt-pill--brass">Still forming your trip plan…</span>
              <p className="gt-panel gt-panel--paper prompt-preview">{prompt}</p>
              <p className="markdown-strong">{clarification.question}</p>
              <form className="form-grid" onSubmit={handleClarificationSubmit}>
                <label className="gt-field">
                  <span>Your answer</span>
                  <input
                    className="gt-input"
                    type="text"
                    value={clarificationAnswer}
                    onChange={(event) => setClarificationAnswer(event.target.value)}
                    required
                  />
                </label>
                {plannerError ? (
                  <p className="error-text" role="alert">
                    {plannerError}
                  </p>
                ) : null}
                <button
                  type="submit"
                  className="gt-btn gt-btn--primary"
                  disabled={plannerPending}
                >
                  {plannerPending ? 'Sending…' : 'Answer'}
                </button>
              </form>
            </div>
          ) : (
            <form className="form-grid" onSubmit={handlePlanSubmit}>
              <label className="gt-field">
                <span>Prompt</span>
                <textarea
                  className="gt-textarea"
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  rows={7}
                  required
                />
              </label>
              <label className="gt-field gt-compact-field">
                <span>RAG top K</span>
                <input
                  className="gt-input"
                  type="number"
                  min={1}
                  max={8}
                  value={retrievalTopK}
                  onChange={(event) => setRetrievalTopK(Number(event.target.value))}
                />
              </label>
              {plannerError ? (
                <p className="error-text" role="alert">
                  {plannerError}
                </p>
              ) : null}
              <button type="submit" className="gt-btn gt-btn--primary" disabled={plannerPending}>
                {plannerPending ? 'Planning trip…' : 'Run agent'}
              </button>
            </form>
          )}
        </div>
      </section>
```

- [ ] **Step 4: Run the frontend build/typecheck**

Run: `npm run build` (from `frontend/`)
Expected: build succeeds with no type errors

- [ ] **Step 5: Manually verify in the browser**

Start the dev server and drive the flow: submit a vague prompt (e.g. "take me somewhere"), confirm the clarification question renders with the "Still forming your trip plan…" badge and the main form is hidden, answer it, confirm either another question or the final results render, and that `result` only populates once the run actually completes.

Run: `npm run dev` (from `frontend/`), then open the app in the browser preview and exercise the flow described above.
Expected: clarification question appears after submitting a vague prompt; answering it either asks a follow-up or shows the completed trip plan; no console errors.

- [ ] **Step 6: Run existing frontend tests to check for regressions**

Run: `npm run test` (from `frontend/`)
Expected: all passed (`JsonPayload.test.tsx`, `WhyThisPick.test.tsx` — neither touches `App.tsx`, so this is a smoke check that nothing else broke)

- [ ] **Step 7: Commit**

```bash
git add src/App.tsx
git commit -m "feat(frontend): chat-style clarification UI for the trip planner

Renders the agent's follow-up question as a message with a single answer
field when a run needs_input, hiding the main prompt form so it reads as
a continuation of the same request thread rather than a new submission."
```

---

## Self-Review

**Spec coverage:** Section 1 (Runtime split, checkpointer, interrupt/resume) → Tasks 1-2. Section 2 (required bar, question priority, merge-fix) → Task 2. Section 3 (query enrichment) → Task 3. Section 4 (API contract, no-DB-until-completion, logging) → Tasks 4, 5, 6, 7. Section 5 (frontend) → Tasks 8-9. Section 6 (tests) → a dedicated test file/step in every backend task; frontend covered by build + manual verification (no existing App.tsx test harness to extend proportionately for this scope). Out-of-scope items (confidence scoring, real activity-tag taxonomy, Postgres checkpointer, placeholder AgentRun rows) are not implemented anywhere in this plan.

**Placeholder scan:** No TBD/TODO markers; every step has complete, runnable code.

**Type consistency:** `TripPlannerRuntime(tool_registry, tool_context)` used identically in Tasks 1, 2, 5. `PlannerNeedsInput(thread_id, question, turn)` used identically in Tasks 5, 6, 7. `AgentRunNeedsInput(status, thread_id, question, turn)` used identically in Tasks 4, 7. `clarification_outcome` values (`"satisfied"`, `"cap_reached"`, `"answered"`) set in Task 2's node and read only by Task 2's own router — no drift elsewhere. `_build_enriched_query_text` signature matches between Task 3's definition and its call site.
