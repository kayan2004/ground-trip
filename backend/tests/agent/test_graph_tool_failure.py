"""Priority 6 coverage: a failing tool degrades a node to
tool_logs status=failed + graph status=partial, without raising -
app/agent/graph.py's core "tool failures are data, not exceptions" pattern
(see CLAUDE.md's "Conventions to follow when editing").
"""

import pytest
from pydantic import BaseModel

from app.agent.graph import retrieve_context_node
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

    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
        "tool_registry": registry,
        "tool_context": context,
    }

    result = await retrieve_context_node(state)

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
    state = {
        "prompt": "a trip to nowhere",
        "retrieval_top_k": 5,
        "status": "completed",
        "response_sections": [],
        "tool_logs": [],
        "tool_registry": None,
        "tool_context": None,
    }

    result = await retrieve_context_node(state)

    assert result["status"] == "partial"
    assert result["tool_logs"][0]["status"] == "failed"
