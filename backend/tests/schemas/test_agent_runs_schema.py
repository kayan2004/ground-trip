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
