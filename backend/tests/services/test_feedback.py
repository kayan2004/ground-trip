"""Priority 2 coverage: app/services/feedback.py."""

import uuid

import pytest
from sqlalchemy import insert

from app.db.models.agent_run import AgentRun
from app.db.models.recommendation import Recommendation
from app.schemas.feedback import FeedbackCreate
from app.services.feedback import RecommendationNotFoundError, submit_feedback


async def _make_recommendation(db_session, seeded_destinations, test_user) -> int:
    agent_run = AgentRun(
        user_id=test_user.id, prompt="p", response="r", status="completed"
    )
    db_session.add(agent_run)
    await db_session.flush()
    result = await db_session.execute(
        insert(Recommendation)
        .values(
            agent_run_id=agent_run.id,
            destination_id=seeded_destinations[0].id,
            rank_position=1,
            score=0.9,
            features={"cosine_sim": 0.9, "tag_match_count": 0, "budget_delta": None, "region_match": True},
        )
        .returning(Recommendation.id)
    )
    await db_session.commit()
    return result.scalar_one()


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_creates_row(db_session, seeded_destinations, test_user):
    recommendation_id = await _make_recommendation(db_session, seeded_destinations, test_user)
    session_uuid = uuid.uuid4()

    result = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=1),
    )

    assert result.recommendation_id == recommendation_id
    assert result.session_uuid == session_uuid
    assert result.verdict == 1
    assert result.channel == "web"


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_is_idempotent_on_recommendation_and_session(
    db_session, seeded_destinations, test_user
):
    recommendation_id = await _make_recommendation(db_session, seeded_destinations, test_user)
    session_uuid = uuid.uuid4()

    first = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=1),
    )
    second = await submit_feedback(
        db_session,
        FeedbackCreate(recommendation_id=recommendation_id, session_uuid=session_uuid, verdict=-1),
    )

    # Same row updated in place, not a second row inserted.
    assert first.id == second.id
    assert second.verdict == -1


@pytest.mark.asyncio(loop_scope="session")
async def test_submit_feedback_raises_for_unknown_recommendation(db_session):
    with pytest.raises(RecommendationNotFoundError):
        await submit_feedback(
            db_session,
            FeedbackCreate(recommendation_id=999999, session_uuid=uuid.uuid4(), verdict=1),
        )
