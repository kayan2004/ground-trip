import uuid

from pydantic import ValidationError
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.destination import Destination
from app.db.models.recommendation import Recommendation
from app.schemas.recommendation_read import RecommendationRead
from app.schemas.recommendations import DestinationFeatureSnapshot


async def persist_recommendation_slate(
    session: AsyncSession,
    agent_run_id: int,
    recommended_destinations: list[dict],
) -> list[Recommendation]:
    if not recommended_destinations:
        return []

    # Recommendation.destination_id's ForeignKey("destinations.id") can never
    # resolve through SQLAlchemy's ORM metadata lookup - Destination lives on
    # a separate DeclarativeBase/MetaData (DestinationCorpusBase), and
    # session.add()-triggered flushes (or an ORM bulk-insert passed a params
    # list) need to resolve every FK target of the flushed mapper's table for
    # internal dependency-sort bookkeeping, even when that table isn't part
    # of the current flush. A single Core INSERT with all rows baked into
    # .values([...]) sidesteps that flush machinery entirely while still
    # returning fully-populated ORM Recommendation instances via RETURNING.
    values = [
        {
            "agent_run_id": agent_run_id,
            "destination_id": uuid.UUID(item["destination_id"]),
            "rank_position": item["rank_position"],
            "score": item["score"],
            "features": item["features"],
        }
        for item in recommended_destinations
    ]
    statement = insert(Recommendation).values(values).returning(Recommendation)
    result = await session.execute(statement)
    await session.commit()
    return list(result.scalars().all())


async def get_recommendations_for_agent_run(
    session: AsyncSession,
    agent_run_id: int,
) -> list[RecommendationRead]:
    statement = (
        select(Recommendation, Destination.name, Destination.country)
        .join(Destination, Recommendation.destination_id == Destination.id)
        .where(Recommendation.agent_run_id == agent_run_id)
        .where(Recommendation.deleted_at.is_(None))
        .order_by(Recommendation.rank_position)
    )
    rows = (await session.execute(statement)).all()
    return [
        RecommendationRead(
            id=recommendation.id,
            destination_id=recommendation.destination_id,
            destination_name=name,
            country=country,
            rank_position=recommendation.rank_position,
            score=recommendation.score,
            features=_parse_features(recommendation.features),
            created_at=recommendation.created_at,
        )
        for recommendation, name, country in rows
    ]


def _parse_features(raw_features: dict) -> DestinationFeatureSnapshot | None:
    """Best-effort parse of the stored features JSONB into the typed
    snapshot. A row written before this shape existed, or edited by hand,
    shouldn't crash the whole recommendations list - degrade to None so the
    API caller can render "ranking details unavailable" instead of a 500.
    """
    try:
        return DestinationFeatureSnapshot.model_validate(raw_features)
    except ValidationError:
        return None
