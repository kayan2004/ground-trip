from collections.abc import Sequence

import httpx
from sqlalchemy import Float, cast, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models.destination import Destination
from app.schemas.recommendations import (
    DestinationFeatureSnapshot,
    DestinationRecommendationItem,
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)
from app.services.voyage_embeddings import embed_texts

BUDGET_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
NO_REGION_PREFERENCE = "flexible"


async def recommend_destinations(
    session: AsyncSession,
    http_client: httpx.AsyncClient,
    settings: Settings,
    payload: DestinationRecommendationRequest,
) -> DestinationRecommendationResponse:
    query_text = payload.query_text.strip()
    query_embedding = (
        await embed_texts(http_client, settings, [query_text], input_type="query")
    )[0]

    fetch_limit = max(payload.limit, payload.min_candidates)

    rows = await _fetch_ranked_candidates(
        session, payload, query_embedding, fetch_limit=fetch_limit, apply_filters=True
    )
    used_relaxed_constraints = False
    if len(rows) < payload.min_candidates:
        rows = await _fetch_ranked_candidates(
            session, payload, query_embedding, fetch_limit=fetch_limit, apply_filters=False
        )
        used_relaxed_constraints = True

    ranked = rows[: payload.limit]
    results = [
        DestinationRecommendationItem(
            destination_id=destination.id,
            destination=destination.name,
            country=destination.country,
            region=destination.region,
            budget_level=destination.budget_level,
            score=round(1.0 - distance, 4),
            rank_position=index + 1,
            features=DestinationFeatureSnapshot(
                cosine_sim=round(1.0 - distance, 4),
                tag_match_count=_count_matching_tags(
                    destination.tags, payload.required_tags, payload.tag_weight_threshold
                ),
                budget_delta=_budget_delta(destination.budget_level, payload.budget_level),
                region_match=_region_match(destination.region, payload.region),
            ),
        )
        for index, (destination, distance) in enumerate(ranked)
    ]

    return DestinationRecommendationResponse(
        query_text=query_text,
        count=len(results),
        used_relaxed_constraints=used_relaxed_constraints,
        results=results,
    )


async def _fetch_ranked_candidates(
    session: AsyncSession,
    payload: DestinationRecommendationRequest,
    query_embedding: list[float],
    *,
    fetch_limit: int,
    apply_filters: bool,
) -> list[tuple[Destination, float]]:
    distance_expr = Destination.embedding.cosine_distance(query_embedding)
    statement = (
        select(Destination, distance_expr.label("distance"))
        .where(Destination.deleted_at.is_(None))
        .where(Destination.embedding.is_not(None))
    )

    if apply_filters:
        if payload.budget_level is not None:
            ceiling = BUDGET_ORDER[payload.budget_level]
            allowed_levels = [
                level for level, order in BUDGET_ORDER.items() if order <= ceiling
            ]
            statement = statement.where(
                or_(
                    Destination.budget_level.in_(allowed_levels),
                    Destination.budget_level.is_(None),
                )
            )

        if (
            payload.region is not None
            and payload.region.strip().casefold() != NO_REGION_PREFERENCE
        ):
            statement = statement.where(
                func.lower(Destination.region) == payload.region.strip().casefold()
            )

        for tag in payload.required_tags:
            statement = statement.where(
                cast(Destination.tags[tag].astext, Float) >= payload.tag_weight_threshold
            )

    statement = statement.order_by(distance_expr).limit(fetch_limit)
    rows = (await session.execute(statement)).all()
    return [(row[0], float(row[1])) for row in rows]


def _count_matching_tags(
    tags: dict,
    required_tags: Sequence[str],
    threshold: float,
) -> int:
    return sum(1 for tag in required_tags if float(tags.get(tag, 0.0)) >= threshold)


def _budget_delta(destination_budget: str | None, requested_ceiling: str | None) -> int | None:
    if destination_budget is None or requested_ceiling is None:
        return None
    return BUDGET_ORDER[destination_budget] - BUDGET_ORDER[requested_ceiling]


def _region_match(destination_region: str | None, requested_region: str | None) -> bool:
    if requested_region is None or requested_region.strip().casefold() == NO_REGION_PREFERENCE:
        return True
    if destination_region is None:
        return False
    return destination_region.casefold() == requested_region.strip().casefold()
