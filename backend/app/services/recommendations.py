from pathlib import Path

import pandas as pd

from app.schemas.recommendations import (
    DestinationRecommendationItem,
    DestinationRecommendationRequest,
    DestinationRecommendationResponse,
)

DESTINATIONS_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "travel_destinations_labeled.csv"
)


def load_destination_catalog() -> pd.DataFrame:
    if not DESTINATIONS_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Destination catalog was not found at "
            f"{DESTINATIONS_DATASET_PATH}."
        )
    return pd.read_csv(DESTINATIONS_DATASET_PATH)


def recommend_destinations(
    catalog: pd.DataFrame,
    payload: DestinationRecommendationRequest,
) -> DestinationRecommendationResponse:
    filtered = catalog.loc[catalog["travel_style"] == payload.travel_style].copy()

    if payload.budget_level is not None:
        filtered = filtered.loc[filtered["budget_level"] == payload.budget_level]
    if payload.region is not None:
        filtered = filtered.loc[
            filtered["region"].str.casefold() == payload.region.casefold()
        ]
    if payload.has_hiking is not None:
        filtered = filtered.loc[filtered["has_hiking"] == int(payload.has_hiking)]
    if payload.has_beach is not None:
        filtered = filtered.loc[filtered["has_beach"] == int(payload.has_beach)]

    if filtered.empty:
        return DestinationRecommendationResponse(
            travel_style=payload.travel_style,
            count=0,
            results=[],
        )

    scored = filtered.assign(
        match_score=filtered.apply(_score_destination, axis=1, travel_style=payload.travel_style)
    )
    ranked = scored.sort_values("match_score", ascending=False).head(payload.limit)

    results = [
        DestinationRecommendationItem(
            destination=str(row["destination"]),
            country=str(row["country"]),
            region=str(row["region"]),
            budget_level=str(row["budget_level"]),
            tourism_level=str(row["tourism_level"]),
            travel_style=str(row["travel_style"]),
            has_hiking=bool(row["has_hiking"]),
            has_beach=bool(row["has_beach"]),
            culture_score=float(row["culture_score"]),
            luxury_score=float(row["luxury_score"]),
            family_friendly=float(row["family_friendly"]),
            nightlife_level=float(row["nightlife_level"]),
            avg_temp_peak=float(row["avg_temp_peak"]),
            match_score=round(float(row["match_score"]), 4),
        )
        for _, row in ranked.iterrows()
    ]

    return DestinationRecommendationResponse(
        travel_style=payload.travel_style,
        count=len(results),
        results=results,
    )


def _score_destination(row: pd.Series, *, travel_style: str) -> float:
    warm_weather_score = min(max(float(row["avg_temp_peak"]) / 35.0, 0.0), 1.0)
    low_budget_score = {"low": 1.0, "medium": 0.6, "high": 0.2}[str(row["budget_level"])]
    high_tourism_score = {"low": 0.3, "medium": 0.6, "high": 1.0}[str(row["tourism_level"])]

    if travel_style == "Adventure":
        return 0.6 * float(row["has_hiking"]) + 0.4 * float(row["culture_score"])
    if travel_style == "Relaxation":
        return 0.6 * float(row["has_beach"]) + 0.4 * warm_weather_score
    if travel_style == "Culture":
        return 0.7 * float(row["culture_score"]) + 0.3 * high_tourism_score
    if travel_style == "Budget":
        return 0.6 * low_budget_score + 0.4 * (1.0 - float(row["luxury_score"]))
    if travel_style == "Luxury":
        return 0.7 * float(row["luxury_score"]) + 0.3 * high_tourism_score
    if travel_style == "Family":
        return 0.7 * float(row["family_friendly"]) + 0.3 * float(row["has_beach"])
    return 0.0
