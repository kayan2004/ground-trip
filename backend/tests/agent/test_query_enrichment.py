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
