"""Serve-time ranker: feature-vector encoding + model loading + reranking.

The feature vector produced here MUST match the column order used when
training (see app/services/ranker_training.py and scripts/train_ranker.py) -
this module is the single shared definition both sides import, so train and
serve can never drift apart.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from joblib import load

MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "ranker" / "model.joblib"
)

# Fixed column order for the ranker's feature vector. Both training
# (ranker_training.py) and serving (destination_recommendations.py) build
# vectors through feature_vector() below, never by hand, so this order is
# the only place it's defined.
FEATURE_NAMES: list[str] = ["cosine_sim", "tag_match_count", "budget_delta", "region_match"]


def feature_vector(features: dict[str, Any]) -> list[float]:
    """Encodes a DestinationFeatureSnapshot-shaped dict into the ranker's input order.

    budget_delta is nullable (no budget constraint was requested, or the
    destination has no budget_level) - encoded as 0.0, a neutral "no signal"
    value. region_match is a bool, encoded as 1.0/0.0.
    """
    budget_delta = features.get("budget_delta")
    return [
        float(features["cosine_sim"]),
        float(features["tag_match_count"]),
        float(budget_delta) if budget_delta is not None else 0.0,
        1.0 if features["region_match"] else 0.0,
    ]


@lru_cache(maxsize=1)
def load_ranker_model() -> Any | None:
    """Loads the trained LGBMRanker, or None if no model has been trained yet."""
    if not MODEL_ARTIFACT_PATH.exists():
        return None
    return load(MODEL_ARTIFACT_PATH)


def rank_order(model: Any, feature_rows: list[list[float]]) -> list[int]:
    """Returns indices into feature_rows, best-predicted-relevance first."""
    scores = model.predict(feature_rows)
    return sorted(range(len(feature_rows)), key=lambda index: scores[index], reverse=True)
