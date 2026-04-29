from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from app.schemas.classifier import (
    TravelStylePredictionRequest,
    TravelStylePredictionResponse,
)

MODEL_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "artifacts" / "ml" / "best_model.joblib"
)


def load_travel_style_model() -> Any:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Travel style model artifact was not found at {MODEL_ARTIFACT_PATH}."
        )
    return load(MODEL_ARTIFACT_PATH)


def predict_travel_style(
    model: Any,
    payload: TravelStylePredictionRequest,
) -> TravelStylePredictionResponse:
    features = pd.DataFrame(
        [
            {
                "region": payload.region,
                "budget_level": payload.budget_level,
                "tourism_level": payload.tourism_level,
                "has_hiking": int(payload.has_hiking),
                "has_beach": int(payload.has_beach),
                "culture_score": payload.culture_score,
                "luxury_score": payload.luxury_score,
                "family_friendly": payload.family_friendly,
                "nightlife_level": payload.nightlife_level,
                "avg_temp_peak": payload.avg_temp_peak,
            }
        ]
    )
    predicted_style = str(model.predict(features)[0])

    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(features)[0]
        probabilities = {
            label: float(probability)
            for label, probability in sorted(
                zip(model.classes_, raw_probabilities, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    return TravelStylePredictionResponse(
        predicted_style=predicted_style,
        probabilities=probabilities,
    )
