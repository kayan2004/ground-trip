"""Recall@k / MRR / NDCG@k evaluation harness for the destination recommender.

Analogous to scripts/evaluate_rag.py, but for recommend_destinations()
(structured pre-filter + pgvector cosine re-rank) rather than RAG chunk
retrieval - and computes real IR metrics (recall@k, MRR, NDCG@k), which
evaluate_rag.py itself does not (it only checks a single-query binary "did
any expected destination appear in the top-5" hit).

Ground truth is binary relevance: a query's `expected_destinations` list
names destinations considered relevant; nothing else is. Hand-written
(data/recommendation_eval_queries.json), same convention as
data/rag_eval_queries.json. A mix of free-text-only queries (pure cosine
similarity, like RAG) and queries that also set budget_level/region
(exercising the SQL pre-filter + relax-fallback, the recommender's actual
differentiator from RAG).

Runs against cosine order only (RANKER_ENABLED=false, the real production
default right now) - the LightGBM ranker's own quality is tracked
separately (NDCG in artifacts/ranker/model_metadata.json, against its own
synthetic bootstrap heuristic, not against this hand-labeled ground truth).
"""

import asyncio
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.schemas.recommendations import DestinationRecommendationRequest
from app.services.destination_recommendations import recommend_destinations

QUERY_FIXTURES_PATH = BACKEND_DIR / "data" / "recommendation_eval_queries.json"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts" / "recommendations"
JSON_REPORT_PATH = ARTIFACTS_DIR / "recommendation_eval.json"
CSV_REPORT_PATH = ARTIFACTS_DIR / "recommendation_eval.csv"

# limit=10 (not production's limit=3) so recall@k/NDCG@k can be reported at
# several k values from one run. min_candidates=5, deliberately lower than
# limit - the strict-filter trigger for relax-fallback is `len(rows) <
# min_candidates`, so a narrow-but-legitimate region+budget combination with,
# say, 8 real candidates still gets evaluated on its own (narrower) merits
# instead of being silently widened to a full-corpus cosine ranking that
# would defeat the point of a region/budget-scoped query.
RESULT_LIMIT = 10
MIN_CANDIDATES = 5
RECALL_KS = (3, 5, 10)
NDCG_KS = (3, 5, 10)


def _recall_at_k(returned: list[str], expected: list[str], k: int) -> float:
    if not expected:
        return 0.0
    top_k = set(returned[:k])
    hits = sum(1 for name in expected if name in top_k)
    return hits / len(expected)


def _reciprocal_rank(returned: list[str], expected: list[str]) -> float:
    expected_set = set(expected)
    for index, name in enumerate(returned, start=1):
        if name in expected_set:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(returned: list[str], expected: list[str], k: int) -> float:
    expected_set = set(expected)
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, name in enumerate(returned[:k], start=1)
        if name in expected_set
    )
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


async def main() -> None:
    settings = get_settings()
    if settings.ranker_enabled:
        print(
            "WARNING: ranker_enabled=True in current settings - this harness "
            "measures cosine order specifically. Results below will reflect "
            "ranker reordering, not the cosine baseline this script reports on."
        )

    queries = _load_queries()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(follow_redirects=True)

    try:
        async with session_factory() as session:
            detailed_results: list[dict[str, Any]] = []

            for index, item in enumerate(queries, start=1):
                query_text = item["query"]
                expected_destinations = item["expected_destinations"]
                budget_level = item.get("budget_level")
                region = item.get("region")
                print(f"[{index}/{len(queries)}] Evaluating: {query_text}")

                payload = DestinationRecommendationRequest(
                    query_text=query_text,
                    budget_level=budget_level,
                    region=region,
                    limit=RESULT_LIMIT,
                    min_candidates=MIN_CANDIDATES,
                )
                response = await recommend_destinations(session, http_client, settings, payload)
                returned_destinations = [result.destination for result in response.results]

                metrics: dict[str, float] = {
                    f"recall_at_{k}": _recall_at_k(returned_destinations, expected_destinations, k)
                    for k in RECALL_KS
                }
                metrics["mrr"] = _reciprocal_rank(returned_destinations, expected_destinations)
                metrics.update(
                    {
                        f"ndcg_at_{k}": _ndcg_at_k(returned_destinations, expected_destinations, k)
                        for k in NDCG_KS
                    }
                )

                detailed_results.append(
                    {
                        "query": query_text,
                        "budget_level": budget_level,
                        "region": region,
                        "expected_destinations": expected_destinations,
                        "returned_destinations": returned_destinations,
                        "used_relaxed_constraints": response.used_relaxed_constraints,
                        **metrics,
                    }
                )

            _write_reports(detailed_results)
            print(f"Saved JSON report to {JSON_REPORT_PATH}")
            print(f"Saved CSV report to {CSV_REPORT_PATH}")
    finally:
        await http_client.aclose()
        await engine.dispose()


def _load_queries() -> list[dict[str, Any]]:
    return json.loads(QUERY_FIXTURES_PATH.read_text(encoding="utf-8"))


def _metric_keys() -> list[str]:
    return [f"recall_at_{k}" for k in RECALL_KS] + ["mrr"] + [f"ndcg_at_{k}" for k in NDCG_KS]


def _write_reports(detailed_results: list[dict[str, Any]]) -> None:
    metric_keys = _metric_keys()
    aggregate = (
        {
            key: sum(item[key] for item in detailed_results) / len(detailed_results)
            for key in metric_keys
        }
        if detailed_results
        else {}
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "query_count": len(detailed_results),
                "result_limit": RESULT_LIMIT,
                "min_candidates": MIN_CANDIDATES,
                "aggregate_metrics": aggregate,
                "results": detailed_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with CSV_REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "query",
            "budget_level",
            "region",
            "expected_destinations",
            "returned_destinations",
            "used_relaxed_constraints",
            *metric_keys,
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in detailed_results:
            writer.writerow(
                {
                    "query": item["query"],
                    "budget_level": item["budget_level"],
                    "region": item["region"],
                    "expected_destinations": ", ".join(item["expected_destinations"]),
                    "returned_destinations": ", ".join(item["returned_destinations"]),
                    "used_relaxed_constraints": item["used_relaxed_constraints"],
                    **{key: item[key] for key in metric_keys},
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
