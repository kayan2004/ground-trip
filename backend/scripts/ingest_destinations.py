import argparse
import asyncio
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db.session import create_db_engine, create_session_factory
from app.services.destination_ingestion import (
    DestinationIngestionSummary,
    ingest_destinations,
    load_seed_manifest,
)

ARTIFACT_DIR = BACKEND_DIR / "artifacts" / "destinations"
JSON_REPORT_PATH = ARTIFACT_DIR / "data_quality_report.json"
CSV_REPORT_PATH = ARTIFACT_DIR / "data_quality_report.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest the destination corpus.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only ingest the first N destinations from the seed manifest (smoke-test run).",
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    settings = get_settings()
    engine = create_db_engine(settings)
    session_factory = create_session_factory(engine)
    http_client = httpx.AsyncClient(follow_redirects=True)

    manifest = load_seed_manifest(settings.destination_seed_manifest_path)
    if args.limit is not None:
        manifest = manifest.model_copy(
            update={"destinations": manifest.destinations[: args.limit]}
        )

    try:
        async with session_factory() as session:
            summary = await ingest_destinations(session, http_client, settings, manifest=manifest)
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            _write_reports(summary)
            print(f"Ingested {summary.total_destinations} destinations.")
            print(f"Saved JSON report to {JSON_REPORT_PATH}")
            print(f"Saved CSV report to {CSV_REPORT_PATH}")
    finally:
        await http_client.aclose()
        await engine.dispose()


def _write_reports(summary: DestinationIngestionSummary) -> None:
    JSON_REPORT_PATH.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with CSV_REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["timestamp", summary.timestamp])
        writer.writerow(["total_destinations", summary.total_destinations])
        writer.writerow(["embedded_count", summary.embedded_count])
        writer.writerow(["skipped_embedding_count", summary.skipped_embedding_count])
        writer.writerow(["numbeo_lookup_available", summary.numbeo_lookup_available])
        writer.writerow(["opentripmap_configured", summary.opentripmap_configured])
        for region, count in sorted(summary.region_counts.items(), key=lambda kv: -kv[1]):
            writer.writerow([f"region_count:{region}", count])
        for field_name, rate in summary.missing_field_rates.items():
            writer.writerow([f"missing_field_rate:{field_name}", round(rate, 4)])
        for stat_name, value in summary.details_length_stats.items():
            writer.writerow([f"details_length:{stat_name}", round(value, 1)])
        for source, count in summary.sources_failed_counts.items():
            writer.writerow([f"sources_failed:{source}", count])


if __name__ == "__main__":
    asyncio.run(main())
