#!/usr/bin/env python3
"""
Load processed master CSV files into PostgreSQL.

Reads batch analytics outputs and bulk-loads them into:
  - traffic_detections
  - congestion_metrics
  - object_summary

Usage:
    python scripts/load_to_postgres.py
    python scripts/load_to_postgres.py --verbose

Prerequisites:
    docker compose -f docker/docker-compose.yml up -d
    python scripts/compute_metrics.py --master
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.db.connection import Base, SessionLocal, engine, get_database_url  # noqa: E402
from traffic_safety.db.models import CongestionMetric, ObjectSummary, TrafficDetection  # noqa: E402

logger = logging.getLogger("load_to_postgres")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DETECTIONS_CSV = PROJECT_ROOT / "data" / "processed" / "traffic_detections_master.csv"
DEFAULT_CONGESTION_CSV = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "data" / "processed" / "object_summary_master.csv"

DETECTION_COLUMNS = [
    "camera_id",
    "video_id",
    "timestamp",
    "frame_id",
    "object_type",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

CONGESTION_COLUMNS = [
    "camera_id",
    "video_id",
    "minute",
    "sampled_frame_count",
    "vehicle_count",
    "pedestrian_count",
    "avg_vehicles_per_frame",
    "avg_pedestrians_per_frame",
    "traffic_density",
    "congestion_level",
]

SUMMARY_COLUMNS = ["camera_id", "object_type", "count", "percentage"]


@dataclass(frozen=True)
class LoadResult:
    table: str
    rows_loaded: int


class LoadError(Exception):
    """Raised when CSV loading or database operations fail."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv(path: Path, label: str, required_columns: list[str]) -> pd.DataFrame:
    """Load and validate a CSV file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise LoadError(
            f"Missing {label}: {resolved}\n"
            "Run: python scripts/batch_detection.py && python scripts/compute_metrics.py --master"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise LoadError(f"{label} is empty: {resolved}") from exc
    except OSError as exc:
        raise LoadError(f"Cannot read {label}: {exc}") from exc

    if df.empty:
        raise LoadError(f"{label} contains no rows: {resolved}")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise LoadError(f"{label} missing columns: {sorted(missing)}")

    logger.info("Loaded %d row(s) from %s", len(df), resolved.name)
    return df[required_columns].copy()


def create_tables() -> None:
    """Create all ORM tables if they do not exist."""
    logger.info("Ensuring database tables exist.")
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as exc:
        raise LoadError(
            f"Database connection error: {exc}\n"
            "Verify DATABASE_URL and ensure PostgreSQL is running "
            "(docker compose -f docker/docker-compose.yml up -d)."
        ) from exc


def clear_table(session: Session, model) -> None:
    """Delete all existing rows from a table."""
    session.execute(delete(model))


def bulk_load(session: Session, model, df: pd.DataFrame, chunk_size: int = 2000) -> int:
    """Bulk insert dataframe rows using SQLAlchemy bulk mappings."""
    records = df.to_dict(orient="records")
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        session.bulk_insert_mappings(model, chunk)
    return len(records)


def load_detections(session: Session, df: pd.DataFrame) -> LoadResult:
    clear_table(session, TrafficDetection)
    count = bulk_load(session, TrafficDetection, df)
    return LoadResult(table=TrafficDetection.__tablename__, rows_loaded=count)


def load_congestion(session: Session, df: pd.DataFrame) -> LoadResult:
    clear_table(session, CongestionMetric)
    count = bulk_load(session, CongestionMetric, df)
    return LoadResult(table=CongestionMetric.__tablename__, rows_loaded=count)


def load_summary(session: Session, df: pd.DataFrame) -> LoadResult:
    clear_table(session, ObjectSummary)
    count = bulk_load(session, ObjectSummary, df)
    return LoadResult(table=ObjectSummary.__tablename__, rows_loaded=count)


def run_load_pipeline(
    detections_path: Path = DEFAULT_DETECTIONS_CSV,
    congestion_path: Path = DEFAULT_CONGESTION_CSV,
    summary_path: Path = DEFAULT_SUMMARY_CSV,
) -> list[LoadResult]:
    """Load all master CSV files into PostgreSQL within a single transaction."""
    detections_df = read_csv(detections_path, "traffic detections", DETECTION_COLUMNS)
    congestion_df = read_csv(congestion_path, "congestion metrics", CONGESTION_COLUMNS)
    summary_df = read_csv(summary_path, "object summary", SUMMARY_COLUMNS)

    create_tables()

    session = SessionLocal()
    results: list[LoadResult] = []

    try:
        logger.info("Connected to database: %s", _safe_db_url())
        results.append(load_detections(session, detections_df))
        results.append(load_congestion(session, congestion_df))
        results.append(load_summary(session, summary_df))
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        raise LoadError(f"Database error during load: {exc}") from exc
    finally:
        session.close()

    return results


def _safe_db_url() -> str:
    """Return database URL with password masked for logging."""
    url = get_database_url()
    if "@" in url and "://" in url:
        prefix, rest = url.split("://", 1)
        if "@" in rest:
            creds, host = rest.rsplit("@", 1)
            if ":" in creds:
                user = creds.split(":", 1)[0]
                return f"{prefix}://{user}:****@{host}"
    return url


def print_load_summary(results: list[LoadResult]) -> None:
    print("\n=== PostgreSQL Load Summary ===")
    for result in results:
        print(f"  {result.table:22s}: {result.rows_loaded:,} rows")
    print("================================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load master CSV analytics into PostgreSQL.")
    parser.add_argument(
        "--detections",
        type=Path,
        default=DEFAULT_DETECTIONS_CSV,
        help="Path to traffic_detections_master.csv",
    )
    parser.add_argument(
        "--congestion",
        type=Path,
        default=DEFAULT_CONGESTION_CSV,
        help="Path to congestion_metrics_master.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Path to object_summary_master.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        results = run_load_pipeline(
            detections_path=args.detections,
            congestion_path=args.congestion,
            summary_path=args.summary,
        )
    except LoadError as exc:
        logger.error("Load failed: %s", exc)
        return 1
    except SQLAlchemyError as exc:
        logger.error(
            "Database connection error: %s. Check DATABASE_URL and PostgreSQL.",
            exc,
        )
        return 1

    print_load_summary(results)
    logger.info("PostgreSQL load completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
