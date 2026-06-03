#!/usr/bin/env python3
"""
Generate synthetic external weather and event features for congestion analytics.

Joins batch congestion metrics with camera metadata and enriches each
camera/video/minute row with realistic simulated weather and nearby-event
features for downstream modeling.

Usage:
    python scripts/generate_external_features.py
    python scripts/generate_external_features.py --verbose

Prerequisites:
    python scripts/compute_metrics.py --master
    python scripts/generate_camera_metadata.py

Output:
    data/processed/congestion_features_master.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("generate_external_features")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONGESTION = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "camera_metadata.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "congestion_features_master.csv"

HIGH_EVENT_SCENES: frozenset[str] = frozenset({"city_street", "intersection"})
EVENT_TYPES: tuple[str, ...] = ("sports", "concert", "campus_event", "none")
WEATHER_CONDITIONS: tuple[str, ...] = ("clear", "cloudy", "rain", "fog", "snow")

CONGESTION_REQUIRED_COLUMNS = [
    "camera_id",
    "video_id",
    "minute",
    "vehicle_count",
    "pedestrian_count",
    "traffic_density",
    "congestion_level",
]

METADATA_REQUIRED_COLUMNS = ["camera_id", "scene_type", "road_type"]

OUTPUT_COLUMNS = [
    "camera_id",
    "video_id",
    "scene_type",
    "road_type",
    "minute",
    "vehicle_count",
    "pedestrian_count",
    "traffic_density",
    "congestion_level",
    "temperature_f",
    "precipitation_flag",
    "visibility_miles",
    "weather_condition",
    "event_nearby_flag",
    "event_type",
    "estimated_event_attendance",
    "distance_to_event_miles",
]


class ExternalFeaturesError(Exception):
    """Raised when external feature generation fails."""


@dataclass(frozen=True)
class WeatherFeatures:
    temperature_f: float
    precipitation_flag: int
    visibility_miles: float
    weather_condition: str


@dataclass(frozen=True)
class EventFeatures:
    event_nearby_flag: int
    event_type: str
    estimated_event_attendance: int
    distance_to_event_miles: float


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_csv(path: Path, label: str, required_columns: list[str]) -> pd.DataFrame:
    """Load and validate a CSV file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ExternalFeaturesError(f"Missing {label}: {resolved}")

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise ExternalFeaturesError(f"{label} is empty: {resolved}") from exc
    except OSError as exc:
        raise ExternalFeaturesError(f"Cannot read {label}: {exc}") from exc

    if df.empty:
        raise ExternalFeaturesError(f"{label} contains no rows: {resolved}")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ExternalFeaturesError(f"{label} missing columns: {sorted(missing)}")

    logger.info("Loaded %d row(s) from %s", len(df), resolved.name)
    return df


def row_seed(camera_id: str, video_id: str, minute: int | float) -> int:
    """Return a deterministic seed for a camera/video/minute row."""
    key = f"{camera_id}:{video_id}:{minute}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def generate_weather_features(seed: int) -> WeatherFeatures:
    """Generate realistic synthetic weather features."""
    rng = np.random.default_rng(seed)

    temperature_f = float(rng.normal(loc=58.0, scale=18.0))
    temperature_f = float(np.clip(temperature_f, 20.0, 95.0))

    precipitation_flag = int(rng.random() < 0.28)

    if precipitation_flag:
        visibility_miles = float(rng.uniform(0.8, 4.5))
        if temperature_f <= 34.0:
            weather_condition = "snow"
        else:
            weather_condition = "rain"
    else:
        visibility_miles = float(rng.uniform(5.0, 12.0))
        fog_roll = rng.random()
        if fog_roll < 0.08:
            weather_condition = "fog"
            visibility_miles = float(rng.uniform(0.5, 2.5))
        elif fog_roll < 0.45:
            weather_condition = "cloudy"
        else:
            weather_condition = "clear"

    return WeatherFeatures(
        temperature_f=round(temperature_f, 1),
        precipitation_flag=precipitation_flag,
        visibility_miles=round(visibility_miles, 2),
        weather_condition=weather_condition,
    )


def event_probability(scene_type: str) -> float:
    """Return the probability of a nearby event for a scene type."""
    normalized = str(scene_type).strip().lower()
    if normalized in HIGH_EVENT_SCENES:
        return 0.55
    if normalized in {"night_city", "composite"}:
        return 0.35
    return 0.12


def generate_event_features(seed: int, scene_type: str) -> EventFeatures:
    """Generate synthetic nearby-event features."""
    rng = np.random.default_rng(seed + 1)

    event_nearby_flag = int(rng.random() < event_probability(scene_type))
    if not event_nearby_flag:
        return EventFeatures(
            event_nearby_flag=0,
            event_type="none",
            estimated_event_attendance=0,
            distance_to_event_miles=round(float(rng.uniform(3.0, 12.0)), 2),
        )

    event_type = str(
        rng.choice(
            ["sports", "concert", "campus_event"],
            p=[0.35, 0.30, 0.35],
        )
    )

    attendance_ranges: dict[str, tuple[int, int]] = {
        "sports": (8_000, 60_000),
        "concert": (3_000, 25_000),
        "campus_event": (500, 15_000),
    }
    low, high = attendance_ranges[event_type]
    estimated_event_attendance = int(rng.integers(low, high + 1))
    distance_to_event_miles = round(float(rng.uniform(0.1, 2.0)), 2)

    return EventFeatures(
        event_nearby_flag=1,
        event_type=event_type,
        estimated_event_attendance=estimated_event_attendance,
        distance_to_event_miles=distance_to_event_miles,
    )


def join_congestion_metadata(
    congestion_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join congestion metrics with camera metadata."""
    metadata = metadata_df[["camera_id", "scene_type", "road_type"]].drop_duplicates()
    joined = congestion_df.merge(metadata, on="camera_id", how="left")

    missing_scene = joined["scene_type"].isna().sum()
    if missing_scene:
        logger.warning(
            "%d row(s) missing camera metadata — filling scene/road as unknown.",
            missing_scene,
        )
        joined["scene_type"] = joined["scene_type"].fillna("unknown")
        joined["road_type"] = joined["road_type"].fillna("unknown")

    return joined


def enrich_with_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append synthetic weather and event columns to each row."""
    weather_records: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        seed = row_seed(str(row.camera_id), str(row.video_id), int(row.minute))
        weather = generate_weather_features(seed)
        events = generate_event_features(seed, str(row.scene_type))

        weather_records.append(
            {
                "temperature_f": weather.temperature_f,
                "precipitation_flag": weather.precipitation_flag,
                "visibility_miles": weather.visibility_miles,
                "weather_condition": weather.weather_condition,
            }
        )
        event_records.append(
            {
                "event_nearby_flag": events.event_nearby_flag,
                "event_type": events.event_type,
                "estimated_event_attendance": events.estimated_event_attendance,
                "distance_to_event_miles": events.distance_to_event_miles,
            }
        )

    enriched = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(weather_records), pd.DataFrame(event_records)],
        axis=1,
    )

    enriched["precipitation_flag"] = enriched["precipitation_flag"].astype(int)
    enriched["event_nearby_flag"] = enriched["event_nearby_flag"].astype(int)
    enriched["estimated_event_attendance"] = enriched["estimated_event_attendance"].astype(int)
    enriched["vehicle_count"] = enriched["vehicle_count"].astype(int)
    enriched["pedestrian_count"] = enriched["pedestrian_count"].astype(int)
    enriched["minute"] = enriched["minute"].astype(int)

    logger.info(
        "Generated external features — precipitation: %d row(s), nearby events: %d row(s)",
        int(enriched["precipitation_flag"].sum()),
        int(enriched["event_nearby_flag"].sum()),
    )
    return enriched[OUTPUT_COLUMNS]


def save_features(df: pd.DataFrame, output_path: Path) -> Path:
    """Write enriched feature dataset to CSV."""
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=False)
    except OSError as exc:
        raise ExternalFeaturesError(f"Failed to write output CSV: {exc}") from exc
    logger.info("Saved %d row(s) to %s", len(df), resolved)
    return resolved


def run_external_features_pipeline(
    congestion_path: Path = DEFAULT_CONGESTION,
    metadata_path: Path = DEFAULT_METADATA,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """End-to-end external feature engineering pipeline."""
    congestion_df = load_csv(congestion_path, "congestion metrics", CONGESTION_REQUIRED_COLUMNS)
    metadata_df = load_csv(metadata_path, "camera metadata", METADATA_REQUIRED_COLUMNS)

    joined = join_congestion_metadata(congestion_df, metadata_df)
    features = enrich_with_external_features(joined)
    save_features(features, output_path)
    return features


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic weather and event features for congestion data."
    )
    parser.add_argument(
        "--congestion",
        type=Path,
        default=DEFAULT_CONGESTION,
        help="Path to congestion_metrics_master.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to camera_metadata.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for congestion_features_master.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        features = run_external_features_pipeline(
            congestion_path=args.congestion,
            metadata_path=args.metadata,
            output_path=args.output,
        )
    except ExternalFeaturesError as exc:
        logger.error("External feature generation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    print(f"\nGenerated {len(features)} feature row(s) -> {args.output.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
