"""Near-miss detection from YOLO traffic detection records."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "traffic_detections_master.csv"
DEFAULT_EVENTS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "near_miss_events.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "near_miss_summary.csv"

VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle"})
RISK_OBJECT_CLASSES: frozenset[str] = frozenset({"person", "bicycle"})

DEFAULT_VEHICLE_CONFIDENCE_MIN = 0.35
DEFAULT_RISK_CONFIDENCE_MIN = 0.35
HIGH_RISK_DISTANCE_PX = 40.0
MEDIUM_RISK_DISTANCE_PX = 80.0

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
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
    }
)

EVENT_COLUMNS: list[str] = [
    "camera_id",
    "video_id",
    "timestamp",
    "frame_id",
    "risk_object_type",
    "vehicle_type",
    "risk_confidence",
    "vehicle_confidence",
    "risk_center_x",
    "risk_center_y",
    "vehicle_center_x",
    "vehicle_center_y",
    "distance_pixels",
    "risk_level",
]

SUMMARY_COLUMNS: list[str] = ["camera_id", "risk_level", "near_miss_count"]


class NearMissError(Exception):
    """Raised when near-miss detection cannot proceed."""


@dataclass(frozen=True)
class NearMissResult:
    """Outputs produced by the near-miss pipeline."""

    events: pd.DataFrame
    summary: pd.DataFrame
    events_path: Path
    summary_path: Path


def load_detections(path: Path) -> pd.DataFrame:
    """Load and validate the master detections CSV."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise NearMissError(
            f"Input file not found: {resolved}\n"
            "Run: python scripts/batch_detection.py"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise NearMissError(f"Input file is empty: {resolved}") from exc
    except OSError as exc:
        raise NearMissError(f"Failed to read detections CSV: {exc}") from exc

    if df.empty:
        raise NearMissError(f"Input file contains no rows: {resolved}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise NearMissError(f"Detections CSV missing columns: {sorted(missing)}")

    logger.info("Loaded %d detection row(s) from %s", len(df), resolved.name)
    return df


def add_bounding_box_centers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute bounding-box center coordinates for each detection."""
    enriched = df.copy()
    enriched["center_x"] = (enriched["x1"] + enriched["x2"]) / 2.0
    enriched["center_y"] = (enriched["y1"] + enriched["y2"]) / 2.0
    return enriched


def classify_risk_level(distance_pixels: float) -> str | None:
    """Map pixel distance to a near-miss risk tier."""
    if distance_pixels < HIGH_RISK_DISTANCE_PX:
        return "High"
    if distance_pixels < MEDIUM_RISK_DISTANCE_PX:
        return "Medium"
    return None


def compute_euclidean_distance(
    risk_center_x: float,
    risk_center_y: float,
    vehicle_center_x: float,
    vehicle_center_y: float,
) -> float:
    """Calculate Euclidean distance between two bounding-box centers."""
    return float(
        np.hypot(risk_center_x - vehicle_center_x, risk_center_y - vehicle_center_y)
    )


def _empty_events_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def detect_near_misses_in_frame(
    frame_df: pd.DataFrame,
    vehicle_confidence_min: float = DEFAULT_VEHICLE_CONFIDENCE_MIN,
    risk_confidence_min: float = DEFAULT_RISK_CONFIDENCE_MIN,
) -> pd.DataFrame:
    """Identify near-miss events within a single frame."""
    vehicles = frame_df[
        frame_df["object_type"].isin(VEHICLE_CLASSES)
        & (frame_df["confidence"] >= vehicle_confidence_min)
    ]
    risk_objects = frame_df[
        frame_df["object_type"].isin(RISK_OBJECT_CLASSES)
        & (frame_df["confidence"] >= risk_confidence_min)
    ]

    if vehicles.empty or risk_objects.empty:
        return _empty_events_frame()

    risks = risk_objects.reset_index(drop=True)
    veh = vehicles.reset_index(drop=True)
    risks["_pair_key"] = 1
    veh["_pair_key"] = 1
    pairs = risks.merge(veh, on="_pair_key", suffixes=("_risk", "_vehicle")).drop(
        columns="_pair_key"
    )

    pairs["distance_pixels"] = np.hypot(
        pairs["center_x_risk"] - pairs["center_x_vehicle"],
        pairs["center_y_risk"] - pairs["center_y_vehicle"],
    )

    near_misses = pairs[pairs["distance_pixels"] < MEDIUM_RISK_DISTANCE_PX].copy()
    if near_misses.empty:
        return _empty_events_frame()

    near_misses["risk_level"] = np.where(
        near_misses["distance_pixels"] < HIGH_RISK_DISTANCE_PX,
        "High",
        "Medium",
    )

    timestamp = float(frame_df["timestamp"].iloc[0])
    camera_id = str(frame_df["camera_id"].iloc[0])
    video_id = str(frame_df["video_id"].iloc[0])
    frame_id = int(frame_df["frame_id"].iloc[0])

    events = pd.DataFrame(
        {
            "camera_id": camera_id,
            "video_id": video_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "risk_object_type": near_misses["object_type_risk"].astype(str),
            "vehicle_type": near_misses["object_type_vehicle"].astype(str),
            "risk_confidence": near_misses["confidence_risk"].astype(float).round(4),
            "vehicle_confidence": near_misses["confidence_vehicle"].astype(float).round(4),
            "risk_center_x": near_misses["center_x_risk"].astype(float).round(2),
            "risk_center_y": near_misses["center_y_risk"].astype(float).round(2),
            "vehicle_center_x": near_misses["center_x_vehicle"].astype(float).round(2),
            "vehicle_center_y": near_misses["center_y_vehicle"].astype(float).round(2),
            "distance_pixels": near_misses["distance_pixels"].astype(float).round(2),
            "risk_level": near_misses["risk_level"].astype(str),
        }
    )
    return events[EVENT_COLUMNS]


def detect_near_miss_events(
    detections_df: pd.DataFrame,
    vehicle_confidence_min: float = DEFAULT_VEHICLE_CONFIDENCE_MIN,
    risk_confidence_min: float = DEFAULT_RISK_CONFIDENCE_MIN,
) -> pd.DataFrame:
    """
    Scan all frames and flag pedestrian/bicycle vs vehicle proximity events.

    Compares risk objects (person, bicycle) against vehicles (car, truck, bus,
    motorcycle) within each camera/video/frame group.
    """
    centered = add_bounding_box_centers(detections_df)
    frame_groups = centered.groupby(["camera_id", "video_id", "frame_id"], sort=False)

    event_frames: list[pd.DataFrame] = []
    frames_scanned = 0

    for _, frame_df in frame_groups:
        frames_scanned += 1
        frame_events = detect_near_misses_in_frame(
            frame_df,
            vehicle_confidence_min=vehicle_confidence_min,
            risk_confidence_min=risk_confidence_min,
        )
        if not frame_events.empty:
            event_frames.append(frame_events)

    logger.info("Scanned %d unique frame(s) for near-miss events", frames_scanned)

    if not event_frames:
        logger.info("No near-miss events detected.")
        return _empty_events_frame()

    events = pd.concat(event_frames, ignore_index=True)
    events = events.sort_values(
        ["camera_id", "video_id", "frame_id", "distance_pixels"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    logger.info(
        "Detected %d near-miss event(s) — High: %d, Medium: %d",
        len(events),
        int((events["risk_level"] == "High").sum()),
        int((events["risk_level"] == "Medium").sum()),
    )
    return events


def build_near_miss_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate near-miss counts by camera and risk level."""
    if events_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        events_df.groupby(["camera_id", "risk_level"], as_index=False)
        .size()
        .rename(columns={"size": "near_miss_count"})
        .sort_values(["camera_id", "risk_level"])
        .reset_index(drop=True)
    )
    return summary[SUMMARY_COLUMNS]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=False)
    except OSError as exc:
        raise NearMissError(f"Failed to write CSV: {resolved}: {exc}") from exc
    logger.info("Saved %d row(s) to %s", len(df), resolved.name)


def run_near_miss_pipeline(
    input_path: Path = DEFAULT_INPUT,
    events_output_path: Path = DEFAULT_EVENTS_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    vehicle_confidence_min: float = DEFAULT_VEHICLE_CONFIDENCE_MIN,
    risk_confidence_min: float = DEFAULT_RISK_CONFIDENCE_MIN,
) -> NearMissResult:
    """End-to-end near-miss detection pipeline."""
    detections = load_detections(input_path)
    events = detect_near_miss_events(
        detections,
        vehicle_confidence_min=vehicle_confidence_min,
        risk_confidence_min=risk_confidence_min,
    )
    summary = build_near_miss_summary(events)

    save_dataframe(events, events_output_path)
    save_dataframe(summary, summary_output_path)

    return NearMissResult(
        events=events,
        summary=summary,
        events_path=events_output_path.resolve(),
        summary_path=summary_output_path.resolve(),
    )
