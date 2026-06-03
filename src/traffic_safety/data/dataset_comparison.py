"""Build cross-dataset comparison metrics for portfolio benchmarking."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from traffic_safety.data.datasets import (
    COMPARISON_OUTPUT,
    DATASET_REGISTRY,
    DatasetConfig,
    DatasetError,
    load_manifest,
)

logger = logging.getLogger(__name__)

VEHICLE_TYPES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
PEDESTRIAN_TYPE = "person"

COMPARISON_COLUMNS: list[str] = [
    "dataset",
    "display_name",
    "video_count",
    "camera_count",
    "total_detections",
    "vehicle_count",
    "pedestrian_count",
    "near_miss_count",
    "avg_density",
    "max_density",
    "status",
]


def _safe_load_yaml_videos(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError:
        return []
    if not isinstance(raw, dict):
        return []
    videos = raw.get("videos", [])
    return videos if isinstance(videos, list) else []


def _manifest_video_count(config: DatasetConfig) -> int:
    videos = _safe_load_yaml_videos(config.manifest_path)
    return len(videos)


def _manifest_camera_count(config: DatasetConfig) -> int:
    videos = _safe_load_yaml_videos(config.manifest_path)
    cameras = {
        str(item.get("camera_id"))
        for item in videos
        if isinstance(item, dict) and item.get("camera_id")
    }
    return len(cameras)


def _read_detections(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Cannot read detections %s: %s", path.name, exc)
        return None
    return df if not df.empty else None


def _read_congestion(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Cannot read congestion %s: %s", path.name, exc)
        return None
    return df if not df.empty else None


def _read_near_miss(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Cannot read near-miss %s: %s", path.name, exc)
        return None
    return df if not df.empty else None


def compute_dataset_row(config: DatasetConfig) -> dict[str, float | int | str]:
    """Compute comparison metrics for a single dataset."""
    detections = _read_detections(config.detections_csv())
    congestion = _read_congestion(config.congestion_csv())
    near_miss = _read_near_miss(config.near_miss_events_csv())

    has_manifest = config.manifest_path.exists()
    has_detections = detections is not None

    if not has_manifest and not has_detections:
        return {
            "dataset": config.name,
            "display_name": config.display_name,
            "video_count": 0,
            "camera_count": 0,
            "total_detections": 0,
            "vehicle_count": 0,
            "pedestrian_count": 0,
            "near_miss_count": 0,
            "avg_density": 0.0,
            "max_density": 0.0,
            "status": "not_available",
        }

    video_count = _manifest_video_count(config)
    camera_count = _manifest_camera_count(config)

    if detections is not None:
        if "video_id" in detections.columns:
            video_count = max(video_count, int(detections["video_id"].nunique()))
        if "camera_id" in detections.columns:
            camera_count = max(camera_count, int(detections["camera_id"].nunique()))

        total_detections = len(detections)
        vehicle_count = int(detections["object_type"].isin(VEHICLE_TYPES).sum())
        pedestrian_count = int((detections["object_type"] == PEDESTRIAN_TYPE).sum())
    else:
        total_detections = vehicle_count = pedestrian_count = 0

    near_miss_count = len(near_miss) if near_miss is not None else 0

    avg_density = 0.0
    max_density = 0.0
    if congestion is not None and "traffic_density" in congestion.columns:
        density = pd.to_numeric(congestion["traffic_density"], errors="coerce").dropna()
        if not density.empty:
            avg_density = round(float(density.mean()), 2)
            max_density = round(float(density.max()), 2)

    status = "ready" if has_detections else "manifest_only"

    return {
        "dataset": config.name,
        "display_name": config.display_name,
        "video_count": video_count,
        "camera_count": camera_count,
        "total_detections": total_detections,
        "vehicle_count": vehicle_count,
        "pedestrian_count": pedestrian_count,
        "near_miss_count": near_miss_count,
        "avg_density": avg_density,
        "max_density": max_density,
        "status": status,
    }


def build_comparison_dataframe(
    dataset_names: list[str] | None = None,
) -> pd.DataFrame:
    """Build comparison table for all or selected datasets."""
    names = dataset_names or list(DATASET_REGISTRY.keys())
    rows: list[dict[str, float | int | str]] = []

    for name in names:
        try:
            config = DATASET_REGISTRY[name.strip().lower()]
        except KeyError:
            logger.warning("Skipping unknown dataset: %s", name)
            continue
        rows.append(compute_dataset_row(config))

    if not rows:
        raise DatasetError("No datasets available for comparison.")

    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def save_comparison_csv(
    output_path: Path | None = None,
    dataset_names: list[str] | None = None,
) -> Path:
    """Write dataset comparison CSV and return output path."""
    df = build_comparison_dataframe(dataset_names)
    resolved = (output_path or COMPARISON_OUTPUT).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(resolved, index=False)
    logger.info("Saved dataset comparison (%d rows) -> %s", len(df), resolved)
    return resolved


def count_ready_datasets(df: pd.DataFrame) -> int:
    """Number of datasets with processed detections."""
    if df.empty:
        return 0
    return int((df["status"] == "ready").sum())
