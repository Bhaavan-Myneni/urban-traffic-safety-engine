"""Load processed analytics CSV files for the dashboard."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# MVP paths
DETECTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_detections.csv"
CONGESTION_PATH = PROJECT_ROOT / "data" / "processed" / "congestion_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "object_summary.csv"

# Batch master paths
DETECTIONS_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "traffic_detections_master.csv"
CONGESTION_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
SUMMARY_MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "object_summary_master.csv"

# Extended analytics paths
CAMERA_METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "camera_metadata.csv"
NEAR_MISS_EVENTS_PATH = PROJECT_ROOT / "data" / "processed" / "near_miss_events.csv"
NEAR_MISS_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "near_miss_summary.csv"
DELAY_COST_ANALYSIS_PATH = PROJECT_ROOT / "data" / "processed" / "delay_cost_analysis.csv"
DELAY_COST_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "delay_cost_summary.csv"
AB_SIMULATION_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "ab_simulation_results.csv"
AB_SIMULATION_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "ab_simulation_summary.csv"
CONGESTION_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "congestion_features_master.csv"
FORECAST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_predictions.csv"
DATASET_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "dataset_comparison.csv"
HOTSPOT_MAP_PATH = PROJECT_ROOT / "outputs" / "traffic_hotspot_map.html"

VEHICLE_TYPES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
PEDESTRIAN_TYPE = "person"


@dataclass(frozen=True)
class DataPaths:
    detections: Path
    congestion: Path
    summary: Path
    mode: str  # "master" or "mvp"


@dataclass(frozen=True)
class DashboardData:
    """Container for all dashboard datasets."""

    detections: pd.DataFrame
    congestion: pd.DataFrame
    summary: pd.DataFrame
    mode: str
    paths: DataPaths
    camera_metadata: pd.DataFrame | None = None
    near_miss_events: pd.DataFrame | None = None
    near_miss_summary: pd.DataFrame | None = None
    delay_cost_analysis: pd.DataFrame | None = None
    delay_cost_summary: pd.DataFrame | None = None
    ab_simulation_results: pd.DataFrame | None = None
    ab_simulation_summary: pd.DataFrame | None = None
    congestion_features: pd.DataFrame | None = None
    forecast_predictions: pd.DataFrame | None = None
    dataset_comparison: pd.DataFrame | None = None
    hotspot_map_path: Path | None = None
    optional_available: dict[str, bool] = field(default_factory=dict)


class DataLoadError(Exception):
    """Raised when required dashboard data files are missing or invalid."""


def is_csv_only_mode() -> bool:
    """True when the dashboard must not use PostgreSQL or rebuild datasets."""
    flag = os.environ.get("DASHBOARD_CSV_ONLY", "").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return True
    return bool(os.environ.get("RENDER"))


def resolve_data_paths() -> DataPaths:
    """Prefer batch master files; fall back to MVP files when master is unavailable."""
    master_ready = all(
        p.exists() for p in (DETECTIONS_MASTER_PATH, CONGESTION_MASTER_PATH, SUMMARY_MASTER_PATH)
    )
    if master_ready:
        logger.info("Using batch master dataset.")
        return DataPaths(
            detections=DETECTIONS_MASTER_PATH,
            congestion=CONGESTION_MASTER_PATH,
            summary=SUMMARY_MASTER_PATH,
            mode="master",
        )

    mvp_ready = all(p.exists() for p in (DETECTIONS_PATH, CONGESTION_PATH, SUMMARY_PATH))
    if mvp_ready:
        logger.info("Master files not found — falling back to MVP dataset.")
        return DataPaths(
            detections=DETECTIONS_PATH,
            congestion=CONGESTION_PATH,
            summary=SUMMARY_PATH,
            mode="mvp",
        )

    raise DataLoadError(
        "No dashboard dataset found in data/processed/.\n\n"
        "Commit demo CSVs (traffic_detections_master.csv, congestion_metrics_master.csv, "
        "object_summary_master.csv) or run the local pipeline:\n"
        "  python scripts/batch_detection.py\n"
        "  python scripts/compute_metrics.py --master"
    )


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"{label} is empty: {path}") from exc
    except OSError as exc:
        raise DataLoadError(f"Cannot read {label}: {exc}") from exc

    if df.empty:
        raise DataLoadError(f"{label} contains no rows: {path}")
    return df


def _read_csv_optional(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.info("Optional dataset not found: %s", path.name)
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError) as exc:
        logger.warning("Could not load optional %s: %s", label, exc)
        return None
    if df.empty:
        logger.warning("Optional %s is empty: %s", label, path.name)
        return None
    logger.info("Loaded optional %s (%d rows)", label, len(df))
    return df


def _normalize_detections(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["object_type"] = out["object_type"].astype(str).str.strip().str.lower()
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    if "camera_id" in out.columns:
        out["camera_id"] = out["camera_id"].astype(str).str.strip()
    if "video_id" in out.columns:
        out["video_id"] = out["video_id"].astype(str).str.strip()
    if out["confidence"].isna().any():
        raise DataLoadError("Detections file contains invalid confidence values.")
    return out


def load_dashboard_data(paths: DataPaths | None = None) -> DashboardData:
    """Load core and optional dashboard datasets."""
    resolved = paths or resolve_data_paths()

    detections = _normalize_detections(_read_csv(resolved.detections, "traffic detections"))
    congestion = _read_csv(resolved.congestion, "congestion metrics")
    summary = _read_csv(resolved.summary, "object summary")

    optional_files = {
        "camera_metadata": CAMERA_METADATA_PATH,
        "near_miss_events": NEAR_MISS_EVENTS_PATH,
        "near_miss_summary": NEAR_MISS_SUMMARY_PATH,
        "delay_cost_analysis": DELAY_COST_ANALYSIS_PATH,
        "delay_cost_summary": DELAY_COST_SUMMARY_PATH,
        "ab_simulation_results": AB_SIMULATION_RESULTS_PATH,
        "ab_simulation_summary": AB_SIMULATION_SUMMARY_PATH,
        "congestion_features": CONGESTION_FEATURES_PATH,
        "forecast_predictions": FORECAST_PREDICTIONS_PATH,
        "dataset_comparison": DATASET_COMPARISON_PATH,
    }

    optional_data: dict[str, pd.DataFrame | None] = {
        key: _read_csv_optional(path, key.replace("_", " "))
        for key, path in optional_files.items()
    }

    hotspot_map_path = HOTSPOT_MAP_PATH if HOTSPOT_MAP_PATH.exists() else None
    if hotspot_map_path:
        logger.info("Found hotspot map: %s", hotspot_map_path.name)

    dataset_comparison = optional_data["dataset_comparison"]
    if dataset_comparison is None:
        dataset_comparison = load_or_build_dataset_comparison()

    optional_available = {
        key: value is not None for key, value in optional_data.items()
    }
    optional_available["dataset_comparison"] = dataset_comparison is not None
    optional_available["hotspot_map"] = hotspot_map_path is not None

    return DashboardData(
        detections=detections,
        congestion=congestion,
        summary=summary,
        mode=resolved.mode,
        paths=resolved,
        camera_metadata=optional_data["camera_metadata"],
        near_miss_events=optional_data["near_miss_events"],
        near_miss_summary=optional_data["near_miss_summary"],
        delay_cost_analysis=optional_data["delay_cost_analysis"],
        delay_cost_summary=optional_data["delay_cost_summary"],
        ab_simulation_results=optional_data["ab_simulation_results"],
        ab_simulation_summary=optional_data["ab_simulation_summary"],
        congestion_features=optional_data["congestion_features"],
        forecast_predictions=optional_data["forecast_predictions"],
        dataset_comparison=dataset_comparison,
        hotspot_map_path=hotspot_map_path,
        optional_available=optional_available,
    )


def attach_scene_type(
    df: pd.DataFrame,
    metadata: pd.DataFrame | None,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join scene_type onto a dataframe when missing."""
    if df.empty or "camera_id" not in df.columns:
        return df
    if "scene_type" in df.columns:
        return df

    lookup = None
    if metadata is not None and "scene_type" in metadata.columns:
        lookup = metadata[["camera_id", "scene_type"]].drop_duplicates()
    elif features is not None and "scene_type" in features.columns:
        lookup = features[["camera_id", "scene_type"]].drop_duplicates()

    if lookup is None:
        return df

    return df.merge(lookup, on="camera_id", how="left")


def filter_by_cameras(df: pd.DataFrame, camera_ids: list[str] | None) -> pd.DataFrame:
    if not camera_ids or "camera_id" not in df.columns:
        return df
    return df[df["camera_id"].isin(camera_ids)]


def filter_by_scene_types(df: pd.DataFrame, scene_types: list[str] | None) -> pd.DataFrame:
    if not scene_types or "scene_type" not in df.columns:
        return df
    return df[df["scene_type"].isin(scene_types)]


def filter_detections(
    detections: pd.DataFrame,
    object_types: list[str] | None = None,
    camera_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Apply sidebar filters to detections."""
    filtered = detections[detections["confidence"] >= min_confidence]
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    if object_types:
        filtered = filtered[filtered["object_type"].isin(object_types)]
    if video_ids and "video_id" in filtered.columns:
        filtered = filtered[filtered["video_id"].isin(video_ids)]
    return filtered


def filter_congestion(
    congestion: pd.DataFrame,
    camera_ids: list[str] | None = None,
    video_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
    congestion_levels: list[str] | None = None,
) -> pd.DataFrame:
    """Apply sidebar filters to congestion metrics."""
    filtered = congestion.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    if video_ids and "video_id" in filtered.columns:
        filtered = filtered[filtered["video_id"].isin(video_ids)]
    if congestion_levels:
        filtered = filtered[filtered["congestion_level"].isin(congestion_levels)]
    return filtered


def filter_summary(
    summary: pd.DataFrame,
    object_types: list[str] | None = None,
    camera_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
) -> pd.DataFrame:
    """Filter object summary by type, camera, and scene."""
    filtered = summary.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    if object_types:
        filtered = filtered[filtered["object_type"].isin(object_types)]
    return filtered.reset_index(drop=True)


def filter_near_miss_events(
    events: pd.DataFrame,
    camera_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
    risk_levels: list[str] | None = None,
) -> pd.DataFrame:
    """Filter near-miss event records."""
    filtered = events.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    if risk_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    return filtered


def filter_near_miss_summary(
    summary: pd.DataFrame,
    camera_ids: list[str] | None = None,
    risk_levels: list[str] | None = None,
) -> pd.DataFrame:
    """Filter near-miss summary aggregates."""
    filtered = summary.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    if risk_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    return filtered


def filter_delay_cost(
    delay_cost: pd.DataFrame,
    camera_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
    congestion_levels: list[str] | None = None,
) -> pd.DataFrame:
    """Filter delay cost analysis rows."""
    filtered = delay_cost.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    if congestion_levels and "congestion_level" in filtered.columns:
        filtered = filtered[filtered["congestion_level"].isin(congestion_levels)]
    return filtered


def filter_ab_simulation(
    ab_results: pd.DataFrame,
    camera_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
) -> pd.DataFrame:
    """Filter A/B simulation results."""
    filtered = ab_results.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    return filtered


def filter_forecast_predictions(
    predictions: pd.DataFrame,
    camera_ids: list[str] | None = None,
    scene_types: list[str] | None = None,
) -> pd.DataFrame:
    """Filter forecast prediction rows."""
    filtered = predictions.copy()
    filtered = filter_by_cameras(filtered, camera_ids)
    filtered = filter_by_scene_types(filtered, scene_types)
    return filtered


def build_filtered_summary_from_detections(
    detections: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """Build object summary from filtered detections for charts."""
    if mode == "master" and "camera_id" in detections.columns:
        grouped = (
            detections.groupby(["camera_id", "object_type"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        grouped["percentage"] = grouped.groupby("camera_id")["count"].transform(
            lambda s: (s / s.sum() * 100).round(2)
        )
        return grouped

    counts = (
        detections["object_type"]
        .value_counts()
        .rename_axis("object_type")
        .reset_index(name="count")
    )
    total = counts["count"].sum()
    counts["percentage"] = (counts["count"] / total * 100).round(2)
    return counts


def compute_kpis(
    detections: pd.DataFrame,
    congestion: pd.DataFrame,
    mode: str,
    near_miss_events: pd.DataFrame | None = None,
    delay_cost_analysis: pd.DataFrame | None = None,
    ab_simulation_results: pd.DataFrame | None = None,
    forecast_predictions: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    """Compute top-level KPI values for metric cards."""
    vehicle_detections = int(detections["object_type"].isin(VEHICLE_TYPES).sum())
    pedestrian_detections = int((detections["object_type"] == PEDESTRIAN_TYPE).sum())

    total_vehicles = (
        int(congestion["vehicle_count"].sum()) if "vehicle_count" in congestion.columns else vehicle_detections
    )
    total_pedestrians = (
        int(congestion["pedestrian_count"].sum())
        if "pedestrian_count" in congestion.columns
        else pedestrian_detections
    )

    avg_density = float(congestion["traffic_density"].mean()) if not congestion.empty else 0.0

    peak_label: str | int = "—"
    if not congestion.empty:
        peak_idx = congestion["traffic_density"].idxmax()
        peak_row = congestion.loc[peak_idx]
        if mode == "master" and "camera_id" in peak_row:
            peak_label = f"{peak_row['camera_id']} (min {int(peak_row['minute'])})"
        else:
            peak_label = int(peak_row["minute"])

    kpis: dict[str, float | int | str] = {
        "total_detections": len(detections),
        "total_vehicles": total_vehicles,
        "total_pedestrians": total_pedestrians,
        "vehicle_detections": vehicle_detections,
        "pedestrian_detections": pedestrian_detections,
        "average_density": round(avg_density, 2),
        "peak_congestion": peak_label,
        "total_near_miss_events": 0,
        "high_risk_near_misses": 0,
        "total_delay_cost_usd": 0.0,
        "ab_cost_savings_usd": 0.0,
        "forecast_rmse": "—",
    }

    if mode == "master" and "camera_id" in detections.columns:
        kpis["camera_count"] = int(detections["camera_id"].nunique())
        kpis["video_count"] = int(detections["video_id"].nunique())

    if near_miss_events is not None and not near_miss_events.empty:
        kpis["total_near_miss_events"] = len(near_miss_events)
        if "risk_level" in near_miss_events.columns:
            kpis["high_risk_near_misses"] = int((near_miss_events["risk_level"] == "High").sum())

    if delay_cost_analysis is not None and not delay_cost_analysis.empty:
        kpis["total_delay_cost_usd"] = round(
            float(delay_cost_analysis["estimated_delay_cost_usd"].sum()), 2
        )

    if ab_simulation_results is not None and not ab_simulation_results.empty:
        savings = float(
            ab_simulation_results["baseline_delay_cost_usd"].sum()
            - ab_simulation_results["simulated_delay_cost_usd"].sum()
        )
        kpis["ab_cost_savings_usd"] = round(savings, 2)

    if forecast_predictions is not None and not forecast_predictions.empty:
        if "prediction_error" in forecast_predictions.columns:
            rmse = math.sqrt(float((forecast_predictions["prediction_error"] ** 2).mean()))
            kpis["forecast_rmse"] = round(rmse, 4)

    return kpis


def get_filter_options(data: DashboardData) -> dict[str, list[str]]:
    """Collect sidebar filter options from loaded datasets."""
    scene_lookup = attach_scene_type(
        data.congestion.copy(),
        data.camera_metadata,
        data.congestion_features,
    )

    if "camera_id" in data.congestion.columns:
        cameras = sorted(data.congestion["camera_id"].unique().tolist())
    elif "camera_id" in data.detections.columns:
        cameras = sorted(data.detections["camera_id"].unique().tolist())
    else:
        cameras = []

    scene_types = (
        sorted(scene_lookup["scene_type"].dropna().unique().tolist())
        if "scene_type" in scene_lookup.columns
        else []
    )
    object_types = sorted(data.detections["object_type"].unique().tolist())
    congestion_levels = sorted(data.congestion["congestion_level"].unique().tolist())
    risk_levels = ["High", "Medium"]

    if data.near_miss_events is not None and "risk_level" in data.near_miss_events.columns:
        risk_levels = sorted(data.near_miss_events["risk_level"].unique().tolist())

    return {
        "cameras": cameras,
        "scene_types": scene_types,
        "object_types": object_types,
        "congestion_levels": congestion_levels,
        "risk_levels": risk_levels,
    }


def load_or_build_dataset_comparison() -> pd.DataFrame | None:
    """Load comparison CSV or build it on the fly when missing (local dev only)."""
    loaded = _read_csv_optional(DATASET_COMPARISON_PATH, "dataset comparison")
    if loaded is not None:
        return loaded

    if is_csv_only_mode():
        logger.info("Skipping dataset comparison build (CSV-only / deploy mode).")
        return None

    try:
        from traffic_safety.data.dataset_comparison import build_comparison_dataframe

        df = build_comparison_dataframe()
        if df.empty:
            return None
        return df
    except Exception as exc:
        logger.warning("Could not build dataset comparison: %s", exc)
        return None


def count_comparison_ready_datasets(comparison: pd.DataFrame | None) -> int:
    """Count datasets with processed detections in the comparison table."""
    if comparison is None or comparison.empty:
        return 0
    if "status" in comparison.columns:
        return int((comparison["status"] == "ready").sum())
    return int((comparison["total_detections"] > 0).sum())
