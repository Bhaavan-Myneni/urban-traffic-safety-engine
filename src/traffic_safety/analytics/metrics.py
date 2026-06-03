"""
Transform raw YOLO detections into traffic analytics KPIs.

Supports MVP (single-video) and batch (multi-camera) detection CSVs.

Usage:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py \\
        --input data/processed/traffic_detections_master.csv \\
        --output data/processed/congestion_metrics_master.csv \\
        --summary-output data/processed/object_summary_master.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# MVP defaults
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "traffic_detections.csv"
DEFAULT_CONGESTION_OUTPUT = PROJECT_ROOT / "data" / "processed" / "congestion_metrics.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "object_summary.csv"

# Batch master defaults
DEFAULT_MASTER_INPUT = PROJECT_ROOT / "data" / "processed" / "traffic_detections_master.csv"
DEFAULT_MASTER_CONGESTION_OUTPUT = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_MASTER_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "object_summary_master.csv"

VEHICLE_TYPES: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
PEDESTRIAN_TYPE = "person"

CONGESTION_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("Low", 5.0),
    ("Medium", 10.0),
    ("High", 20.0),
)

REQUIRED_DETECTION_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "frame_id", "object_type", "confidence", "x1", "y1", "x2", "y2"}
)
BATCH_DETECTION_COLUMNS: frozenset[str] = REQUIRED_DETECTION_COLUMNS | {"camera_id", "video_id"}

CONGESTION_COLUMNS_MVP: list[str] = [
    "minute",
    "sampled_frame_count",
    "vehicle_count",
    "pedestrian_count",
    "avg_vehicles_per_frame",
    "avg_pedestrians_per_frame",
    "traffic_density",
    "congestion_level",
]

CONGESTION_COLUMNS_BATCH: list[str] = [
    "camera_id",
    "video_id",
    *CONGESTION_COLUMNS_MVP,
]

SUMMARY_COLUMNS: list[str] = ["object_type", "count", "percentage"]
SUMMARY_COLUMNS_BATCH: list[str] = ["camera_id", "object_type", "count", "percentage"]


@dataclass(frozen=True)
class SummaryStatistics:
    """High-level statistics computed across the full detection dataset."""

    peak_density: float
    average_density: float
    peak_minute: int
    total_vehicle_detections: int
    total_pedestrian_detections: int
    peak_camera: str | None = None
    camera_count: int | None = None
    video_count: int | None = None

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "peak_density": round(self.peak_density, 2),
            "average_density": round(self.average_density, 2),
            "peak_minute": self.peak_minute,
            "total_vehicle_detections": self.total_vehicle_detections,
            "total_pedestrian_detections": self.total_pedestrian_detections,
            "peak_camera": self.peak_camera,
            "camera_count": self.camera_count,
            "video_count": self.video_count,
        }


class MetricsError(Exception):
    """Raised when metrics computation cannot proceed."""


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def is_batch_dataset(df: pd.DataFrame) -> bool:
    """Return True when detections include multi-camera batch columns."""
    return BATCH_DETECTION_COLUMNS.issubset(set(df.columns))


def load_detections(input_path: Path) -> pd.DataFrame:
    """Load and validate a YOLO detections CSV (MVP or batch)."""
    resolved = input_path.expanduser().resolve()

    if not resolved.exists():
        raise MetricsError(
            f"Detections file not found: {resolved}\n"
            "Run: python scripts/run_detection.py  OR  python scripts/batch_detection.py"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise MetricsError(f"Detections file is empty: {resolved}") from exc
    except OSError as exc:
        raise MetricsError(f"Cannot read detections file: {exc}") from exc

    if df.empty:
        raise MetricsError(f"Detections file contains no rows: {resolved}")

    required = BATCH_DETECTION_COLUMNS if is_batch_dataset(df) else REQUIRED_DETECTION_COLUMNS
    missing = required - set(df.columns)
    if missing:
        raise MetricsError(f"Detections file missing required columns: {sorted(missing)}")

    df = df.copy()
    df["object_type"] = df["object_type"].astype(str).str.strip().str.lower()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")

    if df["timestamp"].isna().any():
        raise MetricsError("Detections file contains invalid timestamp values.")
    if df["frame_id"].isna().any():
        raise MetricsError("Detections file contains invalid frame_id values.")

    if is_batch_dataset(df):
        df["camera_id"] = df["camera_id"].astype(str).str.strip()
        df["video_id"] = df["video_id"].astype(str).str.strip()

    mode = "batch" if is_batch_dataset(df) else "MVP"
    logger.info("Loaded %d %s detection(s) from %s", len(df), mode, resolved)
    return df


def classify_congestion(density: float) -> str:
    """Map normalized traffic density to Low / Medium / High / Severe."""
    for level, threshold in CONGESTION_THRESHOLDS:
        if density < threshold:
            return level
    return "Severe"


def compute_traffic_density(
    avg_vehicles_per_frame: float,
    avg_pedestrians_per_frame: float,
) -> float:
    return avg_vehicles_per_frame + 0.5 * avg_pedestrians_per_frame


def compute_per_frame_averages(
    vehicle_count: int,
    pedestrian_count: int,
    sampled_frame_count: int,
) -> tuple[float, float]:
    if sampled_frame_count <= 0:
        return 0.0, 0.0
    return vehicle_count / sampled_frame_count, pedestrian_count / sampled_frame_count


def _apply_congestion_calculations(grouped: pd.DataFrame) -> pd.DataFrame:
    """Add per-frame averages, traffic density, and congestion level columns."""
    averages = grouped.apply(
        lambda row: compute_per_frame_averages(
            int(row["vehicle_count"]),
            int(row["pedestrian_count"]),
            int(row["sampled_frame_count"]),
        ),
        axis=1,
        result_type="expand",
    )
    grouped = grouped.copy()
    grouped["avg_vehicles_per_frame"] = averages[0].round(4)
    grouped["avg_pedestrians_per_frame"] = averages[1].round(4)
    grouped["traffic_density"] = grouped.apply(
        lambda row: round(
            compute_traffic_density(
                row["avg_vehicles_per_frame"],
                row["avg_pedestrians_per_frame"],
            ),
            4,
        ),
        axis=1,
    )
    grouped["congestion_level"] = grouped["traffic_density"].apply(classify_congestion)
    return grouped


def compute_congestion_metrics(detections: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate detections by minute (MVP) or camera_id + video_id + minute (batch).
    """
    df = detections.copy()
    df["minute"] = (df["timestamp"] // 60).astype(int)

    if is_batch_dataset(df):
        group_cols = ["camera_id", "video_id", "minute"]
        output_cols = CONGESTION_COLUMNS_BATCH
    else:
        group_cols = ["minute"]
        output_cols = CONGESTION_COLUMNS_MVP

    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            sampled_frame_count=("frame_id", "nunique"),
            vehicle_count=("object_type", lambda s: s.isin(VEHICLE_TYPES).sum()),
            pedestrian_count=("object_type", lambda s: (s == PEDESTRIAN_TYPE).sum()),
        )
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    grouped = _apply_congestion_calculations(grouped)
    return grouped[output_cols]


def compute_object_summary(detections: pd.DataFrame) -> pd.DataFrame:
    """Compute global detection counts and percentages by object type (MVP)."""
    counts = (
        detections["object_type"]
        .value_counts()
        .rename_axis("object_type")
        .reset_index(name="count")
    )
    total = counts["count"].sum()
    counts["percentage"] = (counts["count"] / total * 100).round(2)
    return counts[SUMMARY_COLUMNS].sort_values("count", ascending=False).reset_index(drop=True)


def compute_object_summary_by_camera(detections: pd.DataFrame) -> pd.DataFrame:
    """Compute detection counts by camera_id and object_type (batch)."""
    if not is_batch_dataset(detections):
        raise MetricsError("Batch summary requires camera_id and video_id columns.")

    grouped = (
        detections.groupby(["camera_id", "object_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    grouped["percentage"] = grouped.groupby("camera_id")["count"].transform(
        lambda s: (s / s.sum() * 100).round(2)
    )
    return grouped[SUMMARY_COLUMNS_BATCH].sort_values(
        ["camera_id", "count"], ascending=[True, False]
    ).reset_index(drop=True)


def compute_object_summary_auto(detections: pd.DataFrame) -> pd.DataFrame:
    """Route to MVP or batch object summary based on input columns."""
    if is_batch_dataset(detections):
        return compute_object_summary_by_camera(detections)
    return compute_object_summary(detections)


def compute_summary_statistics(
    detections: pd.DataFrame,
    congestion_metrics: pd.DataFrame,
) -> SummaryStatistics:
    """Derive high-level summary statistics."""
    total_vehicle = int(detections["object_type"].isin(VEHICLE_TYPES).sum())
    total_pedestrian = int((detections["object_type"] == PEDESTRIAN_TYPE).sum())

    batch = is_batch_dataset(detections)
    camera_count = int(detections["camera_id"].nunique()) if batch else None
    video_count = int(detections["video_id"].nunique()) if batch else None

    if congestion_metrics.empty:
        return SummaryStatistics(
            peak_density=0.0,
            average_density=0.0,
            peak_minute=0,
            total_vehicle_detections=total_vehicle,
            total_pedestrian_detections=total_pedestrian,
            peak_camera=None,
            camera_count=camera_count,
            video_count=video_count,
        )

    peak_idx = congestion_metrics["traffic_density"].idxmax()
    peak_row = congestion_metrics.loc[peak_idx]
    peak_camera = str(peak_row["camera_id"]) if batch and "camera_id" in peak_row else None

    return SummaryStatistics(
        peak_density=float(peak_row["traffic_density"]),
        average_density=float(congestion_metrics["traffic_density"].mean()),
        peak_minute=int(peak_row["minute"]),
        total_vehicle_detections=total_vehicle,
        total_pedestrian_detections=total_pedestrian,
        peak_camera=peak_camera,
        camera_count=camera_count,
        video_count=video_count,
    )


def save_dataframe(df: pd.DataFrame, output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d row(s) -> %s", len(df), output_path)
    return output_path


def print_summary_statistics(stats: SummaryStatistics) -> None:
    logger.info("Summary statistics computed.")
    print("\n=== Traffic Analytics Summary ===")
    for key, value in stats.as_dict().items():
        if value is not None:
            print(f"  {key:28s}: {value}")
    print("=================================\n")


def run_metrics_pipeline(
    input_path: Path = DEFAULT_INPUT,
    congestion_output: Path = DEFAULT_CONGESTION_OUTPUT,
    summary_output: Path = DEFAULT_SUMMARY_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame, SummaryStatistics]:
    """Execute the full metrics pipeline."""
    detections = load_detections(input_path)
    congestion_metrics = compute_congestion_metrics(detections)
    object_summary = compute_object_summary_auto(detections)
    stats = compute_summary_statistics(detections, congestion_metrics)

    save_dataframe(congestion_metrics, congestion_output)
    save_dataframe(object_summary, summary_output)
    print_summary_statistics(stats)

    return congestion_metrics, object_summary, stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute traffic analytics KPIs from YOLO detection CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input detections CSV",
    )
    parser.add_argument(
        "--output",
        "--congestion-output",
        dest="congestion_output",
        type=Path,
        default=DEFAULT_CONGESTION_OUTPUT,
        help="Congestion metrics output CSV",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Object summary output CSV",
    )
    parser.add_argument(
        "--master",
        action="store_true",
        help=(
            "Use batch master defaults "
            "(traffic_detections_master.csv -> congestion_metrics_master.csv)"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    input_path = DEFAULT_MASTER_INPUT if args.master else args.input
    congestion_output = DEFAULT_MASTER_CONGESTION_OUTPUT if args.master else args.congestion_output
    summary_output = DEFAULT_MASTER_SUMMARY_OUTPUT if args.master else args.summary_output

    # When explicit master paths are passed, args.input/output override --master defaults.
    if args.master and args.input == DEFAULT_INPUT:
        input_path = DEFAULT_MASTER_INPUT
    if args.master and args.congestion_output == DEFAULT_CONGESTION_OUTPUT:
        congestion_output = DEFAULT_MASTER_CONGESTION_OUTPUT
    if args.master and args.summary_output == DEFAULT_SUMMARY_OUTPUT:
        summary_output = DEFAULT_MASTER_SUMMARY_OUTPUT

    try:
        run_metrics_pipeline(
            input_path=input_path,
            congestion_output=congestion_output,
            summary_output=summary_output,
        )
    except MetricsError as exc:
        logger.error("Metrics pipeline failed: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    logger.info("Metrics pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
