"""Economic delay cost analysis from congestion metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_ANALYSIS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "delay_cost_analysis.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "delay_cost_summary.csv"

VALUE_OF_TIME_USD_PER_HOUR = 18.0

DELAY_MINUTES_BY_CONGESTION: dict[str, float] = {
    "Low": 0.0,
    "Medium": 2.0,
    "High": 5.0,
    "Severe": 10.0,
}

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "camera_id",
        "video_id",
        "minute",
        "vehicle_count",
        "traffic_density",
        "congestion_level",
    }
)

ANALYSIS_COLUMNS: list[str] = [
    "camera_id",
    "video_id",
    "minute",
    "congestion_level",
    "vehicle_count",
    "traffic_density",
    "delay_minutes_per_vehicle",
    "total_delay_minutes",
    "total_delay_hours",
    "estimated_delay_cost_usd",
]

SUMMARY_COLUMNS: list[str] = [
    "camera_id",
    "total_vehicle_count",
    "total_delay_minutes",
    "total_delay_hours",
    "estimated_delay_cost_usd",
    "avg_delay_cost_per_vehicle",
]


class DelayCostError(Exception):
    """Raised when delay cost analysis cannot proceed."""


@dataclass(frozen=True)
class DelayCostResult:
    """Outputs produced by the delay cost pipeline."""

    analysis: pd.DataFrame
    summary: pd.DataFrame
    analysis_path: Path
    summary_path: Path


def load_congestion_metrics(path: Path) -> pd.DataFrame:
    """Load and validate congestion metrics CSV."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise DelayCostError(
            f"Input file not found: {resolved}\n"
            "Run: python scripts/compute_metrics.py --master"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise DelayCostError(f"Input file is empty: {resolved}") from exc
    except OSError as exc:
        raise DelayCostError(f"Failed to read congestion metrics: {exc}") from exc

    if df.empty:
        raise DelayCostError(f"Input file contains no rows: {resolved}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DelayCostError(f"Congestion metrics missing columns: {sorted(missing)}")

    logger.info("Loaded %d congestion row(s) from %s", len(df), resolved.name)
    return df


def lookup_delay_minutes_per_vehicle(congestion_level: str) -> float:
    """Return assumed delay minutes per vehicle for a congestion tier."""
    delay = DELAY_MINUTES_BY_CONGESTION.get(congestion_level)
    if delay is None:
        raise DelayCostError(
            f"Unknown congestion level: {congestion_level!r}. "
            f"Expected one of: {sorted(DELAY_MINUTES_BY_CONGESTION)}"
        )
    return delay


def compute_row_delay_cost(
    vehicle_count: int | float,
    delay_minutes_per_vehicle: float,
    value_of_time_usd_per_hour: float = VALUE_OF_TIME_USD_PER_HOUR,
) -> tuple[float, float, float]:
    """
    Compute total delay minutes, hours, and estimated USD cost for one row.

    Returns:
        (total_delay_minutes, total_delay_hours, estimated_delay_cost_usd)
    """
    total_delay_minutes = float(vehicle_count) * delay_minutes_per_vehicle
    total_delay_hours = total_delay_minutes / 60.0
    estimated_delay_cost_usd = total_delay_hours * value_of_time_usd_per_hour
    return total_delay_minutes, total_delay_hours, estimated_delay_cost_usd


def compute_delay_cost_analysis(
    congestion_df: pd.DataFrame,
    value_of_time_usd_per_hour: float = VALUE_OF_TIME_USD_PER_HOUR,
) -> pd.DataFrame:
    """Apply delay-cost assumptions to each congestion metrics row."""
    df = congestion_df.copy()

    unknown_levels = set(df["congestion_level"].astype(str)) - set(DELAY_MINUTES_BY_CONGESTION)
    if unknown_levels:
        raise DelayCostError(f"Unknown congestion levels in input: {sorted(unknown_levels)}")

    df["delay_minutes_per_vehicle"] = df["congestion_level"].map(DELAY_MINUTES_BY_CONGESTION)
    df["total_delay_minutes"] = df["vehicle_count"] * df["delay_minutes_per_vehicle"]
    df["total_delay_hours"] = df["total_delay_minutes"] / 60.0
    df["estimated_delay_cost_usd"] = df["total_delay_hours"] * value_of_time_usd_per_hour

    analysis = df[
        [
            "camera_id",
            "video_id",
            "minute",
            "congestion_level",
            "vehicle_count",
            "traffic_density",
            "delay_minutes_per_vehicle",
            "total_delay_minutes",
            "total_delay_hours",
            "estimated_delay_cost_usd",
        ]
    ].copy()

    numeric_cols = [
        "traffic_density",
        "delay_minutes_per_vehicle",
        "total_delay_minutes",
        "total_delay_hours",
        "estimated_delay_cost_usd",
    ]
    analysis[numeric_cols] = analysis[numeric_cols].round(4)
    analysis["vehicle_count"] = analysis["vehicle_count"].astype(int)

    logger.info(
        "Computed delay costs for %d row(s) — total estimated cost: $%.2f",
        len(analysis),
        analysis["estimated_delay_cost_usd"].sum(),
    )
    return analysis[ANALYSIS_COLUMNS]


def compute_delay_cost_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate delay costs by camera."""
    if analysis_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        analysis_df.groupby("camera_id", as_index=False)
        .agg(
            total_vehicle_count=("vehicle_count", "sum"),
            total_delay_minutes=("total_delay_minutes", "sum"),
            total_delay_hours=("total_delay_hours", "sum"),
            estimated_delay_cost_usd=("estimated_delay_cost_usd", "sum"),
        )
        .sort_values("camera_id")
        .reset_index(drop=True)
    )

    summary["avg_delay_cost_per_vehicle"] = summary.apply(
        lambda row: (
            row["estimated_delay_cost_usd"] / row["total_vehicle_count"]
            if row["total_vehicle_count"] > 0
            else 0.0
        ),
        axis=1,
    )

    round_cols = [
        "total_delay_minutes",
        "total_delay_hours",
        "estimated_delay_cost_usd",
        "avg_delay_cost_per_vehicle",
    ]
    summary[round_cols] = summary[round_cols].round(4)
    summary["total_vehicle_count"] = summary["total_vehicle_count"].astype(int)

    logger.info("Built delay cost summary for %d camera(s)", len(summary))
    return summary[SUMMARY_COLUMNS]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=False)
    except OSError as exc:
        raise DelayCostError(f"Failed to write CSV: {resolved}: {exc}") from exc
    logger.info("Saved %d row(s) to %s", len(df), resolved.name)


def run_delay_cost_pipeline(
    input_path: Path = DEFAULT_INPUT,
    analysis_output_path: Path = DEFAULT_ANALYSIS_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
    value_of_time_usd_per_hour: float = VALUE_OF_TIME_USD_PER_HOUR,
) -> DelayCostResult:
    """End-to-end economic delay cost analysis pipeline."""
    congestion = load_congestion_metrics(input_path)
    analysis = compute_delay_cost_analysis(
        congestion,
        value_of_time_usd_per_hour=value_of_time_usd_per_hour,
    )
    summary = compute_delay_cost_summary(analysis)

    save_dataframe(analysis, analysis_output_path)
    save_dataframe(summary, summary_output_path)

    return DelayCostResult(
        analysis=analysis,
        summary=summary,
        analysis_path=analysis_output_path.resolve(),
        summary_path=summary_output_path.resolve(),
    )
