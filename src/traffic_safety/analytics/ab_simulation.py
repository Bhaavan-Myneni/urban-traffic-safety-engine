"""A/B infrastructure simulation for traffic safety interventions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from traffic_safety.analytics.delay_cost import (
    DELAY_MINUTES_BY_CONGESTION,
    VALUE_OF_TIME_USD_PER_HOUR,
    compute_row_delay_cost,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONGESTION_INPUT = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_DELAY_COST_INPUT = PROJECT_ROOT / "data" / "processed" / "delay_cost_analysis.csv"
DEFAULT_NEAR_MISS_INPUT = PROJECT_ROOT / "data" / "processed" / "near_miss_summary.csv"
DEFAULT_METADATA_INPUT = PROJECT_ROOT / "data" / "processed" / "camera_metadata.csv"
DEFAULT_RESULTS_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ab_simulation_results.csv"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "ab_simulation_summary.csv"

SCENARIO_NAME = "protected_bike_lane"

CITY_INTERSECTION_SCENES: frozenset[str] = frozenset({"city_street", "intersection"})
HIGHWAY_SCENES: frozenset[str] = frozenset({"highway"})

RESULT_COLUMNS: list[str] = [
    "camera_id",
    "scene_type",
    "baseline_density",
    "simulated_density",
    "density_change_pct",
    "baseline_vehicle_count",
    "simulated_vehicle_count",
    "baseline_near_miss_risk",
    "simulated_near_miss_risk",
    "risk_reduction_pct",
    "baseline_delay_cost_usd",
    "simulated_delay_cost_usd",
    "delay_cost_change_usd",
    "scenario",
]

SUMMARY_COLUMNS: list[str] = [
    "scene_type",
    "camera_count",
    "baseline_density_avg",
    "simulated_density_avg",
    "baseline_near_miss_risk_total",
    "simulated_near_miss_risk_total",
    "baseline_delay_cost_usd_total",
    "simulated_delay_cost_usd_total",
    "delay_cost_change_usd_total",
    "scenario",
]


@dataclass(frozen=True)
class ScenarioParameters:
    """Intervention effect parameters for a camera scene group."""

    density_change_pct: float
    vehicle_count_change_pct: float
    risk_reduction_pct: float


@dataclass(frozen=True)
class ABSimulationResult:
    """Outputs produced by the A/B simulation pipeline."""

    results: pd.DataFrame
    summary: pd.DataFrame
    results_path: Path
    summary_path: Path


class ABSimulationError(Exception):
    """Raised when the A/B simulation cannot proceed."""


def load_csv(path: Path, label: str, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Load and validate a CSV input file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ABSimulationError(f"Missing {label}: {resolved}")

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise ABSimulationError(f"{label} is empty: {resolved}") from exc
    except OSError as exc:
        raise ABSimulationError(f"Cannot read {label}: {exc}") from exc

    if df.empty:
        raise ABSimulationError(f"{label} contains no rows: {resolved}")

    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ABSimulationError(f"{label} missing columns: {sorted(missing)}")

    logger.info("Loaded %d row(s) from %s", len(df), resolved.name)
    return df


def resolve_scenario_parameters(scene_type: str) -> ScenarioParameters:
    """Return intervention assumptions for a camera scene type."""
    normalized = str(scene_type).strip().lower()

    if normalized in CITY_INTERSECTION_SCENES:
        return ScenarioParameters(
            density_change_pct=3.0,
            vehicle_count_change_pct=-3.0,
            risk_reduction_pct=0.25,
        )
    if normalized in HIGHWAY_SCENES:
        return ScenarioParameters(
            density_change_pct=0.0,
            vehicle_count_change_pct=0.0,
            risk_reduction_pct=0.0,
        )
    return ScenarioParameters(
        density_change_pct=-2.0,
        vehicle_count_change_pct=0.0,
        risk_reduction_pct=0.10,
    )


def build_camera_baseline(
    congestion_df: pd.DataFrame,
    delay_cost_df: pd.DataFrame,
    near_miss_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge baseline metrics at the camera level."""
    congestion_agg = (
        congestion_df.groupby("camera_id", as_index=False)
        .agg(
            baseline_density=("traffic_density", "mean"),
            baseline_vehicle_count=("vehicle_count", "sum"),
            congestion_level=("congestion_level", lambda s: s.mode().iloc[0]),
        )
    )

    delay_cost_agg = (
        delay_cost_df.groupby("camera_id", as_index=False)
        .agg(baseline_delay_cost_usd=("estimated_delay_cost_usd", "sum"))
    )

    near_miss_agg = (
        near_miss_df.groupby("camera_id", as_index=False)
        .agg(baseline_near_miss_risk=("near_miss_count", "sum"))
    )

    metadata = metadata_df[["camera_id", "scene_type"]].drop_duplicates()

    baseline = metadata.merge(congestion_agg, on="camera_id", how="left")
    baseline = baseline.merge(delay_cost_agg, on="camera_id", how="left")
    baseline = baseline.merge(near_miss_agg, on="camera_id", how="left")

    baseline["baseline_density"] = baseline["baseline_density"].fillna(0.0)
    baseline["baseline_vehicle_count"] = baseline["baseline_vehicle_count"].fillna(0).astype(int)
    baseline["baseline_delay_cost_usd"] = baseline["baseline_delay_cost_usd"].fillna(0.0)
    baseline["baseline_near_miss_risk"] = baseline["baseline_near_miss_risk"].fillna(0).astype(int)
    baseline["congestion_level"] = baseline["congestion_level"].fillna("Low")

    if baseline["camera_id"].isna().any():
        raise ABSimulationError("Camera metadata join produced null camera_id values.")

    logger.info("Built baseline metrics for %d camera(s)", len(baseline))
    return baseline


def apply_scenario_to_camera(row: pd.Series) -> pd.Series:
    """Apply protected bike lane assumptions to a single camera row."""
    params = resolve_scenario_parameters(str(row["scene_type"]))

    density_multiplier = 1.0 + (params.density_change_pct / 100.0)
    vehicle_multiplier = 1.0 + (params.vehicle_count_change_pct / 100.0)

    simulated_density = float(row["baseline_density"]) * density_multiplier
    simulated_vehicle_count = max(
        0,
        int(round(float(row["baseline_vehicle_count"]) * vehicle_multiplier)),
    )

    baseline_near_miss = int(row["baseline_near_miss_risk"])
    simulated_near_miss = round(baseline_near_miss * (1.0 - params.risk_reduction_pct))

    congestion_level = str(row["congestion_level"])
    delay_minutes_per_vehicle = DELAY_MINUTES_BY_CONGESTION.get(congestion_level, 0.0)
    _, _, simulated_delay_cost_usd = compute_row_delay_cost(
        vehicle_count=simulated_vehicle_count,
        delay_minutes_per_vehicle=delay_minutes_per_vehicle,
        value_of_time_usd_per_hour=VALUE_OF_TIME_USD_PER_HOUR,
    )

    baseline_delay_cost_usd = float(row["baseline_delay_cost_usd"])

    return pd.Series(
        {
            "simulated_density": simulated_density,
            "density_change_pct": params.density_change_pct,
            "simulated_vehicle_count": simulated_vehicle_count,
            "simulated_near_miss_risk": simulated_near_miss,
            "risk_reduction_pct": params.risk_reduction_pct,
            "simulated_delay_cost_usd": simulated_delay_cost_usd,
            "delay_cost_change_usd": simulated_delay_cost_usd - baseline_delay_cost_usd,
            "scenario": SCENARIO_NAME,
        }
    )


def run_ab_simulation(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate intervention outcomes for all cameras."""
    simulated_cols = baseline_df.apply(apply_scenario_to_camera, axis=1)
    results = pd.concat([baseline_df, simulated_cols], axis=1)

    output = results[
        [
            "camera_id",
            "scene_type",
            "baseline_density",
            "simulated_density",
            "density_change_pct",
            "baseline_vehicle_count",
            "simulated_vehicle_count",
            "baseline_near_miss_risk",
            "simulated_near_miss_risk",
            "risk_reduction_pct",
            "baseline_delay_cost_usd",
            "simulated_delay_cost_usd",
            "delay_cost_change_usd",
            "scenario",
        ]
    ].copy()

    round_cols = [
        "baseline_density",
        "simulated_density",
        "baseline_delay_cost_usd",
        "simulated_delay_cost_usd",
        "delay_cost_change_usd",
    ]
    output[round_cols] = output[round_cols].round(4)
    output["risk_reduction_pct"] = output["risk_reduction_pct"].round(4)

    logger.info(
        "Simulation complete — delay cost change: $%.2f | near-miss reduction: %d",
        output["delay_cost_change_usd"].sum(),
        int(output["baseline_near_miss_risk"].sum() - output["simulated_near_miss_risk"].sum()),
    )
    return output[RESULT_COLUMNS]


def build_simulation_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation outcomes by scene type."""
    if results_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        results_df.groupby("scene_type", as_index=False)
        .agg(
            camera_count=("camera_id", "count"),
            baseline_density_avg=("baseline_density", "mean"),
            simulated_density_avg=("simulated_density", "mean"),
            baseline_near_miss_risk_total=("baseline_near_miss_risk", "sum"),
            simulated_near_miss_risk_total=("simulated_near_miss_risk", "sum"),
            baseline_delay_cost_usd_total=("baseline_delay_cost_usd", "sum"),
            simulated_delay_cost_usd_total=("simulated_delay_cost_usd", "sum"),
            delay_cost_change_usd_total=("delay_cost_change_usd", "sum"),
            scenario=("scenario", "first"),
        )
        .sort_values("scene_type")
        .reset_index(drop=True)
    )

    round_cols = [
        "baseline_density_avg",
        "simulated_density_avg",
        "baseline_delay_cost_usd_total",
        "simulated_delay_cost_usd_total",
        "delay_cost_change_usd_total",
    ]
    summary[round_cols] = summary[round_cols].round(4)

    logger.info("Built simulation summary for %d scene type(s)", len(summary))
    return summary[SUMMARY_COLUMNS]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=False)
    except OSError as exc:
        raise ABSimulationError(f"Failed to write CSV: {resolved}: {exc}") from exc
    logger.info("Saved %d row(s) to %s", len(df), resolved.name)


def run_ab_simulation_pipeline(
    congestion_path: Path = DEFAULT_CONGESTION_INPUT,
    delay_cost_path: Path = DEFAULT_DELAY_COST_INPUT,
    near_miss_path: Path = DEFAULT_NEAR_MISS_INPUT,
    metadata_path: Path = DEFAULT_METADATA_INPUT,
    results_output_path: Path = DEFAULT_RESULTS_OUTPUT,
    summary_output_path: Path = DEFAULT_SUMMARY_OUTPUT,
) -> ABSimulationResult:
    """End-to-end protected bike lane A/B simulation pipeline."""
    congestion_df = load_csv(
        congestion_path,
        "congestion metrics",
        ["camera_id", "traffic_density", "vehicle_count", "congestion_level"],
    )
    delay_cost_df = load_csv(
        delay_cost_path,
        "delay cost analysis",
        ["camera_id", "estimated_delay_cost_usd"],
    )
    near_miss_df = load_csv(
        near_miss_path,
        "near-miss summary",
        ["camera_id", "near_miss_count"],
    )
    metadata_df = load_csv(
        metadata_path,
        "camera metadata",
        ["camera_id", "scene_type"],
    )

    baseline = build_camera_baseline(congestion_df, delay_cost_df, near_miss_df, metadata_df)
    results = run_ab_simulation(baseline)
    summary = build_simulation_summary(results)

    save_dataframe(results, results_output_path)
    save_dataframe(summary, summary_output_path)

    return ABSimulationResult(
        results=results,
        summary=summary,
        results_path=results_output_path.resolve(),
        summary_path=summary_output_path.resolve(),
    )
