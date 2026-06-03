#!/usr/bin/env python3
"""
Run A/B infrastructure simulation for the Urban Traffic Safety Engine.

Simulates a protected bike lane intervention and estimates impacts on
traffic density, delay cost, and pedestrian/cyclist safety.

Usage:
    python scripts/run_ab_simulation.py
    python scripts/run_ab_simulation.py --verbose

Prerequisites:
    python scripts/compute_metrics.py --master
    python scripts/compute_delay_cost.py
    python scripts/detect_near_misses.py
    python scripts/generate_camera_metadata.py

Outputs:
    data/processed/ab_simulation_results.csv
    data/processed/ab_simulation_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.analytics.ab_simulation import (  # noqa: E402
    DEFAULT_CONGESTION_INPUT,
    DEFAULT_DELAY_COST_INPUT,
    DEFAULT_METADATA_INPUT,
    DEFAULT_NEAR_MISS_INPUT,
    DEFAULT_RESULTS_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    ABSimulationError,
    run_ab_simulation_pipeline,
)

logger = logging.getLogger("run_ab_simulation")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate protected bike lane infrastructure A/B outcomes."
    )
    parser.add_argument(
        "--congestion",
        type=Path,
        default=DEFAULT_CONGESTION_INPUT,
        help="Path to congestion_metrics_master.csv",
    )
    parser.add_argument(
        "--delay-cost",
        type=Path,
        default=DEFAULT_DELAY_COST_INPUT,
        help="Path to delay_cost_analysis.csv",
    )
    parser.add_argument(
        "--near-miss",
        type=Path,
        default=DEFAULT_NEAR_MISS_INPUT,
        help="Path to near_miss_summary.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_INPUT,
        help="Path to camera_metadata.csv",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT,
        help="Path for ab_simulation_results.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for ab_simulation_summary.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def print_summary(result) -> None:
    baseline_cost = result.results["baseline_delay_cost_usd"].sum()
    simulated_cost = result.results["simulated_delay_cost_usd"].sum()
    cost_change = result.results["delay_cost_change_usd"].sum()
    baseline_risk = int(result.results["baseline_near_miss_risk"].sum())
    simulated_risk = int(result.results["simulated_near_miss_risk"].sum())

    print("\n=== A/B Simulation Summary (Protected Bike Lane) ===")
    print(f"  Results file : {result.results_path}")
    print(f"  Summary file : {result.summary_path}")
    print(f"  Cameras      : {len(result.results):,}")
    print(f"  Near-misses  : {baseline_risk:,} -> {simulated_risk:,}")
    print(f"  Delay cost   : ${baseline_cost:,.2f} -> ${simulated_cost:,.2f}")
    print(f"  Cost change  : ${cost_change:,.2f}")
    print("====================================================\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        result = run_ab_simulation_pipeline(
            congestion_path=args.congestion,
            delay_cost_path=args.delay_cost,
            near_miss_path=args.near_miss,
            metadata_path=args.metadata,
            results_output_path=args.results_output,
            summary_output_path=args.summary_output,
        )
    except ABSimulationError as exc:
        logger.error("A/B simulation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    print_summary(result)
    logger.info("A/B simulation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
