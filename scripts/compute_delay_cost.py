#!/usr/bin/env python3
"""
Compute economic delay costs from batch congestion metrics.

Estimates the dollar cost of traffic delays by camera/intersection using
congestion-tier assumptions and a value-of-time model.

Usage:
    python scripts/compute_delay_cost.py
    python scripts/compute_delay_cost.py --verbose

Prerequisites:
    python scripts/compute_metrics.py --master

Outputs:
    data/processed/delay_cost_analysis.csv
    data/processed/delay_cost_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.analytics.delay_cost import (  # noqa: E402
    DEFAULT_ANALYSIS_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_SUMMARY_OUTPUT,
    VALUE_OF_TIME_USD_PER_HOUR,
    DelayCostError,
    run_delay_cost_pipeline,
)

logger = logging.getLogger("compute_delay_cost")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate economic delay costs from congestion metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to congestion_metrics_master.csv",
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=DEFAULT_ANALYSIS_OUTPUT,
        help="Path for delay_cost_analysis.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for delay_cost_summary.csv",
    )
    parser.add_argument(
        "--value-of-time",
        type=float,
        default=VALUE_OF_TIME_USD_PER_HOUR,
        help="Value of time in USD per hour (default: 18)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def print_summary(result) -> None:
    total_cost = result.analysis["estimated_delay_cost_usd"].sum()
    total_delay_hours = result.analysis["total_delay_hours"].sum()
    top_camera = (
        result.summary.sort_values("estimated_delay_cost_usd", ascending=False)
        .iloc[0]["camera_id"]
        if not result.summary.empty
        else "N/A"
    )

    print("\n=== Economic Delay Cost Summary ===")
    print(f"  Analysis file : {result.analysis_path}")
    print(f"  Summary file  : {result.summary_path}")
    print(f"  Rows analyzed : {len(result.analysis):,}")
    print(f"  Cameras       : {result.summary['camera_id'].nunique()}")
    print(f"  Total delay   : {total_delay_hours:,.2f} hours")
    print(f"  Total cost    : ${total_cost:,.2f}")
    print(f"  Top camera    : {top_camera}")
    print("===================================\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        result = run_delay_cost_pipeline(
            input_path=args.input,
            analysis_output_path=args.analysis_output,
            summary_output_path=args.summary_output,
            value_of_time_usd_per_hour=args.value_of_time,
        )
    except DelayCostError as exc:
        logger.error("Delay cost analysis failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    print_summary(result)
    logger.info("Delay cost analysis completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
