#!/usr/bin/env python3
"""
Detect pedestrian/bicycle vs vehicle near-miss events from batch YOLO output.

Usage:
    python scripts/detect_near_misses.py
    python scripts/detect_near_misses.py --verbose

Prerequisites:
    python scripts/batch_detection.py

Outputs:
    data/processed/near_miss_events.csv
    data/processed/near_miss_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.analytics.near_miss import (  # noqa: E402
    DEFAULT_EVENTS_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_RISK_CONFIDENCE_MIN,
    DEFAULT_SUMMARY_OUTPUT,
    DEFAULT_VEHICLE_CONFIDENCE_MIN,
    NearMissError,
    run_near_miss_pipeline,
)

logger = logging.getLogger("detect_near_misses")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect near-miss events from traffic detection CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to traffic_detections_master.csv",
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=DEFAULT_EVENTS_OUTPUT,
        help="Path for near_miss_events.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Path for near_miss_summary.csv",
    )
    parser.add_argument(
        "--vehicle-confidence-min",
        type=float,
        default=DEFAULT_VEHICLE_CONFIDENCE_MIN,
        help="Minimum vehicle detection confidence (default: 0.35)",
    )
    parser.add_argument(
        "--risk-confidence-min",
        type=float,
        default=DEFAULT_RISK_CONFIDENCE_MIN,
        help="Minimum risk-object detection confidence (default: 0.35)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def print_summary(result) -> None:
    print("\n=== Near-Miss Detection Summary ===")
    print(f"  Events file : {result.events_path}")
    print(f"  Summary file: {result.summary_path}")
    print(f"  Total events: {len(result.events):,}")
    if not result.events.empty:
        high_count = int((result.events["risk_level"] == "High").sum())
        medium_count = int((result.events["risk_level"] == "Medium").sum())
        print(f"  High risk   : {high_count:,}")
        print(f"  Medium risk : {medium_count:,}")
        print(f"  Cameras     : {result.events['camera_id'].nunique()}")
    print("===================================\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        result = run_near_miss_pipeline(
            input_path=args.input,
            events_output_path=args.events_output,
            summary_output_path=args.summary_output,
            vehicle_confidence_min=args.vehicle_confidence_min,
            risk_confidence_min=args.risk_confidence_min,
        )
    except NearMissError as exc:
        logger.error("Near-miss detection failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    print_summary(result)
    logger.info("Near-miss detection completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
