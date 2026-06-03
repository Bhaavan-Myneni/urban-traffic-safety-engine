#!/usr/bin/env python3
"""
Compare Mixkit, AI City Challenge, and BDD100K dataset analytics.

Usage:
    python scripts/compare_datasets.py
    python scripts/compare_datasets.py --verbose

Output:
    data/processed/dataset_comparison.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.data.dataset_comparison import (  # noqa: E402
    COMPARISON_OUTPUT,
    build_comparison_dataframe,
    count_ready_datasets,
    save_comparison_csv,
)
from traffic_safety.data.datasets import DatasetError  # noqa: E402

logger = logging.getLogger("compare_datasets")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_comparison_summary(df) -> None:
    ready = count_ready_datasets(df)
    print("\n=== Dataset Comparison ===")
    print(f"  Datasets compared : {len(df)}")
    print(f"  With detections   : {ready}")
    for _, row in df.iterrows():
        print(
            f"  {row['display_name']:28} | videos={row['video_count']:3} | "
            f"detections={row['total_detections']:,} | near_miss={row['near_miss_count']}"
        )
    print("==========================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare traffic datasets and write comparison CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=COMPARISON_OUTPUT,
        help="Output CSV path",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets (default: all registered)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        output = save_comparison_csv(args.output, args.datasets)
        df = build_comparison_dataframe(args.datasets)
        print_comparison_summary(df)
        logger.info("Comparison saved -> %s", output)

        if count_ready_datasets(df) < 2:
            logger.warning(
                "Fewer than two datasets have processed detections. "
                "Run process_dataset.py for additional sources."
            )
    except DatasetError as exc:
        logger.error("Comparison failed: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
