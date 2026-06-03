#!/usr/bin/env python3
"""
End-to-end analytics pipeline for a registered traffic video dataset.

Pipeline:
    YOLO Detection -> Congestion Metrics -> Near-Miss -> Delay Cost

Usage:
    python scripts/process_dataset.py --dataset aicity
    python scripts/process_dataset.py --dataset mixkit --stride 2

Outputs (per dataset):
    traffic_detections_<dataset>.csv
    congestion_metrics_<dataset>.csv
    object_summary_<dataset>.csv
    near_miss_events_<dataset>.csv
    delay_cost_analysis_<dataset>.csv
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.analytics.delay_cost import run_delay_cost_pipeline  # noqa: E402
from traffic_safety.analytics.metrics import run_metrics_pipeline  # noqa: E402
from traffic_safety.analytics.near_miss import run_near_miss_pipeline  # noqa: E402
from traffic_safety.data.datasets import (  # noqa: E402
    DatasetError,
    ensure_processed_dir,
    get_dataset_config,
    load_manifest,
    log_manifest_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger("process_dataset")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_batch_detection_step(
    config,
    *,
    stride: int,
    workers: int | None,
    model: str,
    conf: float,
    verbose: bool,
) -> None:
    """Invoke batch_detection.py for a dataset."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_detection.py"),
        "--manifest",
        str(config.manifest_path),
        "--video-dir",
        str(config.video_dir),
        "--output",
        str(config.detections_csv()),
        "--model",
        model,
        "--stride",
        str(stride),
        "--conf",
        str(conf),
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    if verbose:
        cmd.append("--verbose")

    logger.info("Running batch detection for %s", config.display_name)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    if result.returncode != 0:
        raise DatasetError(
            f"Batch detection failed for {config.name} (exit code {result.returncode})."
        )
    if not config.detections_csv().exists():
        raise DatasetError(f"Detection output not created: {config.detections_csv()}")


def run_analytics_steps(config, *, skip_detection: bool) -> None:
    """Run metrics, near-miss, and delay-cost pipelines."""
    detections = config.detections_csv()
    if not skip_detection and not detections.exists():
        raise DatasetError(f"Detections file missing: {detections}")

    if not detections.exists():
        raise DatasetError(
            f"No detections at {detections}. Run without --skip-detection first."
        )

    logger.info("Computing congestion metrics")
    run_metrics_pipeline(
        input_path=detections,
        congestion_output=config.congestion_csv(),
        summary_output=config.summary_csv(),
    )

    near_miss_summary = (
        PROJECT_ROOT / "data" / "processed" / "near_miss_summary.csv"
        if config.legacy_master
        else config.processed_dir / f"near_miss_summary_{config.name}.csv"
    )
    delay_summary = (
        PROJECT_ROOT / "data" / "processed" / "delay_cost_summary.csv"
        if config.legacy_master
        else config.processed_dir / f"delay_cost_summary_{config.name}.csv"
    )

    logger.info("Detecting near-miss events")
    run_near_miss_pipeline(
        input_path=detections,
        events_output_path=config.near_miss_events_csv(),
        summary_output_path=near_miss_summary,
    )

    logger.info("Computing delay cost analysis")
    run_delay_cost_pipeline(
        input_path=config.congestion_csv(),
        analysis_output_path=config.delay_cost_csv(),
        summary_output_path=delay_summary,
    )


def process_dataset(
    dataset_name: str,
    *,
    stride: int = 2,
    workers: int | None = None,
    model: str = "yolov8n.pt",
    conf: float = 0.25,
    skip_detection: bool = False,
    verbose: bool = False,
) -> None:
    """Execute full pipeline for one dataset."""
    config = get_dataset_config(dataset_name)
    ensure_processed_dir(config)

    result = load_manifest(config.manifest_path, config.video_dir, require_videos=True)
    log_manifest_statistics(result, dataset_label=config.display_name)

    if not skip_detection:
        run_batch_detection_step(
            config,
            stride=stride,
            workers=workers,
            model=model,
            conf=conf,
            verbose=verbose,
        )
    else:
        logger.info("Skipping detection step (--skip-detection)")

    run_analytics_steps(config, skip_detection=skip_detection)
    logger.info("Pipeline complete for %s", config.display_name)
    logger.info("  Detections : %s", config.detections_csv())
    logger.info("  Congestion : %s", config.congestion_csv())
    logger.info("  Near-miss  : %s", config.near_miss_events_csv())
    logger.info("  Delay cost : %s", config.delay_cost_csv())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end detection and analytics for a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name: mixkit, aicity, or bdd100k",
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Reuse existing detection CSV and run analytics only",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        process_dataset(
            args.dataset,
            stride=args.stride,
            workers=args.workers,
            model=args.model,
            conf=args.conf,
            skip_detection=args.skip_detection,
            verbose=args.verbose,
        )
    except DatasetError as exc:
        logger.error("Dataset pipeline failed: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
