#!/usr/bin/env python3
"""
Train a traffic density forecasting model with weather and event features.

Reads enriched congestion feature records, engineers temporal and categorical
encodings, compares RandomForest vs GradientBoosting regressors, and saves
the best model plus prediction outputs.

Usage:
    python scripts/train_forecast_model.py
    python scripts/train_forecast_model.py --verbose

Prerequisites:
    python scripts/compute_metrics.py --master
    python scripts/generate_external_features.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.ml.train_forecast import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_METRICS_SUPPLEMENT,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    ForecastTrainingError,
    run_forecast_training_pipeline,
)

logger = logging.getLogger("train_forecast_model")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train traffic density forecasting models using weather and event features."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to congestion_features_master.csv",
    )
    parser.add_argument(
        "--metrics-supplement",
        type=Path,
        default=DEFAULT_METRICS_SUPPLEMENT,
        help="Path to congestion_metrics_master.csv for per-frame averages",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model bundle (.pkl)",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to save forecast predictions CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction for model evaluation (default: 0.2)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        result = run_forecast_training_pipeline(
            input_path=args.input,
            model_path=args.model_output,
            predictions_path=args.predictions_output,
            metrics_supplement_path=args.metrics_supplement,
            test_size=args.test_size,
        )
    except ForecastTrainingError as exc:
        logger.error("Training failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during training: %s", exc)
        return 1

    logger.info(
        "Training complete — best model: %s | rows: %d | model: %s | predictions: %s",
        result.best_model_name,
        result.rows_used,
        result.model_path,
        result.predictions_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
