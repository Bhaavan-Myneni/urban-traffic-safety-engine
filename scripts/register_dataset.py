#!/usr/bin/env python3
"""
Scan a video directory and generate a YAML dataset manifest.

Usage:
    python scripts/register_dataset.py \\
        --video-dir data/raw/aicity_videos \\
        --dataset aicity

Output:
    data/raw/aicity_videos/aicity_manifest.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.data.datasets import (  # noqa: E402
    DatasetError,
    build_manifest_entry_from_path,
    get_dataset_config,
    scan_video_directory,
    write_manifest,
)

logger = logging.getLogger("register_dataset")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def register_dataset(
    video_dir: Path,
    dataset_name: str,
    manifest_path: Path | None = None,
) -> Path:
    """Scan videos and write manifest YAML."""
    config = get_dataset_config(dataset_name)
    resolved_video_dir = video_dir.expanduser().resolve()
    output_manifest = (manifest_path or config.manifest_path).expanduser().resolve()

    video_files = scan_video_directory(resolved_video_dir)
    entries = [
        build_manifest_entry_from_path(path, config.name, index)
        for index, path in enumerate(video_files, start=1)
    ]

    return write_manifest(output_manifest, entries, dataset_name=config.display_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan a video folder and auto-generate a dataset manifest."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        required=True,
        help="Directory containing source video files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (mixkit, aicity, bdd100k, kaggle_traffic)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override output manifest path (default: dataset registry path)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        path = register_dataset(
            video_dir=args.video_dir,
            dataset_name=args.dataset,
            manifest_path=args.manifest,
        )
        logger.info("Registration complete: %s", path)
    except DatasetError as exc:
        logger.error("Registration failed: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
