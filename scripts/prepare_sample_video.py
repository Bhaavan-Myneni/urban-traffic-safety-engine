#!/usr/bin/env python3
"""
Copy or convert a downloaded BDD100K clip to data/raw/sample_traffic.mp4.

BDD100K videos are typically .mov (H.264). This script copies MP4 files directly
or re-encodes other formats when ffmpeg is available.

Usage:
    python scripts/prepare_sample_video.py ~/Downloads/bdd100k/videos/train/cabc30fc-e7726578.mov
    python scripts/prepare_sample_video.py /path/to/video.mp4 --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"

logger = logging.getLogger("prepare_sample_video")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prepare_sample(source: Path, output: Path, force: bool = False) -> Path:
    """Copy or convert ``source`` video to ``output``."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}. Use --force to overwrite."
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    if source.suffix.lower() == ".mp4":
        logger.info("Copying MP4 source to %s", output)
        shutil.copy2(source, output)
        return output

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        logger.info("Converting %s -> %s via ffmpeg", source, output)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output),
        ]
        subprocess.run(cmd, check=True, capture_output=not logger.isEnabledFor(logging.DEBUG))
        return output

    logger.warning("ffmpeg not found; copying file with .mp4 extension (may fail validation).")
    shutil.copy2(source, output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data/raw/sample_traffic.mp4 from a downloaded dataset clip."
    )
    parser.add_argument("source", type=Path, help="Path to downloaded .mov or .mp4 clip")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    try:
        out = prepare_sample(args.source, args.output, force=args.force)
    except (FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Sample video ready at: %s", out)
    logger.info("Next: python scripts/validate_video.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
