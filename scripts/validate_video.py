#!/usr/bin/env python3
"""
Validate a traffic video file for the Urban Traffic Safety Engine pipeline.

Checks that the file exists, can be opened with OpenCV, and reports metadata
(frame count, FPS, duration, resolution, codec fourcc).

Usage:
    python scripts/validate_video.py
    python scripts/validate_video.py --path data/raw/sample_traffic.mp4
    python scripts/validate_video.py --path data/raw/sample_traffic.mp4 --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

# Project root is two levels up from this script (scripts/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"

logger = logging.getLogger("validate_video")


@dataclass(frozen=True)
class VideoMetadata:
    """Structured metadata extracted from a video file."""

    path: Path
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    fourcc: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "path": str(self.path),
            "frame_count": self.frame_count,
            "fps": round(self.fps, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "fourcc": self.fourcc,
        }


class VideoValidationError(Exception):
    """Raised when a video fails one or more validation checks."""


def _decode_fourcc(raw_fourcc: float) -> str:
    """Convert OpenCV's numeric FOURCC code to a readable four-character string."""
    if raw_fourcc <= 0:
        return "unknown"
    code = int(raw_fourcc)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "unknown"


def validate_video(path: Path) -> VideoMetadata:
    """
    Validate that ``path`` exists and is readable as a video via OpenCV.

    Returns:
        VideoMetadata on success.

    Raises:
        VideoValidationError: If the file is missing or cannot be read as video.
    """
    resolved = path.expanduser().resolve()

    if not resolved.exists():
        raise VideoValidationError(
            f"Video file not found: {resolved}\n"
            f"Expected location: {DEFAULT_VIDEO_PATH}\n"
            "Run: python scripts/download_sample_video.py"
        )

    if not resolved.is_file():
        raise VideoValidationError(f"Path is not a file: {resolved}")

    if resolved.stat().st_size == 0:
        raise VideoValidationError(f"Video file is empty (0 bytes): {resolved}")

    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        raise VideoValidationError(
            f"OpenCV could not open the video: {resolved}\n"
            "The file may be corrupt or use an unsupported codec. "
            "Try re-encoding with: ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4"
        )

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = _decode_fourcc(float(cap.get(cv2.CAP_PROP_FOURCC)))

        # Confirm at least one frame is decodable (guards against broken headers).
        ok, _ = cap.read()
        if not ok:
            raise VideoValidationError(
                f"Video opened but no decodable frames found: {resolved}"
            )

        if frame_count <= 0:
            raise VideoValidationError(
                f"Invalid frame count ({frame_count}) for: {resolved}"
            )

        if fps <= 0:
            raise VideoValidationError(
                f"Invalid FPS ({fps}) for: {resolved}. "
                "The container metadata may be incomplete."
            )

        duration_seconds = frame_count / fps

        return VideoMetadata(
            path=resolved,
            frame_count=frame_count,
            fps=fps,
            duration_seconds=duration_seconds,
            width=width,
            height=height,
            fourcc=fourcc,
        )
    finally:
        cap.release()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _print_metadata(metadata: VideoMetadata) -> None:
    logger.info("Video validation succeeded.")
    print("\n=== Video Metadata ===")
    for key, value in metadata.as_dict().items():
        print(f"  {key:20s}: {value}")
    print("======================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a traffic video for the Urban Traffic Safety Engine."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to the video file (default: {DEFAULT_VIDEO_PATH.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    logger.info("Validating video at: %s", args.path)

    try:
        metadata = validate_video(args.path)
    except VideoValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except cv2.error as exc:
        logger.exception("OpenCV error while reading video: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("Filesystem error: %s", exc)
        return 1

    _print_metadata(metadata)
    return 0


if __name__ == "__main__":
    sys.exit(main())
