#!/usr/bin/env python3
"""
Download a free traffic sample video for the Urban Traffic Safety Engine MVP.

Source: Mixkit (no registration, direct CDN, Mixkit Free License).
Default clip: busy intersection with cars, trucks, motorcycles, and pedestrians.

Usage:
    python scripts/download_sample_video.py
    python scripts/download_sample_video.py --extended   # ~2 min, multiple clips
    python scripts/download_sample_video.py --force
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
import ssl
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"

# Mixkit Free License clips — direct CDN URLs (no registration).
# https://mixkit.co/free-stock-video/traffic/
PRIMARY_CLIP = {
    "url": "https://assets.mixkit.co/videos/57/57-720.mp4",
    "name": "intersection-fast-motion",
    "description": "Busy intersection — cars, trucks, motorcycles, people (33 s)",
}

EXTENDED_CLIPS = [
    PRIMARY_CLIP,
    {
        "url": "https://assets.mixkit.co/videos/100/100-720.mp4",
        "name": "one-way-city-traffic",
        "description": "One-way city street traffic (12 s)",
    },
    {
        "url": "https://assets.mixkit.co/videos/4401/4401-720.mp4",
        "name": "junction-crossing",
        "description": "Crowds and cars at a street junction (25 s)",
    },
    {
        "url": "https://assets.mixkit.co/videos/4000/4000-720.mp4",
        "name": "busy-street",
        "description": "Fast-motion busy city street (15 s)",
    },
    {
        "url": "https://assets.mixkit.co/videos/3418/3418-720.mp4",
        "name": "highway-sunset",
        "description": "Highway with cars at sunset (15 s)",
    },
]

logger = logging.getLogger("download_sample_video")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def download_file(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination`` using curl (preferred) or urllib."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)

    curl = shutil.which("curl")
    if curl:
        result = subprocess.run(
            [curl, "-fsSL", url, "-o", str(destination)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed: {result.stderr.strip()}")
    else:
        # Fallback for environments without curl; use unverified SSL on macOS dev setups.
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(url, context=ctx) as response, destination.open("wb") as out:
                out.write(response.read())
        except ssl.SSLCertVerificationError:
            logger.warning("SSL verification failed; retrying with curl or unverified context.")
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=ctx) as response, destination.open("wb") as out:
                out.write(response.read())

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.info("Saved %.2f MB -> %s", size_mb, destination)
    return destination


def concatenate_videos(sources: list[Path], output: Path) -> Path:
    """Concatenate video clips into a single MP4 using OpenCV (no ffmpeg required)."""
    if not sources:
        raise ValueError("No source videos provided for concatenation.")

    writer: cv2.VideoWriter | None = None
    total_frames = 0
    fps = 30.0
    size: tuple[int, int] | None = None

    for src in sources:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open clip for concatenation: {src}")

        clip_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if size is None:
            size = (width, height)
            fps = clip_fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, size)
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot create output video: {output}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            total_frames += 1

        cap.release()
        logger.info("Appended clip: %s", src.name)

    if writer is not None:
        writer.release()

    duration = total_frames / fps if fps else 0
    logger.info(
        "Concatenated %d clip(s) -> %s (%.1f s, %d frames)",
        len(sources),
        output,
        duration,
        total_frames,
    )
    return output


def download_sample(output: Path, extended: bool = False, force: bool = False) -> Path:
    """Download and optionally concatenate Mixkit traffic clips."""
    output = output.expanduser().resolve()

    if output.exists() and not force:
        raise FileExistsError(
            f"Output already exists: {output}. Use --force to overwrite."
        )

    clips = EXTENDED_CLIPS if extended else [PRIMARY_CLIP]
    temp_dir = PROJECT_ROOT / "data" / "interim" / "download_clips"
    temp_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for clip in clips:
        dest = temp_dir / f"{clip['name']}.mp4"
        if not dest.exists() or force:
            download_file(clip["url"], dest)
        else:
            logger.info("Reusing cached clip: %s", dest)
        downloaded.append(dest)

    if len(downloaded) == 1:
        shutil.copy2(downloaded[0], output)
    else:
        concatenate_videos(downloaded, output)

    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a free traffic sample video (Mixkit, no registration)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Download and concatenate ~5 clips for a ~2-minute sample video.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    try:
        path = download_sample(args.output, extended=args.extended, force=args.force)
    except (FileExistsError, RuntimeError, urllib.error.URLError, OSError) as exc:
        logger.error("Download failed: %s", exc)
        return 1

    logger.info("Sample video ready: %s", path)
    logger.info("Next: python scripts/validate_video.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
