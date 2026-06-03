#!/usr/bin/env python3
"""
Download batch traffic videos for the Urban Traffic Safety Engine.

Reads data/raw/videos/video_manifest.yaml and downloads Mixkit clips via direct CDN
URLs (no registration required).

Usage:
    python scripts/download_batch_videos.py
    python scripts/download_batch_videos.py --force --verbose
    python scripts/download_batch_videos.py --limit 10
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "raw" / "videos" / "video_manifest.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "videos"

logger = logging.getLogger("download_batch_videos")


@dataclass(frozen=True)
class VideoManifestEntry:
    mixkit_id: int | None
    video_id: str
    camera_id: str
    filename: str
    scene: str
    url: str | None
    local_source: str | None = None


class DownloadError(Exception):
    """Raised when batch download fails critically."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_manifest(manifest_path: Path) -> list[VideoManifestEntry]:
    """Parse the YAML manifest into structured entries."""
    if not manifest_path.exists():
        raise DownloadError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    entries: list[VideoManifestEntry] = []
    for item in raw.get("videos", []):
        mixkit_id = item.get("mixkit_id")
        mixkit_id = int(mixkit_id) if mixkit_id is not None else None
        url = (
            f"https://assets.mixkit.co/videos/{mixkit_id}/{mixkit_id}-720.mp4"
            if mixkit_id is not None
            else None
        )
        entries.append(
            VideoManifestEntry(
                mixkit_id=mixkit_id,
                video_id=str(item["video_id"]),
                camera_id=str(item["camera_id"]),
                filename=str(item["filename"]),
                scene=str(item.get("scene", "unknown")),
                url=url,
                local_source=item.get("local_source"),
            )
        )
    return entries


def download_file(url: str, destination: Path) -> None:
    """Download a single file using curl (handles macOS SSL issues)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if not curl:
        raise DownloadError("curl is required for batch downloads but was not found on PATH.")

    result = subprocess.run(
        [curl, "-fsSL", url, "-o", str(destination)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise OSError(result.stderr.strip() or f"curl failed for {url}")


def download_batch(
    entries: list[VideoManifestEntry],
    output_dir: Path,
    manifest_path: Path,
    force: bool = False,
) -> tuple[list[Path], list[VideoManifestEntry]]:
    """
    Download all manifest entries.

    Returns:
        Tuple of (successful paths, successful manifest entries).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    succeeded: list[VideoManifestEntry] = []
    failed: list[str] = []

    for entry in entries:
        dest = output_dir / entry.filename
        if dest.exists() and not force:
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("Skipping existing: %s (%.2f MB)", dest.name, size_mb)
            downloaded.append(dest)
            succeeded.append(entry)
            continue

        if entry.local_source:
            source = (manifest_path.parent / entry.local_source).resolve()
            if not source.exists():
                logger.warning("Local source missing for %s: %s", entry.filename, source)
                failed.append(entry.filename)
                continue
            logger.info("Copying local video %s -> %s", source.name, dest.name)
            shutil.copy2(source, dest)
            downloaded.append(dest)
            succeeded.append(entry)
            continue

        if not entry.url:
            logger.warning("No download URL for %s", entry.filename)
            failed.append(entry.filename)
            continue

        logger.info("Downloading %s from Mixkit #%s", entry.filename, entry.mixkit_id)
        try:
            download_file(entry.url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            if size_mb < 0.1:
                dest.unlink(missing_ok=True)
                raise OSError("Downloaded file is too small (likely invalid ID)")
            logger.info("Saved %.2f MB -> %s", size_mb, dest.name)
            downloaded.append(dest)
            succeeded.append(entry)
        except OSError as exc:
            logger.warning("Failed to download %s: %s", entry.filename, exc)
            failed.append(entry.filename)

    if failed:
        logger.warning("Failed downloads (%d): %s", len(failed), ", ".join(failed))

    if not downloaded:
        raise DownloadError("No videos were downloaded successfully.")

    logger.info(
        "Batch download complete — %d/%d video(s) available in %s",
        len(downloaded),
        len(entries),
        output_dir,
    )
    return downloaded, succeeded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download batch traffic videos from Mixkit.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="YAML manifest path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save videos",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N manifest entries",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        entries = load_manifest(args.manifest)
        if args.limit:
            entries = entries[: args.limit]
        download_batch(entries, args.output_dir, args.manifest, force=args.force)
    except DownloadError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Next: python scripts/batch_detection.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
