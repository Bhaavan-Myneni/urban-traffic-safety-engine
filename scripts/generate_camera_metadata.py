#!/usr/bin/env python3
"""
Generate simulated camera geospatial metadata for the Urban Traffic Safety Engine.

Reads the batch video manifest and assigns each camera deterministic coordinates
around Bloomington, Indiana for portfolio geospatial analytics.

Usage:
    python scripts/generate_camera_metadata.py
    python scripts/generate_camera_metadata.py --verbose
    python scripts/generate_camera_metadata.py --output data/processed/camera_metadata.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("generate_camera_metadata")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "raw" / "videos" / "video_manifest.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "camera_metadata.csv"

# Bloomington, Indiana — downtown / IU campus reference point
BLOOMINGTON_CENTER_LAT = 39.1653
BLOOMINGTON_CENTER_LON = -86.5264

METADATA_COLUMNS = [
    "camera_id",
    "camera_name",
    "scene_type",
    "latitude",
    "longitude",
    "road_type",
]

SCENE_TO_ROAD_TYPE: dict[str, str] = {
    "intersection": "signalized_intersection",
    "city_street": "urban_arterial",
    "highway": "highway",
    "night_city": "urban_arterial",
    "tunnel": "tunnel",
    "aerial": "aerial_corridor",
    "dashcam": "local_street",
    "composite": "mixed_corridor",
}

# Scene clusters — approximate Bloomington neighborhoods / corridors
SCENE_ANCHORS: dict[str, tuple[float, float]] = {
    "intersection": (39.1674, -86.5340),   # Kirkwood Ave / downtown
    "city_street": (39.1610, -86.5290),   # Near College Mall corridor
    "highway": (39.1820, -86.5150),       # I-69 / SR-45 area (north)
    "night_city": (39.1688, -86.5385),    # Downtown nightlife district
    "tunnel": (39.1585, -86.5220),        # East-side underpass corridor
    "aerial": (39.1720, -86.5200),       # Campus / stadium aerial zone
    "dashcam": (39.1635, -86.5410),       # Residential collector streets
    "composite": (39.1653, -86.5264),    # Central reference
}


class MetadataGenerationError(Exception):
    """Raised when camera metadata cannot be generated."""


@dataclass(frozen=True)
class CameraRecord:
    camera_id: str
    scene_type: str


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_manifest_cameras(manifest_path: Path) -> list[CameraRecord]:
    """Parse unique cameras from the video manifest YAML."""
    resolved = manifest_path.expanduser().resolve()
    if not resolved.exists():
        raise MetadataGenerationError(f"Manifest not found: {resolved}")

    try:
        with resolved.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise MetadataGenerationError(f"Invalid YAML in {resolved}: {exc}") from exc
    except OSError as exc:
        raise MetadataGenerationError(f"Cannot read manifest: {exc}") from exc

    if not raw or "videos" not in raw:
        raise MetadataGenerationError(f"No videos found in manifest: {resolved}")

    seen: set[str] = set()
    cameras: list[CameraRecord] = []

    for item in raw["videos"]:
        camera_id = str(item.get("camera_id", "")).strip()
        if not camera_id or camera_id in seen:
            continue
        seen.add(camera_id)
        scene_type = str(item.get("scene", "city_street")).strip().lower()
        cameras.append(CameraRecord(camera_id=camera_id, scene_type=scene_type))

    if not cameras:
        raise MetadataGenerationError("Manifest contains no camera entries.")

    cameras.sort(key=lambda record: record.camera_id)
    logger.info("Loaded %d unique camera(s) from %s", len(cameras), resolved.name)
    return cameras


def camera_id_to_name(camera_id: str) -> str:
    """Convert CAM_CITY_STREET_01 -> City Street Cam 01."""
    body = camera_id.removeprefix("CAM_")
    parts = body.split("_")
    if parts and parts[-1].isdigit():
        number = parts[-1]
        label = " ".join(parts[:-1]).title()
        return f"{label} Cam {number}"
    return body.replace("_", " ").title()


def deterministic_jitter(camera_id: str, scale: float = 0.012) -> tuple[float, float]:
    """Return reproducible lat/lon offsets derived from camera_id."""
    digest = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()
    lat_seed = int(digest[:8], 16) / 0xFFFFFFFF
    lon_seed = int(digest[8:16], 16) / 0xFFFFFFFF
    lat_offset = (lat_seed - 0.5) * scale
    lon_offset = (lon_seed - 0.5) * scale
    return lat_offset, lon_offset


def simulate_coordinates(camera_id: str, scene_type: str) -> tuple[float, float]:
    """Assign simulated Bloomington coordinates with scene-based clustering."""
    anchor_lat, anchor_lon = SCENE_ANCHORS.get(
        scene_type,
        (BLOOMINGTON_CENTER_LAT, BLOOMINGTON_CENTER_LON),
    )
    lat_jitter, lon_jitter = deterministic_jitter(camera_id)
    latitude = round(anchor_lat + lat_jitter, 6)
    longitude = round(anchor_lon + lon_jitter, 6)
    return latitude, longitude


def resolve_road_type(scene_type: str) -> str:
    """Map manifest scene labels to a road infrastructure category."""
    return SCENE_TO_ROAD_TYPE.get(scene_type, "urban_collector")


def build_metadata_dataframe(cameras: list[CameraRecord]) -> pd.DataFrame:
    """Create the camera metadata dataframe."""
    rows: list[dict[str, object]] = []

    for camera in cameras:
        latitude, longitude = simulate_coordinates(camera.camera_id, camera.scene_type)
        rows.append(
            {
                "camera_id": camera.camera_id,
                "camera_name": camera_id_to_name(camera.camera_id),
                "scene_type": camera.scene_type,
                "latitude": latitude,
                "longitude": longitude,
                "road_type": resolve_road_type(camera.scene_type),
            }
        )

    return pd.DataFrame(rows, columns=METADATA_COLUMNS)


def save_metadata(df: pd.DataFrame, output_path: Path) -> None:
    """Write metadata CSV to disk."""
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(resolved, index=False)
    except OSError as exc:
        raise MetadataGenerationError(f"Failed to write metadata CSV: {exc}") from exc
    logger.info("Saved camera metadata (%d rows) to %s", len(df), resolved)


def run_metadata_generation(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Generate and persist camera metadata."""
    cameras = load_manifest_cameras(manifest_path)
    metadata_df = build_metadata_dataframe(cameras)
    save_metadata(metadata_df, output_path)
    return metadata_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulated Bloomington camera geospatial metadata."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to video_manifest.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for camera metadata",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        df = run_metadata_generation(
            manifest_path=args.manifest,
            output_path=args.output,
        )
    except MetadataGenerationError as exc:
        logger.error("Metadata generation failed: %s", exc)
        return 1

    print(f"\nGenerated {len(df)} camera metadata record(s) -> {args.output.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
