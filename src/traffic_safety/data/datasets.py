"""Dataset registry, manifest I/O, and path resolution for multi-source video pipelines."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm"})

REQUIRED_MANIFEST_FIELDS = frozenset({"camera_id", "video_id"})


class DatasetError(Exception):
    """Raised when dataset configuration or manifest operations fail."""


@dataclass(frozen=True)
class DatasetConfig:
    """Paths and naming conventions for a registered traffic video dataset."""

    name: str
    display_name: str
    manifest_path: Path
    video_dir: Path
    processed_dir: Path
    legacy_master: bool = False

    def detections_csv(self) -> Path:
        if self.legacy_master:
            return self.processed_dir / "traffic_detections_master.csv"
        return self.processed_dir / f"traffic_detections_{self.name}.csv"

    def congestion_csv(self) -> Path:
        if self.legacy_master:
            return self.processed_dir / "congestion_metrics_master.csv"
        return self.processed_dir / f"congestion_metrics_{self.name}.csv"

    def summary_csv(self) -> Path:
        if self.legacy_master:
            return self.processed_dir / "object_summary_master.csv"
        return self.processed_dir / f"object_summary_{self.name}.csv"

    def near_miss_events_csv(self) -> Path:
        if self.legacy_master:
            return PROJECT_ROOT / "data" / "processed" / "near_miss_events.csv"
        return self.processed_dir / f"near_miss_events_{self.name}.csv"

    def delay_cost_csv(self) -> Path:
        if self.legacy_master:
            return PROJECT_ROOT / "data" / "processed" / "delay_cost_analysis.csv"
        return self.processed_dir / f"delay_cost_analysis_{self.name}.csv"


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "mixkit": DatasetConfig(
        name="mixkit",
        display_name="Mixkit (MVP / Portfolio)",
        manifest_path=PROJECT_ROOT / "data" / "raw" / "videos" / "video_manifest.yaml",
        video_dir=PROJECT_ROOT / "data" / "raw" / "videos",
        processed_dir=PROJECT_ROOT / "data" / "processed",
        legacy_master=True,
    ),
    "aicity": DatasetConfig(
        name="aicity",
        display_name="AI City Challenge",
        manifest_path=PROJECT_ROOT / "data" / "raw" / "aicity_videos" / "aicity_manifest.yaml",
        video_dir=PROJECT_ROOT / "data" / "raw" / "aicity_videos",
        processed_dir=PROJECT_ROOT / "data" / "processed" / "aicity",
    ),
    "bdd100k": DatasetConfig(
        name="bdd100k",
        display_name="BDD100K",
        manifest_path=PROJECT_ROOT / "data" / "raw" / "bdd100k_videos" / "bdd100k_manifest.yaml",
        video_dir=PROJECT_ROOT / "data" / "raw" / "bdd100k_videos",
        processed_dir=PROJECT_ROOT / "data" / "processed" / "bdd100k",
    ),
    "kaggle_traffic": DatasetConfig(
        name="kaggle_traffic",
        display_name="Kaggle Traffic (Notebook)",
        manifest_path=PROJECT_ROOT
        / "data"
        / "raw"
        / "kaggle_traffic_videos"
        / "kaggle_traffic_manifest.yaml",
        video_dir=PROJECT_ROOT / "data" / "raw" / "kaggle_traffic_videos",
        processed_dir=PROJECT_ROOT / "data" / "processed" / "kaggle_traffic",
    ),
}

COMPARISON_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dataset_comparison.csv"


@dataclass(frozen=True)
class ManifestEntry:
    """Single video entry from a dataset manifest."""

    camera_id: str
    video_id: str
    file_name: str
    scene_type: str
    road_type: str
    video_path: Path


@dataclass(frozen=True)
class ManifestLoadResult:
    """Result of loading and validating a manifest."""

    entries: list[ManifestEntry]
    missing_files: list[str]
    manifest_path: Path
    video_dir: Path


def get_dataset_config(name: str) -> DatasetConfig:
    """Return configuration for a known dataset name."""
    key = name.strip().lower()
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise DatasetError(f"Unknown dataset '{name}'. Known datasets: {known}")
    return DATASET_REGISTRY[key]


def resolve_file_name(item: dict) -> str:
    """Support both new (file_name) and legacy Mixkit (filename) manifest keys."""
    if "file_name" in item and item["file_name"]:
        return str(item["file_name"])
    if "filename" in item and item["filename"]:
        return str(item["filename"])
    raise DatasetError(f"Manifest entry missing file_name/filename: {item}")


def validate_video_directory(video_dir: Path) -> None:
    """Ensure the video directory exists and is not empty."""
    resolved = video_dir.expanduser().resolve()
    if not resolved.exists():
        raise DatasetError(f"Video directory not found: {resolved}")
    if not resolved.is_dir():
        raise DatasetError(f"Video path is not a directory: {resolved}")

    video_files = [
        p for p in resolved.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not video_files:
        raise DatasetError(
            f"Video directory is empty (no supported video files): {resolved}\n"
            f"Supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )


def load_manifest(
    manifest_path: Path,
    video_dir: Path,
    *,
    require_videos: bool = True,
) -> ManifestLoadResult:
    """Load and validate a YAML video manifest."""
    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_video_dir = video_dir.expanduser().resolve()

    if not resolved_manifest.exists():
        raise DatasetError(f"Manifest not found: {resolved_manifest}")

    validate_video_directory(resolved_video_dir)

    try:
        with resolved_manifest.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise DatasetError(f"Invalid YAML in {resolved_manifest}: {exc}") from exc
    except OSError as exc:
        raise DatasetError(f"Cannot read manifest: {exc}") from exc

    if not isinstance(raw, dict) or "videos" not in raw:
        raise DatasetError(f"Manifest must contain a top-level 'videos' list: {resolved_manifest}")

    videos = raw["videos"]
    if not isinstance(videos, list):
        raise DatasetError("'videos' must be a list in the manifest.")

    entries: list[ManifestEntry] = []
    missing: list[str] = []

    for index, item in enumerate(videos):
        if not isinstance(item, dict):
            raise DatasetError(f"Invalid manifest entry at index {index}: expected mapping.")

        missing_fields = REQUIRED_MANIFEST_FIELDS - set(item.keys())
        file_name = resolve_file_name(item)
        missing_fields -= {"file_name"}  # filename satisfies legacy

        if "filename" not in item and "file_name" not in item:
            missing_fields.add("file_name")

        if missing_fields:
            raise DatasetError(
                f"Manifest entry {index} missing fields: {sorted(missing_fields)}"
            )

        video_path = resolved_video_dir / file_name
        entry = ManifestEntry(
            camera_id=str(item["camera_id"]),
            video_id=str(item["video_id"]),
            file_name=file_name,
            scene_type=str(item.get("scene_type", item.get("scene", "unknown"))),
            road_type=str(item.get("road_type", "unknown")),
            video_path=video_path,
        )

        if video_path.exists():
            entries.append(entry)
        else:
            missing.append(file_name)

    if require_videos and not entries:
        raise DatasetError(
            f"No manifest videos found on disk under {resolved_video_dir}.\n"
            f"Missing {len(missing)} file(s). Register or download videos first."
        )

    return ManifestLoadResult(
        entries=entries,
        missing_files=missing,
        manifest_path=resolved_manifest,
        video_dir=resolved_video_dir,
    )


def log_manifest_statistics(
    result: ManifestLoadResult,
    dataset_label: str = "dataset",
) -> None:
    """Log summary statistics for a loaded manifest."""
    cameras = {entry.camera_id for entry in result.entries}
    scenes = {entry.scene_type for entry in result.entries}

    logger.info("Dataset: %s", dataset_label)
    logger.info("  Manifest     : %s", result.manifest_path.name)
    logger.info("  Video dir      : %s", result.video_dir)
    logger.info("  Videos ready   : %d", len(result.entries))
    logger.info("  Unique cameras : %d", len(cameras))
    logger.info("  Scene types    : %d", len(scenes))
    if result.missing_files:
        logger.warning("  Missing files  : %d", len(result.missing_files))


def slugify_video_stem(stem: str) -> str:
    """Create a stable slug from a video filename stem."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return slug or "video"


def build_manifest_entry_from_path(
    video_path: Path,
    dataset_name: str,
    index: int,
) -> dict[str, str]:
    """Auto-generate a manifest entry for a video file."""
    stem = video_path.stem
    slug = slugify_video_stem(stem)
    camera_id = f"CAM_{dataset_name.upper()}_{index:03d}"
    video_id = f"{dataset_name}_{slug}"
    return {
        "camera_id": camera_id,
        "video_id": video_id,
        "file_name": video_path.name,
        "scene_type": "unknown",
        "road_type": "unknown",
    }


def scan_video_directory(video_dir: Path) -> list[Path]:
    """Return sorted video file paths from a directory."""
    validate_video_directory(video_dir)
    files = [
        p
        for p in sorted(video_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return files


def write_manifest(
    manifest_path: Path,
    entries: list[dict[str, str]],
    *,
    dataset_name: str | None = None,
) -> Path:
    """Write manifest entries to YAML."""
    resolved = manifest_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    header = f"# Auto-generated manifest"
    if dataset_name:
        header = f"# {dataset_name} traffic video manifest\n# Auto-generated by register_dataset.py"

    payload = {"videos": entries}
    with resolved.open("w", encoding="utf-8") as handle:
        handle.write(f"{header}\n\n")
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)

    logger.info("Wrote manifest with %d video(s) -> %s", len(entries), resolved)
    return resolved


def ensure_processed_dir(config: DatasetConfig) -> Path:
    """Create processed output directory for a dataset."""
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    return config.processed_dir
