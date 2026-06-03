#!/usr/bin/env python3
"""
Batch YOLOv8 detection across multiple traffic camera videos.

Processes all MP4 files listed in video_manifest.yaml using multiprocessing,
aggregates detections into a single master CSV.

Usage:
    python scripts/download_batch_videos.py
    python scripts/batch_detection.py
    python scripts/batch_detection.py --workers 4 --stride 10 --verbose

Output:
    data/processed/traffic_detections_master.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traffic_safety.data.datasets import (  # noqa: E402
    DatasetError,
    load_manifest,
    log_manifest_statistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "raw" / "videos" / "video_manifest.yaml"
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "traffic_detections_master.csv"
DEFAULT_MODEL = "yolov8n.pt"

TARGET_CLASSES: frozenset[str] = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
)
TARGET_CLASS_IDS: list[int] = [0, 1, 2, 3, 5, 7]

MASTER_CSV_COLUMNS: list[str] = [
    "camera_id",
    "video_id",
    "timestamp",
    "frame_id",
    "object_type",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

logger = logging.getLogger("batch_detection")


@dataclass(frozen=True)
class VideoJob:
    """Work unit passed to each worker process."""

    video_path: str
    camera_id: str
    video_id: str
    model_name: str
    frame_stride: int
    conf_threshold: float


@dataclass
class MasterDetectionRecord:
    camera_id: str
    video_id: str
    timestamp: float
    frame_id: int
    object_type: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "camera_id": self.camera_id,
            "video_id": self.video_id,
            "timestamp": round(self.timestamp, 3),
            "frame_id": self.frame_id,
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


@dataclass(frozen=True)
class VideoJobResult:
    video_id: str
    camera_id: str
    detection_count: int
    elapsed_seconds: float
    error: str | None = None


class BatchDetectionError(Exception):
    """Raised when batch detection cannot proceed."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_manifest_jobs(
    manifest_path: Path,
    video_dir: Path,
    model_name: str,
    frame_stride: int,
    conf_threshold: float,
) -> list[VideoJob]:
    """Build worker jobs from manifest entries that exist on disk."""
    try:
        result = load_manifest(manifest_path, video_dir, require_videos=True)
    except DatasetError as exc:
        raise BatchDetectionError(str(exc)) from exc

    log_manifest_statistics(result, dataset_label=manifest_path.stem)

    if result.missing_files:
        logger.warning(
            "Skipping %d video(s) listed in manifest but not found on disk.",
            len(result.missing_files),
        )
        for name in result.missing_files[:10]:
            logger.warning("  Missing: %s", name)
        if len(result.missing_files) > 10:
            logger.warning("  ... and %d more", len(result.missing_files) - 10)

    jobs = [
        VideoJob(
            video_path=str(entry.video_path.resolve()),
            camera_id=entry.camera_id,
            video_id=entry.video_id,
            model_name=model_name,
            frame_stride=frame_stride,
            conf_threshold=conf_threshold,
        )
        for entry in result.entries
    ]

    if not jobs:
        raise BatchDetectionError(
            "No videos available for batch detection.\n"
            "Run: python scripts/download_batch_videos.py"
        )

    return jobs


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise BatchDetectionError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise BatchDetectionError(f"Invalid FPS for video: {video_path}")
    return fps


def _extract_records_from_result(
    result,
    frame_id: int,
    fps: float,
    camera_id: str,
    video_id: str,
) -> list[MasterDetectionRecord]:
    records: list[MasterDetectionRecord] = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return records

    timestamp = frame_id / fps
    names = result.names
    for box in boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        if label not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        records.append(
            MasterDetectionRecord(
                camera_id=camera_id,
                video_id=video_id,
                timestamp=timestamp,
                frame_id=frame_id,
                object_type=label,
                confidence=float(box.conf.item()),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return records


def process_video_job(job: VideoJob) -> tuple[list[dict], VideoJobResult]:
    """
    Worker entry point — loads YOLO once per process and processes one video.

    Returns:
        Tuple of (detection rows as dicts, job result metadata).
    """
    start = time.perf_counter()
    rows: list[dict] = []

    try:
        model = YOLO(job.model_name)
        fps = _get_video_fps(job.video_path)

        results = model.predict(
            source=job.video_path,
            vid_stride=job.frame_stride,
            classes=TARGET_CLASS_IDS,
            conf=job.conf_threshold,
            stream=True,
            verbose=False,
        )

        for idx, result in enumerate(results):
            frame_id = idx * job.frame_stride
            records = _extract_records_from_result(
                result, frame_id, fps, job.camera_id, job.video_id
            )
            rows.extend(r.as_row() for r in records)

        elapsed = time.perf_counter() - start
        return rows, VideoJobResult(
            video_id=job.video_id,
            camera_id=job.camera_id,
            detection_count=len(rows),
            elapsed_seconds=round(elapsed, 2),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        return [], VideoJobResult(
            video_id=job.video_id,
            camera_id=job.camera_id,
            detection_count=0,
            elapsed_seconds=round(elapsed, 2),
            error=str(exc),
        )


def run_batch_detection(
    jobs: list[VideoJob],
    max_workers: int,
) -> tuple[list[dict], list[VideoJobResult]]:
    """Execute video jobs in parallel using a process pool."""
    all_rows: list[dict] = []
    results: list[VideoJobResult] = []

    logger.info(
        "Starting batch detection — %d video(s), %d worker(s)",
        len(jobs),
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_video_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            rows, result = future.result()
            all_rows.extend(rows)
            results.append(result)

            if result.error:
                logger.error(
                    "Failed %s (%s): %s",
                    result.video_id,
                    result.camera_id,
                    result.error,
                )
            else:
                logger.info(
                    "Completed %s — %d detections in %.1fs",
                    result.video_id,
                    result.detection_count,
                    result.elapsed_seconds,
                )

    return all_rows, results


def save_master_csv(rows: list[dict], output_path: Path) -> Path:
    """Write aggregated detections to the master CSV."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASTER_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved master dataset — %d rows -> %s", len(rows), output_path)
    return output_path


def print_batch_summary(rows: list[dict], results: list[VideoJobResult]) -> None:
    """Print batch processing summary."""
    success = [r for r in results if not r.error]
    failed = [r for r in results if r.error]
    total_detections = len(rows)

    print("\n=== Batch Detection Summary ===")
    print(f"  Videos processed     : {len(success)}/{len(results)}")
    print(f"  Failed videos        : {len(failed)}")
    print(f"  Total detections     : {total_detections:,}")
    if success:
        print(f"  Avg detections/video : {total_detections // len(success):,}")
    print("================================\n")


def resolve_worker_count(requested: int | None, job_count: int) -> int:
    cpu = os.cpu_count() or 2
    default = max(1, min(cpu - 1, 4, job_count))
    if requested is None:
        return default
    return max(1, min(requested, job_count))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch YOLOv8 detection across multiple traffic videos."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--stride", type=int, default=2, help="Process every Nth frame (default: 2)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: min(cpu-1, 4))",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        jobs = load_manifest_jobs(
            manifest_path=args.manifest,
            video_dir=args.video_dir,
            model_name=args.model,
            frame_stride=args.stride,
            conf_threshold=args.conf,
        )
        workers = resolve_worker_count(args.workers, len(jobs))
        rows, results = run_batch_detection(jobs, max_workers=workers)
        save_master_csv(rows, args.output)
        print_batch_summary(rows, results)

        if len(rows) < 20_000:
            logger.warning(
                "Dataset has %d detections (target: 20,000+). "
                "Try --stride 5 or download more videos.",
                len(rows),
            )

    except BatchDetectionError as exc:
        logger.error("Batch detection failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Batch detection interrupted.")
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    logger.info("Batch pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
