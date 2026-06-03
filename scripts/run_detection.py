#!/usr/bin/env python3
"""
Run YOLOv8n object detection on the MVP traffic sample video.

Reads data/raw/sample_traffic.mp4, detects traffic-related COCO classes on every
10th frame, and writes bounding-box results to data/processed/traffic_detections.csv.

Usage:
    python scripts/run_detection.py
    python scripts/run_detection.py --video data/raw/sample_traffic.mp4 --verbose
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "sample_traffic.mp4"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "traffic_detections.csv"
DEFAULT_MODEL = "yolov8n.pt"

# COCO class names targeted for urban traffic safety.
TARGET_CLASSES: frozenset[str] = frozenset(
    {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
)

# COCO class IDs matching TARGET_CLASSES (used to filter YOLO output).
TARGET_CLASS_IDS: list[int] = [0, 1, 2, 3, 5, 7]

CSV_COLUMNS = [
    "timestamp",
    "frame_id",
    "object_type",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]

logger = logging.getLogger("run_detection")


@dataclass(frozen=True)
class DetectionRecord:
    """Single detection row."""

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
            "timestamp": round(self.timestamp, 3),
            "frame_id": self.frame_id,
            "object_type": self.object_type,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


class DetectionError(Exception):
    """Raised when detection cannot proceed."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model(model_name: str) -> YOLO:
    """Load YOLOv8 weights (downloads yolov8n.pt automatically on first run)."""
    logger.info("Loading model: %s", model_name)
    try:
        model = YOLO(model_name)
    except Exception as exc:
        raise DetectionError(f"Failed to load YOLO model '{model_name}': {exc}") from exc
    logger.info("Model loaded successfully.")
    return model


def validate_video_path(video_path: Path) -> tuple[float, int]:
    """
    Confirm the video exists and return (fps, total_frame_count).

    Raises:
        DetectionError: If the file is missing or unreadable.
    """
    resolved = video_path.expanduser().resolve()
    if not resolved.exists():
        raise DetectionError(
            f"Video not found: {resolved}\n"
            "Run: python scripts/download_sample_video.py"
        )

    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        raise DetectionError(f"OpenCV cannot open video: {resolved}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise DetectionError(
            f"Invalid video metadata (fps={fps}, frames={frame_count}): {resolved}"
        )

    logger.info(
        "Video OK — %d frames @ %.2f fps (%.1f s)",
        frame_count,
        fps,
        frame_count / fps,
    )
    return fps, frame_count


def extract_detections_from_result(
    result,
    frame_id: int,
    fps: float,
) -> list[DetectionRecord]:
    """Parse a single YOLO result object into DetectionRecord rows."""
    records: list[DetectionRecord] = []
    timestamp = frame_id / fps
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return records

    names = result.names
    for box in boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        if label not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf.item())
        records.append(
            DetectionRecord(
                timestamp=timestamp,
                frame_id=frame_id,
                object_type=label,
                confidence=confidence,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return records


def run_detection(
    video_path: Path,
    model: YOLO,
    frame_stride: int = 10,
    conf_threshold: float = 0.25,
) -> list[DetectionRecord]:
    """
    Run YOLOv8 inference on every Nth frame of the video.

    Args:
        video_path: Path to input MP4.
        model: Loaded YOLO model.
        frame_stride: Process every Nth frame (default 10).
        conf_threshold: Minimum confidence score.

    Returns:
        List of all detection records across sampled frames.
    """
    fps, total_frames = validate_video_path(video_path)
    sampled_frames = max(1, total_frames // frame_stride)
    logger.info(
        "Running detection — stride=%d (~%d frames of %d total)",
        frame_stride,
        sampled_frames,
        total_frames,
    )

    all_records: list[DetectionRecord] = []

    try:
        results = model.predict(
            source=str(video_path),
            vid_stride=frame_stride,
            classes=TARGET_CLASS_IDS,
            conf=conf_threshold,
            stream=True,
            verbose=False,
        )

        for idx, result in enumerate(results):
            # Ultralytics does not expose frame_id on all versions; derive from stride.
            frame_id = idx * frame_stride
            frame_records = extract_detections_from_result(result, frame_id, fps)
            all_records.extend(frame_records)
            logger.debug(
                "Frame %d (t=%.2fs): %d detection(s)",
                frame_id,
                frame_id / fps,
                len(frame_records),
            )

    except cv2.error as exc:
        raise DetectionError(f"OpenCV error during inference: {exc}") from exc
    except Exception as exc:
        raise DetectionError(f"YOLO inference failed: {exc}") from exc

    logger.info("Detection complete — %d total bounding box(es).", len(all_records))
    return all_records


def save_detections_csv(records: list[DetectionRecord], output_path: Path) -> Path:
    """Write detection records to CSV."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_row())

    logger.info("Saved %d row(s) -> %s", len(records), output_path)
    return output_path


def summarize_detections(records: list[DetectionRecord]) -> None:
    """Print a brief detection summary to stdout."""
    if not records:
        print("\nNo detections found. Try lowering --conf or check video content.\n")
        return

    counts: dict[str, int] = {}
    for r in records:
        counts[r.object_type] = counts.get(r.object_type, 0) + 1

    print("\n=== Detection Summary ===")
    for obj_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {obj_type:15s}: {count}")
    print(f"  {'TOTAL':15s}: {len(records)}")
    print("=========================\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n detection on the MVP traffic sample video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help=f"Input video (default: {DEFAULT_VIDEO.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"YOLO weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Process every Nth frame (default: 10)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        model = load_model(args.model)
        records = run_detection(
            video_path=args.video,
            model=model,
            frame_stride=args.stride,
            conf_threshold=args.conf,
        )
        save_detections_csv(records, args.output)
        summarize_detections(records)
    except DetectionError as exc:
        logger.error("Detection pipeline failed: %s", exc)
        return 1
    except OSError as exc:
        logger.exception("File I/O error: %s", exc)
        return 1

    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
