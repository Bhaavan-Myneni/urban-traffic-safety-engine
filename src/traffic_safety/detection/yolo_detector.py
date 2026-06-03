"""Run YOLOv8 inference on images or video frames."""

from pathlib import Path

from ultralytics import YOLO

from traffic_safety.config import settings


def load_model(model_path: str | None = None) -> YOLO:
    return YOLO(model_path or settings.yolo_model_path)


def detect(source: str | Path, model_path: str | None = None):
    model = load_model(model_path)
    return model.predict(
        source=str(source),
        conf=settings.yolo_confidence_threshold,
        iou=settings.yolo_iou_threshold,
        verbose=False,
    )
