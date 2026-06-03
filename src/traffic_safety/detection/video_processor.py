"""OpenCV helpers for frame preprocessing and annotation."""

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))


def draw_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int, str]]) -> np.ndarray:
    annotated = frame.copy()
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
