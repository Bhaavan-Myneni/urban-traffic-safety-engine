"""Extract frames and metadata from raw video sources."""

from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, every_n: int = 30) -> list[Path]:
    """Sample every Nth frame from a video file and write to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    saved: list[Path] = []
    frame_idx = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            out_path = output_dir / f"{video_path.stem}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append(out_path)
        frame_idx += 1

    cap.release()
    return saved
