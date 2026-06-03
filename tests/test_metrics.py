"""Tests for analytics metrics."""

import pandas as pd
import pytest

from traffic_safety.analytics.metrics import (
    MetricsError,
    classify_congestion,
    compute_congestion_metrics,
    compute_object_summary_by_camera,
    compute_summary_statistics,
    is_batch_dataset,
    load_detections,
)


def _mvp_detections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0.0, 0.0, 5.0, 65.0, 65.0, 70.0],
            "frame_id": [0, 0, 10, 100, 110, 110],
            "object_type": ["car", "car", "person", "car", "truck", "truck"],
            "confidence": [0.9, 0.8, 0.8, 0.85, 0.75, 0.7],
            "x1": [0, 0, 0, 0, 0, 0],
            "y1": [0, 0, 0, 0, 0, 0],
            "x2": [1, 1, 1, 1, 1, 1],
            "y2": [1, 1, 1, 1, 1, 1],
        }
    )


def _batch_detections() -> pd.DataFrame:
    df = _mvp_detections()
    df["camera_id"] = ["CAM_A", "CAM_A", "CAM_A", "CAM_B", "CAM_B", "CAM_B"]
    df["video_id"] = ["vid_1", "vid_1", "vid_1", "vid_2", "vid_2", "vid_2"]
    return df


def test_classify_congestion_levels():
    assert classify_congestion(4.9) == "Low"
    assert classify_congestion(5) == "Medium"
    assert classify_congestion(20) == "Severe"


def test_is_batch_dataset():
    assert not is_batch_dataset(_mvp_detections())
    assert is_batch_dataset(_batch_detections())


def test_compute_congestion_metrics_mvp():
    result = compute_congestion_metrics(_mvp_detections())
    assert "camera_id" not in result.columns
    assert len(result) == 2


def test_compute_congestion_metrics_batch():
    result = compute_congestion_metrics(_batch_detections())
    assert list(result.columns[:3]) == ["camera_id", "video_id", "minute"]
    assert set(result["camera_id"]) == {"CAM_A", "CAM_B"}


def test_compute_object_summary_by_camera():
    summary = compute_object_summary_by_camera(_batch_detections())
    assert summary["count"].sum() == len(_batch_detections())


def test_compute_summary_statistics_batch():
    congestion = compute_congestion_metrics(_batch_detections())
    stats = compute_summary_statistics(_batch_detections(), congestion)
    assert stats.camera_count == 2
    assert stats.peak_camera is not None


def test_load_detections_missing_file(tmp_path):
    with pytest.raises(MetricsError, match="not found"):
        load_detections(tmp_path / "missing.csv")
