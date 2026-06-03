"""Safety metrics, aggregations, and exploratory analytics."""

from traffic_safety.analytics.metrics import (
    MetricsError,
    classify_congestion,
    compute_congestion_metrics,
    compute_object_summary,
    compute_object_summary_auto,
    compute_object_summary_by_camera,
    compute_per_frame_averages,
    compute_summary_statistics,
    compute_traffic_density,
    is_batch_dataset,
    load_detections,
    run_metrics_pipeline,
)

__all__ = [
    "MetricsError",
    "classify_congestion",
    "compute_congestion_metrics",
    "compute_object_summary",
    "compute_object_summary_auto",
    "compute_object_summary_by_camera",
    "compute_per_frame_averages",
    "compute_summary_statistics",
    "compute_traffic_density",
    "is_batch_dataset",
    "load_detections",
    "run_metrics_pipeline",
]
