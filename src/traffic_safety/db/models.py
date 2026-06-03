"""SQLAlchemy ORM models for the Urban Traffic Safety Engine."""

from __future__ import annotations

from sqlalchemy import Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from traffic_safety.db.connection import Base


class TrafficDetection(Base):
    """YOLO detection records loaded from traffic_detections_master.csv."""

    __tablename__ = "traffic_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    video_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    object_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)


class CongestionMetric(Base):
    """Per-camera/video/minute congestion KPIs."""

    __tablename__ = "congestion_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    video_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    sampled_frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_count: Mapped[int] = mapped_column(Integer, nullable=False)
    pedestrian_count: Mapped[int] = mapped_column(Integer, nullable=False)
    avg_vehicles_per_frame: Mapped[float] = mapped_column(Float, nullable=False)
    avg_pedestrians_per_frame: Mapped[float] = mapped_column(Float, nullable=False)
    traffic_density: Mapped[float] = mapped_column(Float, nullable=False)
    congestion_level: Mapped[str] = mapped_column(String(16), index=True, nullable=False)


class ObjectSummary(Base):
    """Detection counts by camera and object type."""

    __tablename__ = "object_summary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    object_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, nullable=False)
