"""Tests for multi-dataset manifest and comparison utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traffic_safety.data.dataset_comparison import build_comparison_dataframe, count_ready_datasets
from traffic_safety.data.datasets import (
    DatasetError,
    build_manifest_entry_from_path,
    get_dataset_config,
    load_manifest,
    resolve_file_name,
    write_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_file_name_legacy_and_new() -> None:
    assert resolve_file_name({"filename": "a.mp4"}) == "a.mp4"
    assert resolve_file_name({"file_name": "b.mp4"}) == "b.mp4"


def test_load_mixkit_manifest() -> None:
    config = get_dataset_config("mixkit")
    result = load_manifest(config.manifest_path, config.video_dir)
    assert len(result.entries) >= 1
    assert result.missing_files == []


def test_build_comparison_includes_mixkit() -> None:
    df = build_comparison_dataframe(["mixkit"])
    assert len(df) == 1
    assert df.iloc[0]["dataset"] == "mixkit"
    assert df.iloc[0]["total_detections"] > 0


def test_count_ready_datasets() -> None:
    df = build_comparison_dataframe()
    assert count_ready_datasets(df) >= 1


def test_invalid_manifest_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: [", encoding="utf-8")
    with pytest.raises(DatasetError, match="Invalid YAML"):
        load_manifest(bad, tmp_path)


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    video = tmp_path / "sample_clip.mp4"
    video.write_bytes(b"fake")
    entry = build_manifest_entry_from_path(video, "testset", 1)
    manifest = tmp_path / "test_manifest.yaml"
    write_manifest(manifest, [entry], dataset_name="testset")
    with manifest.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert data["videos"][0]["file_name"] == "sample_clip.mp4"
    assert data["videos"][0]["scene_type"] == "unknown"
