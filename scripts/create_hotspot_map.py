#!/usr/bin/env python3
"""
Build an interactive Folium traffic hotspot map for the Urban Traffic Safety Engine.

Joins simulated camera geospatial metadata with batch congestion metrics and
renders camera markers plus a traffic-density heatmap.

Usage:
    python scripts/generate_camera_metadata.py
    python scripts/create_hotspot_map.py
    python scripts/create_hotspot_map.py --verbose

Output:
    outputs/traffic_hotspot_map.html
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("create_hotspot_map")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "camera_metadata.csv"
DEFAULT_CONGESTION = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "traffic_hotspot_map.html"

METADATA_COLUMNS = [
    "camera_id",
    "camera_name",
    "scene_type",
    "latitude",
    "longitude",
    "road_type",
]

CONGESTION_COLUMNS = [
    "camera_id",
    "traffic_density",
    "congestion_level",
    "vehicle_count",
    "pedestrian_count",
]

CONGESTION_LEVEL_COLORS: dict[str, str] = {
    "Low": "green",
    "Medium": "orange",
    "High": "darkorange",
    "Severe": "red",
}

BLOOMINGTON_CENTER = (39.1653, -86.5264)


class HotspotMapError(Exception):
    """Raised when hotspot map generation fails."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv(path: Path, label: str, required_columns: list[str]) -> pd.DataFrame:
    """Load and validate a CSV file."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise HotspotMapError(
            f"Missing {label}: {resolved}\n"
            "Run: python scripts/generate_camera_metadata.py && "
            "python scripts/compute_metrics.py --master"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise HotspotMapError(f"{label} is empty: {resolved}") from exc
    except OSError as exc:
        raise HotspotMapError(f"Cannot read {label}: {exc}") from exc

    if df.empty:
        raise HotspotMapError(f"{label} contains no rows: {resolved}")

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise HotspotMapError(f"{label} missing columns: {sorted(missing)}")

    logger.info("Loaded %d row(s) from %s", len(df), resolved.name)
    return df


def aggregate_congestion_by_camera(congestion_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse minute-level congestion rows to one record per camera."""
    aggregated = (
        congestion_df.groupby("camera_id", as_index=False)
        .agg(
            traffic_density=("traffic_density", "mean"),
            congestion_level=("congestion_level", lambda s: s.mode().iloc[0]),
            vehicle_count=("vehicle_count", "sum"),
            pedestrian_count=("pedestrian_count", "sum"),
        )
        .round({"traffic_density": 4})
    )
    logger.info("Aggregated congestion metrics to %d camera(s)", len(aggregated))
    return aggregated


def join_camera_congestion(
    metadata_df: pd.DataFrame,
    congestion_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join metadata with per-camera congestion KPIs."""
    camera_congestion = aggregate_congestion_by_camera(congestion_df)
    joined = metadata_df.merge(camera_congestion, on="camera_id", how="left")

    missing_metrics = joined["traffic_density"].isna().sum()
    if missing_metrics:
        logger.warning(
            "%d camera(s) in metadata have no congestion metrics — "
            "they will appear with zero density on the map.",
            missing_metrics,
        )
        joined["traffic_density"] = joined["traffic_density"].fillna(0.0)
        joined["congestion_level"] = joined["congestion_level"].fillna("Unknown")
        joined["vehicle_count"] = joined["vehicle_count"].fillna(0).astype(int)
        joined["pedestrian_count"] = joined["pedestrian_count"].fillna(0).astype(int)

    logger.info("Joined dataset contains %d camera location(s)", len(joined))
    return joined


def build_popup_html(row: pd.Series) -> str:
    """Render HTML popup content for a camera marker."""
    return (
        f"<b>{row['camera_name']}</b><br>"
        f"<b>Camera ID:</b> {row['camera_id']}<br>"
        f"<b>Scene:</b> {row['scene_type']}<br>"
        f"<b>Road type:</b> {row['road_type']}<br>"
        f"<b>Traffic density:</b> {row['traffic_density']:.4f}<br>"
        f"<b>Congestion level:</b> {row['congestion_level']}<br>"
        f"<b>Vehicle count:</b> {int(row['vehicle_count'])}<br>"
        f"<b>Pedestrian count:</b> {int(row['pedestrian_count'])}"
    )


def marker_color(congestion_level: str) -> str:
    """Map congestion level to a Folium marker color."""
    return CONGESTION_LEVEL_COLORS.get(str(congestion_level), "blue")


def create_folium_map(joined_df: pd.DataFrame):
    """Build a Folium map with camera markers and a traffic-density heatmap."""
    try:
        import folium
        from folium.plugins import HeatMap
    except ImportError as exc:
        raise HotspotMapError(
            "folium is required. Install with: pip install folium"
        ) from exc

    map_center = [
        float(joined_df["latitude"].mean()),
        float(joined_df["longitude"].mean()),
    ]

    traffic_map = folium.Map(
        location=map_center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(traffic_map)

    marker_cluster = folium.FeatureGroup(name="Traffic Cameras", show=True)
    for _, row in joined_df.iterrows():
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=8,
            popup=folium.Popup(build_popup_html(row), max_width=320),
            tooltip=f"{row['camera_name']} — {row['congestion_level']}",
            color=marker_color(str(row["congestion_level"])),
            fill=True,
            fill_color=marker_color(str(row["congestion_level"])),
            fill_opacity=0.85,
            weight=2,
        ).add_to(marker_cluster)

    marker_cluster.add_to(traffic_map)

    heatmap_data = [
        [float(row["latitude"]), float(row["longitude"]), float(row["traffic_density"])]
        for _, row in joined_df.iterrows()
        if float(row["traffic_density"]) > 0
    ]

    if heatmap_data:
        HeatMap(
            heatmap_data,
            name="Traffic Density Heatmap",
            min_opacity=0.35,
            radius=22,
            blur=18,
            max_zoom=15,
            gradient={
                0.2: "blue",
                0.4: "lime",
                0.6: "yellow",
                0.8: "orange",
                1.0: "red",
            },
        ).add_to(traffic_map)
        logger.info("Added heatmap layer with %d weighted point(s)", len(heatmap_data))
    else:
        logger.warning("No non-zero density points — heatmap layer skipped.")

    folium.LayerControl(collapsed=False).add_to(traffic_map)

    title_html = """
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background-color: white;
        padding: 10px 14px;
        border: 2px solid #ccc;
        border-radius: 6px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    ">
        <b>Urban Traffic Safety Engine</b><br>
        Bloomington, IN — Traffic Hotspot Map
    </div>
    """
    traffic_map.get_root().html.add_child(folium.Element(title_html))

    return traffic_map


def save_map(traffic_map, output_path: Path) -> None:
    """Persist the Folium map to an HTML file."""
    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        traffic_map.save(str(resolved))
    except OSError as exc:
        raise HotspotMapError(f"Failed to write map HTML: {exc}") from exc
    logger.info("Saved hotspot map to %s", resolved)


def run_hotspot_map_pipeline(
    metadata_path: Path = DEFAULT_METADATA,
    congestion_path: Path = DEFAULT_CONGESTION,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    """Load data, join, build map, and save HTML output."""
    metadata_df = read_csv(metadata_path, "camera metadata", METADATA_COLUMNS)
    congestion_df = read_csv(
        congestion_path,
        "congestion metrics",
        ["camera_id", "traffic_density", "congestion_level", "vehicle_count", "pedestrian_count"],
    )

    joined_df = join_camera_congestion(metadata_df, congestion_df)
    traffic_map = create_folium_map(joined_df)
    save_map(traffic_map, output_path)
    return output_path.resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Folium traffic hotspot map."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path to camera_metadata.csv",
    )
    parser.add_argument(
        "--congestion",
        type=Path,
        default=DEFAULT_CONGESTION,
        help="Path to congestion_metrics_master.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output HTML map path",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        output = run_hotspot_map_pipeline(
            metadata_path=args.metadata,
            congestion_path=args.congestion,
            output_path=args.output,
        )
    except HotspotMapError as exc:
        logger.error("Hotspot map generation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 1

    print(f"\nTraffic hotspot map saved to: {output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
