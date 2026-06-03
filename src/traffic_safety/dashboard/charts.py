"""Plotly chart builders for the traffic safety dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CONGESTION_COLORS: dict[str, str] = {
    "Low": "#2ecc71",
    "Medium": "#f1c40f",
    "High": "#e67e22",
    "Severe": "#e74c3c",
}

CONGESTION_ORDER = ["Low", "Medium", "High", "Severe"]
CHART_TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#1f4e79"
SECONDARY_COLOR = "#2980b9"
ACCENT_COLOR = "#16a085"


def object_type_bar_chart(summary: pd.DataFrame, title: str = "Object Type Distribution") -> go.Figure:
    """Bar chart of detection counts by object type."""
    if "camera_id" in summary.columns:
        df = (
            summary.groupby("object_type", as_index=False)["count"]
            .sum()
            .sort_values("count", ascending=True)
        )
    else:
        df = summary.sort_values("count", ascending=True)

    fig = px.bar(
        df,
        x="count",
        y="object_type",
        orientation="h",
        text="count",
        color="object_type",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"count": "Detections", "object_type": "Object Type"},
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, template=CHART_TEMPLATE, height=380)
    return fig


def traffic_density_line_chart(
    congestion: pd.DataFrame,
    title: str = "Traffic Density Over Time",
) -> go.Figure:
    """Line chart of traffic density over minute buckets."""
    if "camera_id" in congestion.columns:
        # Aggregate across videos per camera per minute for overview chart
        plot_df = (
            congestion.groupby(["camera_id", "minute"], as_index=False)["traffic_density"]
            .mean()
            .sort_values(["camera_id", "minute"])
        )
        fig = px.line(
            plot_df,
            x="minute",
            y="traffic_density",
            color="camera_id",
            markers=True,
            labels={"minute": "Minute", "traffic_density": "Traffic Density", "camera_id": "Camera"},
            title=title,
        )
    else:
        fig = px.line(
            congestion,
            x="minute",
            y="traffic_density",
            markers=True,
            labels={"minute": "Minute", "traffic_density": "Traffic Density"},
            title=title,
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        fig.update_traces(line=dict(width=3))

    fig.update_layout(template=CHART_TEMPLATE, height=380, hovermode="x unified")
    return fig


def traffic_density_by_camera_chart(
    congestion: pd.DataFrame,
    selected_camera: str,
) -> go.Figure:
    """Traffic density over time for a single selected camera."""
    cam_data = congestion[congestion["camera_id"] == selected_camera].copy()
    if "video_id" in cam_data.columns:
        fig = px.line(
            cam_data,
            x="minute",
            y="traffic_density",
            color="video_id",
            markers=True,
            labels={"minute": "Minute", "traffic_density": "Traffic Density", "video_id": "Video"},
            title=f"Traffic Density Over Time — {selected_camera}",
        )
    else:
        fig = px.line(
            cam_data,
            x="minute",
            y="traffic_density",
            markers=True,
            title=f"Traffic Density Over Time — {selected_camera}",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
    fig.update_layout(template=CHART_TEMPLATE, height=380, hovermode="x unified")
    return fig


def top_cameras_density_chart(congestion: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of top N cameras by average traffic density."""
    camera_avg = (
        congestion.groupby("camera_id", as_index=False)["traffic_density"]
        .mean()
        .sort_values("traffic_density", ascending=False)
        .head(top_n)
        .sort_values("traffic_density", ascending=True)
    )
    fig = px.bar(
        camera_avg,
        x="traffic_density",
        y="camera_id",
        orientation="h",
        text="traffic_density",
        color="traffic_density",
        color_continuous_scale="Blues",
        labels={"traffic_density": "Avg Traffic Density", "camera_id": "Camera"},
        title=f"Top {top_n} Cameras by Traffic Density",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(showlegend=False, template=CHART_TEMPLATE, height=420, coloraxis_showscale=False)
    return fig


def vehicle_distribution_by_camera_chart(detections: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of vehicle types per camera."""
    vehicle_types = {"car", "truck", "bus", "motorcycle", "bicycle"}
    df = detections[detections["object_type"].isin(vehicle_types)].copy()
    if df.empty or "camera_id" not in df.columns:
        return go.Figure().update_layout(title="Vehicle Distribution by Camera (no data)")

    counts = (
        df.groupby(["camera_id", "object_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig = px.bar(
        counts,
        x="camera_id",
        y="count",
        color="object_type",
        barmode="stack",
        labels={"camera_id": "Camera", "count": "Detections", "object_type": "Vehicle Type"},
        title="Vehicle Distribution by Camera",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420, xaxis_tickangle=-45)
    return fig


def congestion_level_by_camera_chart(congestion: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of congestion level counts per camera."""
    if "camera_id" not in congestion.columns:
        return go.Figure().update_layout(title="Congestion Level by Camera (no data)")

    counts = (
        congestion.groupby(["camera_id", "congestion_level"], as_index=False)
        .size()
        .rename(columns={"size": "minute_buckets"})
    )
    counts["congestion_level"] = pd.Categorical(
        counts["congestion_level"], categories=CONGESTION_ORDER, ordered=True
    )
    colors = {level: CONGESTION_COLORS[level] for level in CONGESTION_ORDER}
    fig = px.bar(
        counts,
        x="camera_id",
        y="minute_buckets",
        color="congestion_level",
        barmode="stack",
        category_orders={"congestion_level": CONGESTION_ORDER},
        color_discrete_map=colors,
        labels={
            "camera_id": "Camera",
            "minute_buckets": "Minute Buckets",
            "congestion_level": "Congestion Level",
        },
        title="Congestion Level by Camera",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420, xaxis_tickangle=-45)
    return fig


def vehicles_pedestrians_grouped_bar(congestion: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing vehicle and pedestrian counts per minute."""
    id_cols = [c for c in ("camera_id", "video_id", "minute") if c in congestion.columns]
    if len(id_cols) > 1 and "camera_id" in id_cols:
        plot_df = (
            congestion.groupby("minute", as_index=False)[["vehicle_count", "pedestrian_count"]]
            .sum()
        )
    else:
        plot_df = congestion

    melted = plot_df.melt(
        id_vars=["minute"],
        value_vars=["vehicle_count", "pedestrian_count"],
        var_name="category",
        value_name="count",
    )
    melted["category"] = melted["category"].map(
        {"vehicle_count": "Vehicles", "pedestrian_count": "Pedestrians"}
    )
    fig = px.bar(
        melted,
        x="minute",
        y="count",
        color="category",
        barmode="group",
        labels={"minute": "Minute", "count": "Detections", "category": "Category"},
        title="Vehicles vs Pedestrians per Minute",
        color_discrete_map={"Vehicles": PRIMARY_COLOR, "Pedestrians": SECONDARY_COLOR},
    )
    fig.update_layout(template=CHART_TEMPLATE, height=380)
    return fig


def congestion_donut_chart(congestion: pd.DataFrame) -> go.Figure:
    """Donut chart of congestion level distribution."""
    counts = (
        congestion["congestion_level"]
        .value_counts()
        .rename_axis("congestion_level")
        .reset_index(name="minutes")
    )
    colors = [CONGESTION_COLORS.get(level, "#95a5a6") for level in counts["congestion_level"]]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["congestion_level"],
                values=counts["minutes"],
                hole=0.45,
                marker=dict(colors=colors),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(title="Congestion Level Distribution", template=CHART_TEMPLATE, height=380)
    return fig


def confidence_histogram(detections: pd.DataFrame) -> go.Figure:
    """Histogram of YOLO confidence scores."""
    fig = px.histogram(
        detections,
        x="confidence",
        nbins=30,
        labels={"confidence": "Confidence Score", "count": "Frequency"},
        title="Confidence Score Distribution",
        color_discrete_sequence=[SECONDARY_COLOR],
    )
    fig.update_layout(template=CHART_TEMPLATE, height=380, bargap=0.05)
    return fig


def near_miss_events_by_camera_chart(events: pd.DataFrame) -> go.Figure:
    """Bar chart of near-miss counts by camera."""
    if events.empty or "camera_id" not in events.columns:
        return go.Figure().update_layout(title="Near-Miss Events by Camera (no data)")

    counts = (
        events.groupby("camera_id", as_index=False)
        .size()
        .rename(columns={"size": "near_miss_count"})
        .sort_values("near_miss_count", ascending=True)
    )
    fig = px.bar(
        counts,
        x="near_miss_count",
        y="camera_id",
        orientation="h",
        text="near_miss_count",
        color="near_miss_count",
        color_continuous_scale="Reds",
        labels={"near_miss_count": "Near-Miss Events", "camera_id": "Camera"},
        title="Near-Miss Events by Camera",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template=CHART_TEMPLATE, height=420, showlegend=False, coloraxis_showscale=False)
    return fig


def near_miss_risk_level_chart(events: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of High vs Medium near-miss counts."""
    if events.empty or "risk_level" not in events.columns:
        return go.Figure().update_layout(title="Near-Miss Risk Levels (no data)")

    counts = (
        events.groupby(["camera_id", "risk_level"], as_index=False)
        .size()
        .rename(columns={"size": "near_miss_count"})
    )
    risk_colors = {"High": "#e74c3c", "Medium": "#f1c40f"}
    fig = px.bar(
        counts,
        x="camera_id",
        y="near_miss_count",
        color="risk_level",
        barmode="group",
        color_discrete_map=risk_colors,
        labels={
            "camera_id": "Camera",
            "near_miss_count": "Near-Miss Count",
            "risk_level": "Risk Level",
        },
        title="High vs Medium Risk Near-Miss Counts",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420, xaxis_tickangle=-45)
    return fig


def delay_cost_by_camera_chart(delay_cost: pd.DataFrame) -> go.Figure:
    """Bar chart of estimated delay cost by camera."""
    if delay_cost.empty:
        return go.Figure().update_layout(title="Delay Cost by Camera (no data)")

    grouped = (
        delay_cost.groupby("camera_id", as_index=False)["estimated_delay_cost_usd"]
        .sum()
        .sort_values("estimated_delay_cost_usd", ascending=True)
    )
    fig = px.bar(
        grouped,
        x="estimated_delay_cost_usd",
        y="camera_id",
        orientation="h",
        text="estimated_delay_cost_usd",
        color="estimated_delay_cost_usd",
        color_continuous_scale="Oranges",
        labels={"estimated_delay_cost_usd": "Delay Cost (USD)", "camera_id": "Camera"},
        title="Estimated Delay Cost by Camera",
    )
    fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig.update_layout(template=CHART_TEMPLATE, height=420, showlegend=False, coloraxis_showscale=False)
    return fig


def ab_delay_cost_comparison_chart(ab_results: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing baseline vs simulated delay cost."""
    if ab_results.empty:
        return go.Figure().update_layout(title="Baseline vs Simulated Delay Cost (no data)")

    plot_df = ab_results[
        ["camera_id", "baseline_delay_cost_usd", "simulated_delay_cost_usd"]
    ].copy()
    melted = plot_df.melt(
        id_vars="camera_id",
        value_vars=["baseline_delay_cost_usd", "simulated_delay_cost_usd"],
        var_name="scenario_cost",
        value_name="delay_cost_usd",
    )
    melted["scenario_cost"] = melted["scenario_cost"].map(
        {
            "baseline_delay_cost_usd": "Baseline",
            "simulated_delay_cost_usd": "Simulated",
        }
    )
    fig = px.bar(
        melted,
        x="camera_id",
        y="delay_cost_usd",
        color="scenario_cost",
        barmode="group",
        color_discrete_map={"Baseline": PRIMARY_COLOR, "Simulated": ACCENT_COLOR},
        labels={
            "camera_id": "Camera",
            "delay_cost_usd": "Delay Cost (USD)",
            "scenario_cost": "Scenario",
        },
        title="Baseline vs Simulated Delay Cost",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420, xaxis_tickangle=-45)
    return fig


def ab_near_miss_comparison_chart(ab_results: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing baseline vs simulated near-miss risk."""
    if ab_results.empty:
        return go.Figure().update_layout(title="Baseline vs Simulated Near-Miss Risk (no data)")

    plot_df = ab_results[
        ["camera_id", "baseline_near_miss_risk", "simulated_near_miss_risk"]
    ].copy()
    melted = plot_df.melt(
        id_vars="camera_id",
        value_vars=["baseline_near_miss_risk", "simulated_near_miss_risk"],
        var_name="scenario_risk",
        value_name="near_miss_count",
    )
    melted["scenario_risk"] = melted["scenario_risk"].map(
        {
            "baseline_near_miss_risk": "Baseline",
            "simulated_near_miss_risk": "Simulated",
        }
    )
    fig = px.bar(
        melted,
        x="camera_id",
        y="near_miss_count",
        color="scenario_risk",
        barmode="group",
        color_discrete_map={"Baseline": "#e67e22", "Simulated": "#2ecc71"},
        labels={
            "camera_id": "Camera",
            "near_miss_count": "Near-Miss Count",
            "scenario_risk": "Scenario",
        },
        title="Baseline vs Simulated Near-Miss Risk",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420, xaxis_tickangle=-45)
    return fig


def forecast_actual_vs_predicted_chart(predictions: pd.DataFrame) -> go.Figure:
    """Scatter plot of actual vs predicted traffic density."""
    if predictions.empty:
        return go.Figure().update_layout(title="Actual vs Predicted Density (no data)")

    fig = px.scatter(
        predictions,
        x="traffic_density",
        y="predicted_traffic_density",
        color="scene_type" if "scene_type" in predictions.columns else None,
        hover_data=["camera_id"] if "camera_id" in predictions.columns else None,
        labels={
            "traffic_density": "Actual Density",
            "predicted_traffic_density": "Predicted Density",
            "scene_type": "Scene Type",
        },
        title="Actual vs Predicted Traffic Density",
    )
    max_val = max(
        predictions["traffic_density"].max(),
        predictions["predicted_traffic_density"].max(),
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Perfect prediction",
            line=dict(dash="dash", color="#95a5a6"),
        )
    )
    fig.update_layout(template=CHART_TEMPLATE, height=420)
    return fig


def forecast_residuals_chart(predictions: pd.DataFrame) -> go.Figure:
    """Histogram of forecast prediction residuals."""
    if predictions.empty or "prediction_error" not in predictions.columns:
        return go.Figure().update_layout(title="Forecast Residuals (no data)")

    fig = px.histogram(
        predictions,
        x="prediction_error",
        nbins=20,
        color="scene_type" if "scene_type" in predictions.columns else None,
        labels={"prediction_error": "Prediction Error (Actual − Predicted)", "count": "Frequency"},
        title="Forecast Residual Distribution",
    )
    fig.update_layout(template=CHART_TEMPLATE, height=380, bargap=0.05)
    return fig


def dataset_comparison_metric_chart(
    comparison: pd.DataFrame,
    metric: str,
    title: str,
) -> go.Figure:
    """Grouped bar chart for a single metric across datasets."""
    if comparison.empty or metric not in comparison.columns:
        return go.Figure().update_layout(title=f"{title} (no data)")

    label_col = "display_name" if "display_name" in comparison.columns else "dataset"
    df = comparison.sort_values(metric, ascending=True)
    fig = px.bar(
        df,
        x=metric,
        y=label_col,
        orientation="h",
        text=metric,
        color=label_col,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={metric: title, label_col: "Dataset"},
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, template=CHART_TEMPLATE, height=360)
    return fig


def dataset_comparison_detections_chart(comparison: pd.DataFrame) -> go.Figure:
    """Total detections by dataset."""
    return dataset_comparison_metric_chart(
        comparison,
        "total_detections",
        "Total Detections by Dataset",
    )


def dataset_comparison_vehicles_pedestrians_chart(comparison: pd.DataFrame) -> go.Figure:
    """Grouped bar of vehicle vs pedestrian counts by dataset."""
    if comparison.empty:
        return go.Figure().update_layout(title="Vehicles vs Pedestrians (no data)")

    label_col = "display_name" if "display_name" in comparison.columns else "dataset"
    melted = comparison[[label_col, "vehicle_count", "pedestrian_count"]].melt(
        id_vars=label_col,
        value_vars=["vehicle_count", "pedestrian_count"],
        var_name="object_group",
        value_name="count",
    )
    melted["object_group"] = melted["object_group"].map(
        {
            "vehicle_count": "Vehicles",
            "pedestrian_count": "Pedestrians",
        }
    )
    fig = px.bar(
        melted,
        x=label_col,
        y="count",
        color="object_group",
        barmode="group",
        text="count",
        color_discrete_map={"Vehicles": PRIMARY_COLOR, "Pedestrians": ACCENT_COLOR},
        labels={label_col: "Dataset", "count": "Detections", "object_group": "Type"},
        title="Vehicle vs Pedestrian Detections by Dataset",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template=CHART_TEMPLATE, height=400)
    return fig


def dataset_comparison_near_miss_chart(comparison: pd.DataFrame) -> go.Figure:
    """Near-miss event counts by dataset."""
    return dataset_comparison_metric_chart(
        comparison,
        "near_miss_count",
        "Near-Miss Events by Dataset",
    )


def dataset_comparison_density_chart(comparison: pd.DataFrame) -> go.Figure:
    """Average and max traffic density by dataset."""
    if comparison.empty:
        return go.Figure().update_layout(title="Traffic Density (no data)")

    label_col = "display_name" if "display_name" in comparison.columns else "dataset"
    melted = comparison[[label_col, "avg_density", "max_density"]].melt(
        id_vars=label_col,
        value_vars=["avg_density", "max_density"],
        var_name="density_metric",
        value_name="density",
    )
    melted["density_metric"] = melted["density_metric"].map(
        {"avg_density": "Average", "max_density": "Maximum"}
    )
    fig = px.bar(
        melted,
        x=label_col,
        y="density",
        color="density_metric",
        barmode="group",
        text="density",
        color_discrete_sequence=[SECONDARY_COLOR, ACCENT_COLOR],
        labels={label_col: "Dataset", "density": "Traffic Density", "density_metric": "Metric"},
        title="Traffic Density Comparison by Dataset",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template=CHART_TEMPLATE, height=400)
    return fig
