"""
Urban Traffic Safety Engine — Streamlit analytics dashboard.

Supports batch master dataset with MVP fallback and extended analytics tabs.

Run:
    streamlit run src/traffic_safety/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from traffic_safety.dashboard.charts import (
    ab_delay_cost_comparison_chart,
    ab_near_miss_comparison_chart,
    confidence_histogram,
    congestion_donut_chart,
    congestion_level_by_camera_chart,
    dataset_comparison_density_chart,
    dataset_comparison_detections_chart,
    dataset_comparison_near_miss_chart,
    dataset_comparison_vehicles_pedestrians_chart,
    delay_cost_by_camera_chart,
    forecast_actual_vs_predicted_chart,
    forecast_residuals_chart,
    near_miss_events_by_camera_chart,
    near_miss_risk_level_chart,
    object_type_bar_chart,
    top_cameras_density_chart,
    traffic_density_by_camera_chart,
    traffic_density_line_chart,
    vehicle_distribution_by_camera_chart,
    vehicles_pedestrians_grouped_bar,
)
from traffic_safety.dashboard.data_loader import (
    DashboardData,
    DataLoadError,
    attach_scene_type,
    build_filtered_summary_from_detections,
    compute_kpis,
    count_comparison_ready_datasets,
    filter_ab_simulation,
    filter_congestion,
    filter_delay_cost,
    filter_detections,
    filter_forecast_predictions,
    filter_near_miss_events,
    filter_near_miss_summary,
    filter_summary,
    get_filter_options,
    load_dashboard_data,
    load_or_build_dataset_comparison,
    resolve_data_paths,
)

st.set_page_config(
    page_title="Urban Traffic Safety Engine",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .dashboard-header {
        background: linear-gradient(135deg, #1f4e79 0%, #2980b9 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .dashboard-header h1 { color: white; margin-bottom: 0.25rem; font-size: 2rem; }
    .dashboard-header p  { color: #dceefb; margin: 0; font-size: 1rem; }
    .mode-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label { font-size: 0.85rem; color: #64748b; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #1e293b; }
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1f4e79;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _row_count(df) -> int:
    """Return row count for an optional DataFrame without ambiguous truth checks."""
    return len(df) if df is not None else 0


@st.cache_data(show_spinner="Loading analytics data…")
def get_dashboard_data() -> DashboardData:
    return load_dashboard_data()


@st.cache_data(show_spinner="Resolving dataset paths…")
def get_resolved_paths():
    return resolve_data_paths()


def render_deploy_notices(data: DashboardData) -> None:
    """Surface missing optional analytics files without crashing."""
    from traffic_safety.dashboard.data_loader import is_csv_only_mode

    if is_csv_only_mode():
        st.info(
            "Public demo mode: analytics load from committed CSV files in "
            "`data/processed/`. PostgreSQL and local video pipelines are not required."
        )

    missing = [
        label.replace("_", " ")
        for label, available in data.optional_available.items()
        if not available
    ]
    if missing:
        st.warning(
            "Some optional analytics files are not bundled with this deployment: "
            + ", ".join(missing)
            + ". Related tabs will show placeholders until those CSVs are added."
        )


def render_header(mode: str) -> None:
    mode_label = "Batch Master Dataset" if mode == "master" else "MVP Dataset"
    st.markdown(
        f"""
        <div class="dashboard-header">
            <h1>🚦 Urban Traffic Safety Engine</h1>
            <p>
                End-to-end traffic safety analytics — computer vision, congestion KPIs,
                geospatial hotspots, near-miss detection, delay costs, A/B simulation,
                and ML forecasting.
            </p>
            <span class="mode-badge">{mode_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_filters(data: DashboardData) -> dict:
    st.sidebar.header("Filters")
    options = get_filter_options(data)

    filters: dict = {
        "camera_ids": st.sidebar.multiselect(
            "Camera ID",
            options=options["cameras"],
            default=options["cameras"],
        ),
        "object_types": st.sidebar.multiselect(
            "Object types",
            options=options["object_types"],
            default=options["object_types"],
        ),
        "congestion_levels": st.sidebar.multiselect(
            "Congestion level",
            options=options["congestion_levels"],
            default=options["congestion_levels"],
        ),
        "min_confidence": st.sidebar.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        ),
    }

    if options["scene_types"]:
        filters["scene_types"] = st.sidebar.multiselect(
            "Scene type",
            options=options["scene_types"],
            default=options["scene_types"],
        )
    else:
        filters["scene_types"] = None

    if options["risk_levels"]:
        filters["risk_levels"] = st.sidebar.multiselect(
            "Risk level",
            options=options["risk_levels"],
            default=options["risk_levels"],
        )
    else:
        filters["risk_levels"] = None

    st.sidebar.divider()
    st.sidebar.caption("Core dataset")
    st.sidebar.code(
        "\n".join(
            [
                data.paths.detections.name,
                data.paths.congestion.name,
                data.paths.summary.name,
            ]
        ),
        language=None,
    )

    st.sidebar.caption("Extended outputs")
    for label, available in data.optional_available.items():
        st.sidebar.write(f"{'✅' if available else '⬜'} {label.replace('_', ' ')}")

    return filters


def apply_filters(data: DashboardData, filters: dict) -> dict:
    """Apply sidebar filters across all loaded datasets."""
    congestion = attach_scene_type(data.congestion, data.camera_metadata, data.congestion_features)
    detections = attach_scene_type(data.detections, data.camera_metadata, data.congestion_features)
    summary = attach_scene_type(data.summary, data.camera_metadata, data.congestion_features)

    near_miss_events = data.near_miss_events
    if near_miss_events is not None:
        near_miss_events = attach_scene_type(
            near_miss_events, data.camera_metadata, data.congestion_features
        )

    delay_cost = data.delay_cost_analysis
    if delay_cost is not None:
        delay_cost = attach_scene_type(delay_cost, data.camera_metadata, data.congestion_features)

    ab_results = data.ab_simulation_results
    forecast = data.forecast_predictions

    filtered = {
        "detections": filter_detections(
            detections,
            object_types=filters["object_types"],
            camera_ids=filters["camera_ids"],
            scene_types=filters.get("scene_types"),
            min_confidence=filters["min_confidence"],
        ),
        "congestion": filter_congestion(
            congestion,
            camera_ids=filters["camera_ids"],
            scene_types=filters.get("scene_types"),
            congestion_levels=filters["congestion_levels"],
        ),
        "summary": filter_summary(
            summary,
            object_types=filters["object_types"],
            camera_ids=filters["camera_ids"],
            scene_types=filters.get("scene_types"),
        ),
        "near_miss_events": (
            filter_near_miss_events(
                near_miss_events,
                camera_ids=filters["camera_ids"],
                scene_types=filters.get("scene_types"),
                risk_levels=filters.get("risk_levels"),
            )
            if near_miss_events is not None
            else None
        ),
        "near_miss_summary": (
            filter_near_miss_summary(
                data.near_miss_summary,
                camera_ids=filters["camera_ids"],
                risk_levels=filters.get("risk_levels"),
            )
            if data.near_miss_summary is not None
            else None
        ),
        "delay_cost": (
            filter_delay_cost(
                delay_cost,
                camera_ids=filters["camera_ids"],
                scene_types=filters.get("scene_types"),
                congestion_levels=filters["congestion_levels"],
            )
            if delay_cost is not None
            else None
        ),
        "ab_results": (
            filter_ab_simulation(
                ab_results,
                camera_ids=filters["camera_ids"],
                scene_types=filters.get("scene_types"),
            )
            if ab_results is not None
            else None
        ),
        "forecast": (
            filter_forecast_predictions(
                forecast,
                camera_ids=filters["camera_ids"],
                scene_types=filters.get("scene_types"),
            )
            if forecast is not None
            else None
        ),
        "congestion_features": data.congestion_features,
    }
    filtered["chart_summary"] = build_filtered_summary_from_detections(
        filtered["detections"], data.mode
    )
    return filtered


def render_kpi_cards(kpis: dict) -> None:
    st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c5, c6, c7, c8 = st.columns(4)

    c1.metric("Total Detections", f"{kpis['total_detections']:,}")
    c2.metric("Total Vehicles", f"{kpis['total_vehicles']:,}")
    c3.metric("Total Pedestrians", f"{kpis['total_pedestrians']:,}")
    c4.metric("Near-Miss Events", f"{kpis['total_near_miss_events']:,}")

    c5.metric("High-Risk Near-Misses", f"{kpis['high_risk_near_misses']:,}")
    c6.metric("Est. Delay Cost", f"${kpis['total_delay_cost_usd']:,.2f}")
    savings = kpis["ab_cost_savings_usd"]
    c7.metric("A/B Cost Savings", f"${savings:,.2f}")
    c8.metric("Forecast RMSE", kpis["forecast_rmse"])


def render_overview_tab(data: DashboardData, filtered: dict, kpis: dict) -> None:
    render_kpi_cards(kpis)

    st.markdown('<p class="section-header">Portfolio Overview</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            object_type_bar_chart(filtered["chart_summary"]),
            use_container_width=True,
            key="overview_object_distribution",
        )
    with col2:
        st.plotly_chart(
            top_cameras_density_chart(filtered["congestion"]),
            use_container_width=True,
            key="overview_top_cameras_density",
        )

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            congestion_donut_chart(filtered["congestion"]),
            use_container_width=True,
            key="overview_congestion_donut",
        )
    with col4:
        if filtered["near_miss_events"] is not None and not filtered["near_miss_events"].empty:
            st.plotly_chart(
                near_miss_risk_level_chart(filtered["near_miss_events"]),
                use_container_width=True,
                key="overview_near_miss_risk_level",
            )
        else:
            st.info("Near-miss data not available. Run `python scripts/detect_near_misses.py`.")

    st.markdown("**Pipeline outputs loaded**")
    output_rows = [
        {"Output": "Detections", "Rows": len(data.detections), "Status": "✅ Core"},
        {"Output": "Congestion metrics", "Rows": len(data.congestion), "Status": "✅ Core"},
        {"Output": "Near-miss events", "Rows": _row_count(data.near_miss_events), "Status": "Optional"},
        {"Output": "Delay cost analysis", "Rows": _row_count(data.delay_cost_analysis), "Status": "Optional"},
        {"Output": "A/B simulation", "Rows": _row_count(data.ab_simulation_results), "Status": "Optional"},
        {"Output": "Forecast predictions", "Rows": _row_count(data.forecast_predictions), "Status": "Optional"},
    ]
    st.dataframe(output_rows, use_container_width=True, hide_index=True, key="overview_pipeline_outputs")


def render_detections_tab(data: DashboardData, filtered: dict) -> None:
    st.markdown('<p class="section-header">Computer Vision Detections</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            object_type_bar_chart(filtered["chart_summary"]),
            use_container_width=True,
            key="detections_object_distribution",
        )
    with col2:
        if data.mode == "master":
            st.plotly_chart(
                vehicle_distribution_by_camera_chart(filtered["detections"]),
                use_container_width=True,
                key="detections_vehicle_distribution_by_camera",
            )
        else:
            st.plotly_chart(
                vehicles_pedestrians_grouped_bar(filtered["congestion"]),
                use_container_width=True,
                key="detections_vehicles_pedestrians_grouped",
            )

    st.plotly_chart(
        confidence_histogram(filtered["detections"]),
        use_container_width=True,
        key="detections_confidence_histogram",
    )

    st.dataframe(
        filtered["detections"].head(500),
        use_container_width=True,
        hide_index=True,
        key="detections_table",
    )
    if len(filtered["detections"]) > 500:
        st.caption(f"Showing first 500 of {len(filtered['detections']):,} filtered rows.")


def render_congestion_tab(data: DashboardData, filtered: dict) -> None:
    st.markdown('<p class="section-header">Congestion Analytics</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            top_cameras_density_chart(filtered["congestion"]),
            use_container_width=True,
            key="congestion_top_cameras_density",
        )
    with col2:
        st.plotly_chart(
            congestion_donut_chart(filtered["congestion"]),
            use_container_width=True,
            key="congestion_donut",
        )

    if data.mode == "master" and "camera_id" in filtered["congestion"].columns:
        cameras = sorted(filtered["congestion"]["camera_id"].unique().tolist())
        selected = st.selectbox(
            "Selected camera (density timeline)",
            options=cameras,
            index=0,
            key="congestion_camera_select",
        )
        st.plotly_chart(
            traffic_density_by_camera_chart(filtered["congestion"], selected),
            use_container_width=True,
            key="congestion_density_by_camera",
        )
        st.plotly_chart(
            congestion_level_by_camera_chart(filtered["congestion"]),
            use_container_width=True,
            key="congestion_level_by_camera",
        )
    else:
        st.plotly_chart(
            traffic_density_line_chart(filtered["congestion"]),
            use_container_width=True,
            key="congestion_density_line",
        )

    st.dataframe(
        filtered["congestion"],
        use_container_width=True,
        hide_index=True,
        key="congestion_table",
    )


def render_hotspot_tab(data: DashboardData) -> None:
    st.markdown('<p class="section-header">Spatial Hotspot Map</p>', unsafe_allow_html=True)

    if data.hotspot_map_path is None:
        st.warning(
            "Hotspot map not found. Generate it with:\n\n"
            "```\n"
            "python scripts/generate_camera_metadata.py\n"
            "python scripts/create_hotspot_map.py\n"
            "```"
        )
        return

    try:
        map_html = Path(data.hotspot_map_path).read_text(encoding="utf-8")
    except OSError as exc:
        st.error(f"Could not read hotspot map: {exc}")
        return

    st.caption(f"Source: `{data.hotspot_map_path.name}` — Bloomington, IN camera locations")
    components.html(map_html, height=620, scrolling=True)

    if data.camera_metadata is not None:
        with st.expander("Camera metadata"):
            st.dataframe(
                data.camera_metadata,
                use_container_width=True,
                hide_index=True,
                key="hotspot_camera_metadata",
            )


def render_near_miss_tab(filtered: dict) -> None:
    st.markdown('<p class="section-header">Near-Miss Safety</p>', unsafe_allow_html=True)

    events = filtered["near_miss_events"]
    summary = filtered["near_miss_summary"]

    if events is None or events.empty:
        st.warning(
            "No near-miss data available. Run `python scripts/detect_near_misses.py`."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            near_miss_events_by_camera_chart(events),
            use_container_width=True,
            key="near_miss_by_camera",
        )
    with col2:
        st.plotly_chart(
            near_miss_risk_level_chart(events),
            use_container_width=True,
            key="near_miss_risk_level",
        )

    tab1, tab2 = st.tabs(["Events", "Summary"])
    with tab1:
        st.dataframe(events, use_container_width=True, hide_index=True, key="near_miss_events_table")
    with tab2:
        if summary is not None and not summary.empty:
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                key="near_miss_summary_table",
            )
        else:
            st.info("Near-miss summary not available.")


def render_delay_cost_tab(data: DashboardData, filtered: dict) -> None:
    st.markdown('<p class="section-header">Delay Cost Impact</p>', unsafe_allow_html=True)

    delay_cost = filtered["delay_cost"]
    if delay_cost is None or delay_cost.empty:
        st.warning(
            "No delay cost data available. Run `python scripts/compute_delay_cost.py`."
        )
        return

    total_cost = delay_cost["estimated_delay_cost_usd"].sum()
    st.metric("Filtered Estimated Delay Cost", f"${total_cost:,.2f}")

    st.plotly_chart(
        delay_cost_by_camera_chart(delay_cost),
        use_container_width=True,
        key="delay_cost_by_camera",
    )

    tab1, tab2 = st.tabs(["Analysis", "Summary"])
    with tab1:
        st.dataframe(
            delay_cost,
            use_container_width=True,
            hide_index=True,
            key="delay_cost_analysis_table",
        )
    with tab2:
        if data.delay_cost_summary is not None:
            summary = data.delay_cost_summary
            if filtered["delay_cost"] is not None and "camera_id" in summary.columns:
                summary = summary[summary["camera_id"].isin(filtered["delay_cost"]["camera_id"])]
            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
                key="delay_cost_summary_table",
            )


def render_ab_simulation_tab(data: DashboardData, filtered: dict) -> None:
    st.markdown('<p class="section-header">A/B Simulation — Protected Bike Lane</p>', unsafe_allow_html=True)

    ab_results = filtered["ab_results"]
    if ab_results is None or ab_results.empty:
        st.warning(
            "No A/B simulation data available. Run `python scripts/run_ab_simulation.py`."
        )
        return

    savings = (
        ab_results["baseline_delay_cost_usd"].sum()
        - ab_results["simulated_delay_cost_usd"].sum()
    )
    risk_reduction = (
        ab_results["baseline_near_miss_risk"].sum()
        - ab_results["simulated_near_miss_risk"].sum()
    )
    col1, col2 = st.columns(2)
    col1.metric("Simulated Delay Cost Savings", f"${savings:,.2f}")
    col2.metric("Simulated Near-Miss Reduction", f"{int(risk_reduction):,}")

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            ab_delay_cost_comparison_chart(ab_results),
            use_container_width=True,
            key="ab_simulation_cost",
        )
    with col4:
        st.plotly_chart(
            ab_near_miss_comparison_chart(ab_results),
            use_container_width=True,
            key="ab_simulation_near_miss",
        )

    tab1, tab2 = st.tabs(["Results", "Summary by Scene"])
    with tab1:
        st.dataframe(
            ab_results,
            use_container_width=True,
            hide_index=True,
            key="ab_simulation_results_table",
        )
    with tab2:
        if data.ab_simulation_summary is not None:
            st.dataframe(
                data.ab_simulation_summary,
                use_container_width=True,
                hide_index=True,
                key="ab_simulation_summary_table",
            )


def render_forecasting_tab(filtered: dict) -> None:
    st.markdown('<p class="section-header">Traffic Density Forecasting</p>', unsafe_allow_html=True)

    forecast = filtered["forecast"]
    if forecast is None or forecast.empty:
        st.warning(
            "No forecast predictions available. Run:\n\n"
            "```\n"
            "python scripts/generate_external_features.py\n"
            "python scripts/train_forecast_model.py\n"
            "```"
        )
        return

    if "prediction_error" in forecast.columns:
        rmse = (forecast["prediction_error"] ** 2).mean() ** 0.5
        mae = forecast["prediction_error"].abs().mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse:.4f}")
        col2.metric("MAE", f"{mae:.4f}")
        col3.metric("Predictions", f"{len(forecast):,}")

    col4, col5 = st.columns(2)
    with col4:
        st.plotly_chart(
            forecast_actual_vs_predicted_chart(forecast),
            use_container_width=True,
            key="forecast_actual_vs_predicted",
        )
    with col5:
        st.plotly_chart(
            forecast_residuals_chart(forecast),
            use_container_width=True,
            key="forecast_residuals",
        )

    st.dataframe(
        forecast,
        use_container_width=True,
        hide_index=True,
        key="forecast_predictions_table",
    )


def render_error_state(error: DataLoadError) -> None:
    st.error("Dashboard data could not be loaded.")
    st.code(str(error))


def render_dataset_comparison_tab(data: DashboardData) -> None:
    """Cross-dataset benchmarking tab (Mixkit, AI City, BDD100K)."""
    st.markdown('<div class="section-header">Dataset Comparison</div>', unsafe_allow_html=True)

    comparison = data.dataset_comparison
    if comparison is None:
        comparison = load_or_build_dataset_comparison()

    if comparison is None or comparison.empty:
        st.warning(
            "No dataset comparison data available. Run "
            "`python scripts/compare_datasets.py` after processing at least one dataset."
        )
        return

    ready_count = count_comparison_ready_datasets(comparison)
    if ready_count < 2:
        st.warning(
            "Only one dataset has processed detections. Register and run "
            "`python scripts/process_dataset.py --dataset aicity` (or bdd100k) "
            "to enable full cross-dataset charts."
        )

    display_cols = [
        c
        for c in [
            "display_name",
            "video_count",
            "camera_count",
            "total_detections",
            "vehicle_count",
            "pedestrian_count",
            "near_miss_count",
            "avg_density",
            "max_density",
            "status",
        ]
        if c in comparison.columns
    ]
    st.subheader("Dataset Summary")
    st.dataframe(
        comparison[display_cols],
        use_container_width=True,
        hide_index=True,
        key="dataset_comparison_summary_table",
    )

    st.subheader("Detections by Dataset")
    st.plotly_chart(
        dataset_comparison_detections_chart(comparison),
        use_container_width=True,
        key="dataset_comparison_detections_chart",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            dataset_comparison_vehicles_pedestrians_chart(comparison),
            use_container_width=True,
            key="dataset_comparison_vp_chart",
        )
    with col2:
        st.plotly_chart(
            dataset_comparison_near_miss_chart(comparison),
            use_container_width=True,
            key="dataset_comparison_near_miss_chart",
        )

    st.subheader("Density Comparison")
    st.plotly_chart(
        dataset_comparison_density_chart(comparison),
        use_container_width=True,
        key="dataset_comparison_density_chart",
    )


def main() -> None:
    try:
        data = get_dashboard_data()
    except DataLoadError as exc:
        render_error_state(exc)
        return

    render_header(data.mode)
    render_deploy_notices(data)
    filters = render_sidebar_filters(data)
    filtered = apply_filters(data, filters)

    if filtered["detections"].empty:
        st.warning("No detections match the current filters. Adjust sidebar settings.")
        return

    kpis = compute_kpis(
        filtered["detections"],
        filtered["congestion"],
        data.mode,
        near_miss_events=filtered["near_miss_events"],
        delay_cost_analysis=filtered["delay_cost"],
        ab_simulation_results=filtered["ab_results"],
        forecast_predictions=filtered["forecast"],
    )

    tabs = st.tabs(
        [
            "Overview",
            "Computer Vision Detections",
            "Congestion Analytics",
            "Spatial Hotspot Map",
            "Near-Miss Safety",
            "Delay Cost Impact",
            "A/B Simulation",
            "Forecasting",
            "Dataset Comparison",
        ]
    )

    with tabs[0]:
        render_overview_tab(data, filtered, kpis)
    with tabs[1]:
        render_detections_tab(data, filtered)
    with tabs[2]:
        render_congestion_tab(data, filtered)
    with tabs[3]:
        render_hotspot_tab(data)
    with tabs[4]:
        render_near_miss_tab(filtered)
    with tabs[5]:
        render_delay_cost_tab(data, filtered)
    with tabs[6]:
        render_ab_simulation_tab(data, filtered)
    with tabs[7]:
        render_forecasting_tab(filtered)
    with tabs[8]:
        render_dataset_comparison_tab(data)

    st.divider()
    st.caption(
        f"Dataset: `{data.paths.detections.name}` · "
        f"{len(filtered['detections']):,} detections · "
        f"{len(filtered['congestion']):,} congestion rows"
    )


if __name__ == "__main__":
    main()
