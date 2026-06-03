# Demo analytics CSVs (committed for Render)

These files power the public Streamlit dashboard without PostgreSQL or raw videos.

**Required (core tabs):**

- `traffic_detections_master.csv`
- `congestion_metrics_master.csv`
- `object_summary_master.csv`

**Optional (extended tabs — warnings if missing):**

- `near_miss_events.csv`, `near_miss_summary.csv`
- `delay_cost_analysis.csv`, `delay_cost_summary.csv`
- `ab_simulation_results.csv`, `ab_simulation_summary.csv`
- `congestion_features_master.csv`, `forecast_predictions.csv`
- `camera_metadata.csv`, `dataset_comparison.csv`

Hotspot map HTML: `outputs/traffic_hotspot_map.html`

Regenerate locally with the scripts in the project `README.md`; then recommit updated CSVs.
