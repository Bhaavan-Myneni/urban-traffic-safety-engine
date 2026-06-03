# Urban Traffic Safety Engine

> End-to-end computer vision, data engineering, and analytics platform for detecting urban traffic risk, quantifying congestion costs, and simulating infrastructure interventions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Detection-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Airflow](https://img.shields.io/badge/Orchestration-Airflow-orange.svg)](https://airflow.apache.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)

---

## Problem Statement

Urban intersections and corridors generate massive volumes of unstructured video data, but cities rarely convert that footage into actionable safety and mobility insights. Manual review does not scale, and raw detection counts alone do not answer questions that planners, DOT analysts, and safety engineers actually care about:

- Where are pedestrians and cyclists most exposed to vehicle conflict?
- How much economic value is lost to congestion at each camera location?
- Which infrastructure changes could reduce near-miss risk without increasing delay?

The **Urban Traffic Safety Engine** solves this by turning traffic camera footage into a repeatable analytics pipeline: detect objects with YOLOv8, aggregate congestion KPIs, flag near-miss events, estimate delay costs, simulate policy interventions, and deliver results through PostgreSQL, SQL analytics, ML forecasting, and an interactive Streamlit dashboard.

---

## Business Impact

| Stakeholder | Value delivered |
|-------------|-----------------|
| **City traffic operations** | Camera-level congestion KPIs and ranked hotspot maps for prioritizing interventions |
| **Safety engineering** | Near-miss detection and risk scoring to identify high-exposure pedestrian/vehicle interactions |
| **Transportation economics** | Delay cost modeling ($18/hr value of time) to quantify congestion in dollars |
| **Policy / planning** | A/B simulation of protected bike lane scenarios with safety and cost trade-offs |
| **Data & ML teams** | Production-style ETL, PostgreSQL loading, Airflow orchestration, and forecasting prototypes |

**Portfolio-scale results on the batch master dataset:**

- **26,699** YOLO detections across **29** traffic videos and **26** camera feeds
- **81** near-miss events flagged (**8** high-risk, **73** medium-risk)
- **$20,770.80** estimated delay cost across the network
- **A/B simulation:** near-miss events reduced **81 → 62** under a protected bike lane scenario (**$431.70** simulated delay cost savings)

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        V[Mixkit Traffic Videos<br/>29 clips / 26 cameras]
        M[video_manifest.yaml]
    end

    subgraph CV["Computer Vision"]
        Y[YOLOv8 Batch Detection]
        D[traffic_detections_master.csv<br/>26,699 rows]
    end

    subgraph Analytics["Analytics & Safety"]
        C[Congestion Metrics]
        N[Near-Miss Detection]
        $[Delay Cost Analysis]
        G[Geospatial Hotspot Map]
        W[Weather & Event Features]
    end

    subgraph Storage["Storage & Orchestration"]
        PG[(PostgreSQL)]
        AF[Apache Airflow DAG]
    end

    subgraph Intelligence["ML & Simulation"]
        F[Density Forecasting]
        AB[A/B Infrastructure Simulation]
    end

    subgraph Presentation["Presentation Layer"]
        SQL[SQL Analytics Queries]
        SB[Streamlit Dashboard<br/>9 tabs]
    end

    V --> Y
    M --> Y
    Y --> D
    D --> C
    D --> N
    C --> $
    C --> W
    C --> F
    C --> PG
    N --> AB
    $ --> AB
    G --> SB
    D --> PG
    C --> SB
    N --> SB
    $ --> SB
    AB --> SB
    F --> SB
    SQL --> PG
    AF --> V
    AF --> Y
    AF --> C
    AF --> PG
    AF --> F
```

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Language** | Python 3.10+ |
| **Computer vision** | YOLOv8 (Ultralytics), OpenCV |
| **Data processing** | pandas, NumPy, PyYAML |
| **Database** | PostgreSQL 16, SQLAlchemy, psycopg2 |
| **Orchestration** | Apache Airflow (daily DAG) |
| **Machine learning** | scikit-learn (RandomForest, GradientBoosting) |
| **Geospatial** | Folium, simulated Bloomington, IN camera metadata |
| **Visualization** | Streamlit, Plotly |
| **Dev tooling** | pytest, ruff, black, Docker Compose |

---

## Dataset Summary

| Asset | Description |
|-------|-------------|
| **Video source** | 29 Mixkit traffic clips (intersections, highways, city streets, aerial, dashcam) |
| **Camera manifest** | `data/raw/videos/video_manifest.yaml` — maps `camera_id`, `video_id`, scene type |
| **Detections** | `traffic_detections_master.csv` — 26,699 bounding-box records |
| **Congestion KPIs** | `congestion_metrics_master.csv` — density, vehicle/pedestrian counts, congestion tier |
| **Object summary** | `object_summary_master.csv` — detection counts by camera and object type |
| **Near-miss events** | `near_miss_events.csv` — pedestrian/bicycle vs vehicle proximity events |
| **Delay costs** | `delay_cost_analysis.csv` — economic impact by camera/minute |
| **External features** | `congestion_features_master.csv` — synthetic weather & event features |
| **Geospatial** | `camera_metadata.csv` + `outputs/traffic_hotspot_map.html` |

**Detection classes:** person, bicycle, car, truck, bus, motorcycle  
**Congestion tiers:** Low · Medium · High · Severe

---

## Pipeline Workflow

```text
1. Download videos          →  scripts/download_batch_videos.py
2. YOLO batch detection     →  scripts/batch_detection.py
3. Compute congestion KPIs  →  scripts/compute_metrics.py --master
4. Near-miss detection      →  scripts/detect_near_misses.py
5. Delay cost analysis      →  scripts/compute_delay_cost.py
6. Camera metadata + map    →  scripts/generate_camera_metadata.py
                              →  scripts/create_hotspot_map.py
7. External feature engineering → scripts/generate_external_features.py
8. Load to PostgreSQL       →  scripts/load_to_postgres.py
9. Train forecast model     →  scripts/train_forecast_model.py
10. A/B simulation          →  scripts/run_ab_simulation.py
11. Launch dashboard          →  streamlit run src/traffic_safety/dashboard/app.py
```

**Airflow DAG** (`dags/traffic_safety_pipeline.py`) automates steps 1–5 and 8–9 on a daily schedule.

---

## Key Outputs

| Metric | Result |
|--------|--------|
| Traffic videos processed | **29** |
| YOLO detections | **26,699** |
| Camera feeds analyzed | **26** |
| Near-miss events | **81** (8 High · 73 Medium) |
| Estimated delay cost | **$20,770.80** |
| A/B near-miss reduction | **81 → 62** events |
| A/B delay cost savings | **$431.70** |
| Forecast RMSE (hold-out) | **5.51** (RandomForestRegressor) |

### Generated artifacts

```
data/processed/
├── traffic_detections_master.csv
├── congestion_metrics_master.csv
├── object_summary_master.csv
├── near_miss_events.csv
├── near_miss_summary.csv
├── delay_cost_analysis.csv
├── delay_cost_summary.csv
├── ab_simulation_results.csv
├── ab_simulation_summary.csv
├── congestion_features_master.csv
├── forecast_predictions.csv
└── camera_metadata.csv

outputs/
└── traffic_hotspot_map.html

models/
└── traffic_forecast_model.pkl
```

---

## Dashboard

Interactive **Streamlit** dashboard with **8 tabs**:

| Tab | Features |
|-----|----------|
| **Overview** | KPI cards, object distribution, top cameras, pipeline status |
| **Computer Vision Detections** | YOLO outputs, confidence histogram, detection table |
| **Congestion Analytics** | Density rankings, congestion tiers, camera timelines |
| **Spatial Hotspot Map** | Embedded Folium map (Bloomington, IN) |
| **Near-Miss Safety** | Risk charts, event tables |
| **Delay Cost Impact** | Cost by camera, analysis tables |
| **A/B Simulation** | Baseline vs simulated cost and safety |
| **Forecasting** | Actual vs predicted density, residuals, RMSE |

### Screenshots

> Add screenshots to `docs/images/` and reference them below before publishing to GitHub.

| Overview | Congestion Analytics |
|----------|-------------------|
| ![Overview dashboard](docs/images/dashboard_overview.png) | ![Congestion analytics](docs/images/dashboard_congestion.png) |

| Near-Miss Safety | Spatial Hotspot Map |
|------------------|---------------------|
| ![Near-miss safety](docs/images/dashboard_near_miss.png) | ![Hotspot map](docs/images/dashboard_hotspot_map.png) |

```bash
PYTHONPATH=src streamlit run src/traffic_safety/dashboard/app.py
```

Open **http://localhost:8501** (or the port shown in your terminal).

---

## PostgreSQL Setup

### 1. Start PostgreSQL with Docker

```bash
docker compose -f docker/docker-compose.yml up -d
```

Docker maps host port **5433** → container port **5432**.

### 2. Configure environment

```bash
cp .env.example .env
```

Set in `.env`:

```bash
DATABASE_URL=postgresql+psycopg2://traffic_user:changeme@localhost:5433/traffic_safety
```

### 3. Load processed data

```bash
PYTHONPATH=src python scripts/load_to_postgres.py --verbose
```

### Tables

| Table | Description |
|-------|-------------|
| `traffic_detections` | YOLO detection records |
| `congestion_metrics` | Per-camera/minute congestion KPIs |
| `object_summary` | Detection counts by camera and object type |

---

## Airflow Setup

### 1. Install Airflow (separate environment recommended)

```bash
pip install "apache-airflow>=2.8.0"
```

### 2. Initialize Airflow

```bash
export AIRFLOW_HOME=./airflow
export PYTHONPATH=./src

airflow db init
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com

airflow variables set DATABASE_URL \
  "postgresql+psycopg2://traffic_user:changeme@localhost:5433/traffic_safety"
```

### 3. Start scheduler and webserver

```bash
airflow dags unpause traffic_safety_pipeline
airflow webserver & airflow scheduler
```

### DAG: `traffic_safety_pipeline`

Daily pipeline:

```text
download_batch_videos → run_batch_detection → compute_congestion_metrics
  → load_to_postgres → train_forecast_model
```

---

## How to Run the Project

### Prerequisites

- Python 3.10+
- Docker (optional, for PostgreSQL)
- 4 GB+ RAM recommended for YOLO batch detection

### Installation

```bash
git clone https://github.com/<your-username>/urban-traffic-safety-engine.git
cd urban-traffic-safety-engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### Full batch pipeline (recommended)

```bash
# 1. Download videos
python scripts/download_batch_videos.py

# 2. Run YOLO detection (multi-process)
python scripts/batch_detection.py --workers 4 --stride 2

# 3. Analytics & safety
python scripts/compute_metrics.py --master
python scripts/detect_near_misses.py
python scripts/compute_delay_cost.py

# 4. Geospatial & features
python scripts/generate_camera_metadata.py
python scripts/create_hotspot_map.py
python scripts/generate_external_features.py

# 5. ML & simulation
python scripts/train_forecast_model.py
python scripts/run_ab_simulation.py

# 6. Database (optional)
docker compose -f docker/docker-compose.yml up -d
PYTHONPATH=src python scripts/load_to_postgres.py

# 7. Dashboard
PYTHONPATH=src streamlit run src/traffic_safety/dashboard/app.py
```

### MVP quick start (single video)

```bash
python scripts/download_sample_video.py
python scripts/run_detection.py
python scripts/compute_metrics.py
PYTHONPATH=src streamlit run src/traffic_safety/dashboard/app.py
```

---

## SQL Analytics

Portfolio-ready PostgreSQL queries live in [`sql/traffic_analytics_queries.sql`](sql/traffic_analytics_queries.sql).

**10 advanced queries** covering:

- Total detections by object type
- Top 10 busiest cameras by traffic density
- Vehicle vs pedestrian ratio by camera
- Congestion level distribution
- Peak congestion minute per camera
- Average confidence by object type
- Camera ranking with `RANK()`
- Rolling average traffic density
- Percentile ranking of cameras
- High-risk cameras (Severe congestion)

```bash
psql -U traffic_user -d traffic_safety -f sql/traffic_analytics_queries.sql
```

---

## Forecasting Model Results

Traffic density forecasting uses enriched congestion features (weather, events, lag/rolling density).

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **RandomForestRegressor** | 3.69 | **5.51** | **0.54** |
| GradientBoostingRegressor | 4.31 | 6.12 | 0.44 |

**Top features:** `rolling_mean_3_density`, `avg_vehicles_per_frame`, `vehicle_count`

> Note: The current dataset has 26 rows (prototype scale). The pipeline is structured to scale to hourly multi-camera feeds in production.

```bash
python scripts/generate_external_features.py
PYTHONPATH=src python scripts/train_forecast_model.py --verbose
```

Outputs: `models/traffic_forecast_model.pkl`, `data/processed/forecast_predictions.csv`

---

## Dataset Expansion

The engine supports multiple real-world traffic video sources while keeping the **29-video Mixkit MVP** pipeline unchanged. The same YOLO → metrics → near-miss → delay-cost workflow runs per dataset via manifests and CLI flags.

| Dataset | Role | Video folder | Manifest |
|---------|------|--------------|----------|
| **Mixkit** | Portfolio MVP (default) | `data/raw/videos/` | `video_manifest.yaml` |
| **AI City Challenge** | Urban intersection / highway benchmarks | `data/raw/aicity_videos/` | `aicity_manifest.yaml` |
| **BDD100K** | Diverse driving scenes (Berkeley DeepDrive) | `data/raw/bdd100k_videos/` | `bdd100k_manifest.yaml` |
| **Kaggle Traffic (extra)** | 3 clips from notebook exploration | `data/raw/kaggle_traffic_videos/` | `kaggle_traffic_manifest.yaml` |

External dataset notes: `data/external/aicity/`, `data/external/bdd100k/`, `data/external/kaggle_traffic/`.  
Reference notebook: `notebooks/object-detection-from-a-traffic-video.ipynb`.

### Phase 2: AI City / WTS development subset

Use this workflow to add **20–50 clips** from the AI City Challenge **Track 2 (WTS)** external **BDD_PC_5K** pack. The helper script only copies from a folder you already extracted — it does **not** download data (access may require form approval).

**Workflow:**

1. Submit the [WTS dataset access form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSe6eshgQQyf1wZmJkgnqsoDaFb_h-673qG7VHPxapkhh30_Gw/viewform).
2. Download the **BDD_PC_5K** video archive (see [WTS dataset README](https://github.com/woven-visionai/wts-dataset) or `data/external/aicity/README.md`).
3. Extract the zip locally (e.g. `external/BDD_PC_5K/videos/train/`).
4. Copy a subset into the project and register the manifest:

```bash
bash scripts/fetch_aicity_subset.sh \
  --source-dir /path/to/extracted/videos \
  --limit 30
```

5. Run the full AI City analytics pipeline:

```bash
python scripts/process_dataset.py --dataset aicity
```

6. Compare Mixkit vs AI City (and BDD100K when ready):

```bash
python scripts/compare_datasets.py
```

Copied files are renamed to `aicity_001.mp4`, `aicity_002.mp4`, … under `data/raw/aicity_videos/`. The script then runs `register_dataset.py` automatically.

### Extra: Kaggle traffic videos (from your `.ipynb`)

The notebook `notebooks/object-detection-from-a-traffic-video.ipynb` uses the Kaggle dataset **road-traffic-video-monitoring** (`traffic_video.avi`, `traffic_detection.mp4`, `road_trafifc.mp4`). Import those files after you download them locally:

```bash
bash scripts/fetch_kaggle_traffic_subset.sh \
  --source-dir /path/to/road-traffic-video-monitoring \
  --limit 10

python scripts/process_dataset.py --dataset kaggle_traffic
python scripts/compare_datasets.py
```

This does **not** auto-download from Kaggle. The engine runs **YOLOv8** batch detection on the copied clips (the notebook’s PixelLib/Mask R-CNN path is reference only).

Generic import for any supported dataset:

```bash
bash scripts/fetch_dataset_subset.sh --dataset kaggle_traffic --source-dir PATH --limit 10
```

### Manifest schema

```yaml
videos:
  - camera_id: CAM_AICITY_001
    video_id: aicity_clip_01
    file_name: clip_01.mp4
    scene_type: intersection
    road_type: urban
```

Legacy Mixkit entries use `filename` and `scene` instead of `file_name` / `scene_type` — both are supported.

### 1. Register a dataset

Place `.mp4` (or `.mov`, `.avi`, `.mkv`, `.webm`) files in the dataset folder, then auto-generate a manifest:

```bash
python scripts/register_dataset.py \
  --video-dir data/raw/aicity_videos \
  --dataset aicity
```

### 2. Run batch detection (any manifest)

Defaults still target Mixkit (`traffic_detections_master.csv`):

```bash
python scripts/batch_detection.py

python scripts/batch_detection.py \
  --manifest data/raw/aicity_videos/aicity_manifest.yaml \
  --video-dir data/raw/aicity_videos \
  --output data/processed/aicity/traffic_detections_aicity.csv
```

### 3. End-to-end analytics per dataset

```bash
python scripts/process_dataset.py --dataset aicity
python scripts/process_dataset.py --dataset bdd100k
python scripts/process_dataset.py --dataset mixkit --skip-detection   # analytics only
```

Outputs per dataset (Mixkit uses `*_master.csv` in `data/processed/`):

- `traffic_detections_<dataset>.csv`
- `congestion_metrics_<dataset>.csv`
- `object_summary_<dataset>.csv`
- `near_miss_events_<dataset>.csv`
- `delay_cost_analysis_<dataset>.csv`

### 4. Compare datasets

```bash
python scripts/compare_datasets.py
```

Writes `data/processed/dataset_comparison.csv` and powers the **Dataset Comparison** tab in the Streamlit dashboard.

### Validation

The pipeline validates missing manifests, invalid YAML, empty video folders, missing files listed in manifests, and logs per-dataset statistics before detection. Failed video jobs are logged without stopping the full batch.

---

## Future Improvements

- [ ] Expand to hourly/minute-level time series across multi-day camera feeds
- [ ] Replace synthetic weather/event features with live API integrations (OpenWeather, event calendars)
- [ ] Add Alembic migrations and DB-backed dashboard reads
- [ ] Upgrade near-miss logic with trajectory tracking and TTC (time-to-collision)
- [ ] Deploy dashboard and API on cloud infrastructure (AWS/GCP)
- [ ] Add Great Expectations or dbt tests for data quality gates
- [ ] Real-time inference pipeline with Kafka or RTSP stream ingestion
- [ ] Calibrate A/B simulation with observed before/after intervention data

---

## Resume Bullets

Use these directly on a data engineering, analytics, or ML portfolio resume:

- Built an end-to-end **Urban Traffic Safety Engine** processing **29 traffic videos** and **26,699 YOLOv8 detections** across **26 camera feeds** with batch multiprocessing and modular Python pipelines.
- Engineered congestion KPIs, **81 near-miss safety events**, and **$20,770.80 delay cost estimates** to translate computer vision outputs into operational and economic insights.
- Designed a **PostgreSQL + SQLAlchemy** data layer and **10 advanced SQL analytics queries** using CTEs, window functions, and percentile ranking for camera-level traffic analysis.
- Delivered a **9-tab Streamlit dashboard** with Plotly visualizations, Folium geospatial hotspot maps, multi-dataset comparison, and interactive filters for stakeholders.
- Implemented **scikit-learn traffic density forecasting** (RMSE **5.51**) with weather/event feature engineering and prototype lag/rolling temporal features.
- Simulated a **protected bike lane A/B intervention**, reducing near-miss events **81 → 62** and modeling **$431.70** in delay cost savings.
- Orchestrated daily batch pipelines with **Apache Airflow** (download → detect → metrics → PostgreSQL → forecast) using production-style CLI script integration.

---

## Project Structure

```
urban-traffic-safety-engine/
├── dags/                    # Airflow DAG definitions
├── data/
│   ├── external/            # AI City / BDD100K download notes
│   ├── raw/
│   │   ├── videos/          # Mixkit MVP videos + manifest
│   │   ├── aicity_videos/   # AI City Challenge clips
│   │   ├── bdd100k_videos/  # BDD100K clips
│   │   └── kaggle_traffic_videos/  # Kaggle notebook extra clips
│   └── processed/           # Analytics CSV outputs (+ per-dataset subfolders)
├── docker/                  # PostgreSQL Docker Compose
├── models/                  # YOLO weights & forecast model pickle
├── outputs/                 # Folium hotspot map HTML
├── scripts/                 # CLI pipeline entry points
├── sql/                     # PostgreSQL analytics queries
└── src/traffic_safety/
    ├── analytics/           # Metrics, near-miss, delay cost, A/B sim
    ├── dashboard/           # Streamlit app
    ├── db/                  # SQLAlchemy models & connection
    ├── detection/           # YOLO detection utilities
    └── ml/                  # Forecasting training pipeline
```

---

## Development

```bash
pytest
ruff check src tests
black src tests
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Bhaavan Myneni** — [GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)
