"""
Urban Traffic Safety Engine — end-to-end Airflow pipeline.

Orchestrates the batch video → detection → analytics → PostgreSQL → ML workflow
by invoking existing CLI scripts (no duplicated business logic).

Schedule : daily
Tasks    : download → detect → metrics → postgres → forecast

Airflow setup (local):
    export AIRFLOW_HOME=./airflow
    export PYTHONPATH=./src
    airflow db init
    airflow variables set DATABASE_URL "postgresql+psycopg2://traffic_user:changeme@localhost:5433/traffic_safety"
    airflow dags unpause traffic_safety_pipeline
    airflow webserver & airflow scheduler
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# ---------------------------------------------------------------------------
# Paths & runtime configuration
# ---------------------------------------------------------------------------
# DAG lives in <project_root>/dags/ — resolve project root once for all tasks.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = os.environ.get("TRAFFIC_SAFETY_PYTHON_BIN", "python3")
SRC_PATH = PROJECT_ROOT / "src"

# Shared environment injected into every BashOperator task.
# DATABASE_URL can be overridden via an Airflow Variable (see load task below).
PIPELINE_ENV: dict[str, str] = {
    "PYTHONPATH": str(SRC_PATH),
    "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
    # Batch detection tuning (optional overrides)
    "BATCH_DETECTION_WORKERS": os.environ.get("BATCH_DETECTION_WORKERS", "4"),
    "BATCH_DETECTION_STRIDE": os.environ.get("BATCH_DETECTION_STRIDE", "2"),
}

DEFAULT_ARGS = {
    "owner": "traffic-safety",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def project_script_cmd(script_name: str, script_args: str = "") -> str:
    """
    Build a fail-fast shell command that runs a project script from repo root.

    Args:
        script_name: Script filename under scripts/ (e.g. batch_detection.py).
        script_args: Additional CLI flags passed verbatim to the script.
    """
    script_path = PROJECT_ROOT / "scripts" / script_name
    args_suffix = f" {script_args}" if script_args else ""
    return (
        "set -euo pipefail\n"
        f'cd "{PROJECT_ROOT}"\n'
        f'export PYTHONPATH="{SRC_PATH}"\n'
        f'"{PYTHON_BIN}" "{script_path}"{args_suffix}'
    )


DAG_DOC = """
## Urban Traffic Safety Engine Pipeline

Daily batch pipeline for the Urban Traffic Safety Engine portfolio project.

### Stages
1. **download_batch_videos** — fetch Mixkit traffic clips from manifest
2. **run_batch_detection** — YOLOv8 batch inference across all cameras
3. **compute_congestion_metrics** — aggregate detections into KPI CSVs
4. **load_to_postgres** — bulk-load master CSVs into PostgreSQL
5. **train_forecast_model** — train traffic density forecasting model

### Required Airflow Variable
- `DATABASE_URL` — PostgreSQL connection string for the load step
  (use port **5433** — Docker maps host 5433 → container 5432)

### Optional environment variables
- `TRAFFIC_SAFETY_PYTHON_BIN` — Python interpreter (default: python3)
- `BATCH_DETECTION_WORKERS` — parallel detection workers (default: 4)
- `BATCH_DETECTION_STRIDE` — frame sampling stride (default: 2)
- `LOG_LEVEL` — logging verbosity (default: INFO)
"""

with DAG(
    dag_id="traffic_safety_pipeline",
    default_args=DEFAULT_ARGS,
    description="Daily traffic video ingestion, detection, analytics, DB load, and forecasting",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["traffic-safety", "detection", "analytics", "postgres", "ml"],
    doc_md=DAG_DOC,
) as dag:
    start = EmptyOperator(task_id="start")

    # ------------------------------------------------------------------
    # Task 1: Download batch traffic videos from Mixkit manifest
    # Script : scripts/download_batch_videos.py
    # ------------------------------------------------------------------
    download_batch_videos = BashOperator(
        task_id="download_batch_videos",
        bash_command=project_script_cmd("download_batch_videos.py", "--verbose"),
        env=PIPELINE_ENV,
        execution_timeout=timedelta(hours=2),
    )

    # ------------------------------------------------------------------
    # Task 2: Run YOLOv8 batch detection across all manifest videos
    # Script : scripts/batch_detection.py
    # Output : data/processed/traffic_detections_master.csv
    # ------------------------------------------------------------------
    run_batch_detection = BashOperator(
        task_id="run_batch_detection",
        bash_command=(
            "set -euo pipefail\n"
            f'cd "{PROJECT_ROOT}"\n'
            f'export PYTHONPATH="{SRC_PATH}"\n'
            'export BATCH_DETECTION_WORKERS="${BATCH_DETECTION_WORKERS:-4}"\n'
            'export BATCH_DETECTION_STRIDE="${BATCH_DETECTION_STRIDE:-2}"\n'
            f'"{PYTHON_BIN}" "{PROJECT_ROOT / "scripts" / "batch_detection.py"}" '
            '--workers "${BATCH_DETECTION_WORKERS}" '
            '--stride "${BATCH_DETECTION_STRIDE}" '
            "--verbose"
        ),
        env=PIPELINE_ENV,
        execution_timeout=timedelta(hours=6),
    )

    # ------------------------------------------------------------------
    # Task 3: Compute congestion metrics and object summaries (master mode)
    # Script : scripts/compute_metrics.py --master
    # Outputs: congestion_metrics_master.csv, object_summary_master.csv
    # ------------------------------------------------------------------
    compute_congestion_metrics = BashOperator(
        task_id="compute_congestion_metrics",
        bash_command=project_script_cmd("compute_metrics.py", "--master --verbose"),
        env=PIPELINE_ENV,
        execution_timeout=timedelta(hours=1),
    )

    # ------------------------------------------------------------------
    # Task 4: Load processed master CSVs into PostgreSQL
    # Script : scripts/load_to_postgres.py
    # Requires DATABASE_URL (Airflow Variable or host environment)
    # ------------------------------------------------------------------
    load_to_postgres = BashOperator(
        task_id="load_to_postgres",
        bash_command=project_script_cmd("load_to_postgres.py", "--verbose"),
        env={
            **PIPELINE_ENV,
            # Templated at runtime — set via: airflow variables set DATABASE_URL "..."
            "DATABASE_URL": "{{ var.value.get('DATABASE_URL', '') }}",
        },
        execution_timeout=timedelta(hours=1),
    )

    # ------------------------------------------------------------------
    # Task 5: Train traffic density forecasting model
    # Script : scripts/train_forecast_model.py
    # Outputs: models/traffic_forecast_model.pkl, forecast_predictions.csv
    # ------------------------------------------------------------------
    train_forecast_model = BashOperator(
        task_id="train_forecast_model",
        bash_command=project_script_cmd("train_forecast_model.py", "--verbose"),
        env=PIPELINE_ENV,
        execution_timeout=timedelta(hours=1),
    )

    end = EmptyOperator(task_id="end")

    # Linear dependency chain — each stage depends on the previous one.
    (
        start
        >> download_batch_videos
        >> run_batch_detection
        >> compute_congestion_metrics
        >> load_to_postgres
        >> train_forecast_model
        >> end
    )
