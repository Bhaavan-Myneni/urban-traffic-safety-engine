"""Train and evaluate traffic density forecasting models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "congestion_features_master.csv"
DEFAULT_METRICS_SUPPLEMENT = PROJECT_ROOT / "data" / "processed" / "congestion_metrics_master.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "traffic_forecast_model.pkl"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_predictions.csv"

MIN_RECOMMENDED_ROWS = 100
RANDOM_STATE = 42
TARGET_COLUMN = "traffic_density"

CATEGORICAL_SOURCE_COLUMNS = [
    "camera_id",
    "video_id",
    "scene_type",
    "road_type",
    "weather_condition",
    "event_type",
]

NUMERIC_FEATURE_COLUMNS = [
    "minute",
    "vehicle_count",
    "pedestrian_count",
    "avg_vehicles_per_frame",
    "avg_pedestrians_per_frame",
    "temperature_f",
    "precipitation_flag",
    "visibility_miles",
    "event_nearby_flag",
    "estimated_event_attendance",
    "distance_to_event_miles",
    "lag_1_density",
    "rolling_mean_3_density",
]

REQUIRED_COLUMNS = [
    *CATEGORICAL_SOURCE_COLUMNS,
    "minute",
    "vehicle_count",
    "pedestrian_count",
    "traffic_density",
    "temperature_f",
    "precipitation_flag",
    "visibility_miles",
    "event_nearby_flag",
    "estimated_event_attendance",
    "distance_to_event_miles",
]

SUPPLEMENT_COLUMNS = ["avg_vehicles_per_frame", "avg_pedestrians_per_frame"]


class ForecastTrainingError(Exception):
    """Raised when forecast training cannot proceed."""


@dataclass(frozen=True)
class EvaluationMetrics:
    """Regression evaluation metrics for a single model."""

    model_name: str
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class ForecastTrainingResult:
    """Artifacts and metrics produced by the training pipeline."""

    best_model_name: str
    metrics: list[EvaluationMetrics]
    feature_importance: pd.DataFrame
    model_path: Path
    predictions_path: Path
    rows_used: int


@dataclass
class ForecastModelBundle:
    """Persisted bundle for inference."""

    model: RegressorMixin
    model_name: str
    encoders: dict[str, LabelEncoder]
    feature_columns: list[str]
    metrics: list[EvaluationMetrics]
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


def encoded_column_name(source_column: str) -> str:
    """Return the encoded feature column name for a categorical source column."""
    return f"{source_column}_encoded"


def build_feature_columns() -> list[str]:
    """Return the full ordered list of model feature columns."""
    encoded = [encoded_column_name(col) for col in CATEGORICAL_SOURCE_COLUMNS]
    return encoded + NUMERIC_FEATURE_COLUMNS


FEATURE_COLUMNS = build_feature_columns()


def load_feature_data(path: Path) -> pd.DataFrame:
    """Load and validate the congestion features CSV."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise ForecastTrainingError(
            f"Input file not found: {resolved}\n"
            "Run: python scripts/generate_external_features.py"
        )

    try:
        df = pd.read_csv(resolved)
    except pd.errors.EmptyDataError as exc:
        raise ForecastTrainingError(f"Input file is empty: {resolved}") from exc
    except OSError as exc:
        raise ForecastTrainingError(f"Failed to read {resolved}: {exc}") from exc

    if df.empty:
        raise ForecastTrainingError(f"Input file contains no rows: {resolved}")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ForecastTrainingError(f"Missing required columns: {sorted(missing)}")

    logger.info("Loaded %d feature row(s) from %s", len(df), resolved.name)
    return df


def enrich_with_per_frame_averages(
    features_df: pd.DataFrame,
    metrics_path: Path = DEFAULT_METRICS_SUPPLEMENT,
) -> pd.DataFrame:
    """Join avg_vehicles_per_frame and avg_pedestrians_per_frame when absent."""
    if all(col in features_df.columns for col in SUPPLEMENT_COLUMNS):
        return features_df

    resolved = metrics_path.expanduser().resolve()
    if not resolved.exists():
        raise ForecastTrainingError(
            f"Cannot derive per-frame averages — missing supplement file: {resolved}\n"
            "Run: python scripts/compute_metrics.py --master"
        )

    metrics = pd.read_csv(resolved)
    join_cols = ["camera_id", "video_id", "minute"]
    supplement = metrics[join_cols + SUPPLEMENT_COLUMNS].drop_duplicates()

    enriched = features_df.merge(supplement, on=join_cols, how="left")
    if enriched[SUPPLEMENT_COLUMNS].isna().any().any():
        raise ForecastTrainingError(
            "Failed to join avg_vehicles_per_frame / avg_pedestrians_per_frame "
            "from congestion metrics."
        )

    logger.info("Joined per-frame average columns from %s", resolved.name)
    return enriched


def warn_if_small_dataset(row_count: int) -> None:
    """Log a prototype warning when the dataset is too small for robust forecasting."""
    if row_count < MIN_RECOMMENDED_ROWS:
        logger.warning(
            "Current dataset has only %d rows, so this is a prototype forecasting pipeline. "
            "It is structured to scale to real hourly camera-feed data.",
            row_count,
        )


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add camera-level lag and rolling density features."""
    enriched = df.sort_values(["camera_id", "video_id", "minute"]).copy()

    enriched["lag_1_density"] = enriched.groupby("camera_id")["traffic_density"].shift(1)
    enriched["rolling_mean_3_density"] = enriched.groupby("camera_id")[
        "traffic_density"
    ].transform(lambda series: series.rolling(window=3, min_periods=1).mean())

    enriched["lag_1_density"] = enriched["lag_1_density"].fillna(0.0)
    return enriched


def encode_categorical_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode all categorical model inputs."""
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for column in CATEGORICAL_SOURCE_COLUMNS:
        encoder = LabelEncoder()
        encoded[encoded_column_name(column)] = encoder.fit_transform(
            encoded[column].astype(str)
        )
        encoders[column] = encoder

    return encoded, encoders


def engineer_features(
    df: pd.DataFrame,
    metrics_path: Path = DEFAULT_METRICS_SUPPLEMENT,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Build model-ready features from congestion feature records."""
    enriched = enrich_with_per_frame_averages(df, metrics_path=metrics_path)
    with_temporal = add_temporal_features(enriched)
    encoded, encoders = encode_categorical_features(with_temporal)
    return encoded, encoders


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y."""
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(float)
    return X, y


def evaluate_regressor(
    model: RegressorMixin,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationMetrics:
    """Compute MAE, RMSE, and R² on a held-out test set."""
    predictions = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))
    return EvaluationMetrics(model_name=model_name, mae=mae, rmse=rmse, r2=r2)


def build_models() -> dict[str, RegressorMixin]:
    """Return candidate regressors for comparison."""
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, RegressorMixin], list[EvaluationMetrics]]:
    """Train all candidate models and evaluate on the test split."""
    fitted_models: dict[str, RegressorMixin] = {}
    metrics: list[EvaluationMetrics] = []

    for model_name, model in build_models().items():
        logger.info("Training %s...", model_name)
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        result = evaluate_regressor(model, model_name, X_test, y_test)
        metrics.append(result)
        logger.info(
            "%s — MAE: %.4f | RMSE: %.4f | R²: %.4f",
            model_name,
            result.mae,
            result.rmse,
            result.r2,
        )

    return fitted_models, metrics


def select_best_model_name(metrics: list[EvaluationMetrics]) -> str:
    """Pick the model with the lowest RMSE (tie-break on higher R²)."""
    return min(metrics, key=lambda item: (item.rmse, -item.r2)).model_name


def extract_feature_importance(
    model: RegressorMixin,
    model_name: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Extract and rank feature importances from a tree-based regressor."""
    if not hasattr(model, "feature_importances_"):
        raise ForecastTrainingError(
            f"Model {model_name} does not expose feature_importances_."
        )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)

    importance["importance"] = importance["importance"].round(6)
    importance["rank"] = np.arange(1, len(importance) + 1)
    return importance[["rank", "feature", "importance"]]


def save_model_bundle(bundle: ForecastModelBundle, path: Path) -> None:
    """Persist the best model, encoders, and metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("Saved forecast model bundle to %s", path)


def build_predictions_frame(
    df: pd.DataFrame,
    predictions: np.ndarray,
    model_name: str,
    split: pd.Series | None = None,
) -> pd.DataFrame:
    """Create a prediction output dataframe with residuals."""
    output = df[
        [
            "camera_id",
            "video_id",
            "scene_type",
            "minute",
            "vehicle_count",
            "pedestrian_count",
            "traffic_density",
            "weather_condition",
            "event_type",
        ]
    ].copy()
    output["predicted_traffic_density"] = np.round(predictions, 4)
    output["prediction_error"] = np.round(
        output["traffic_density"] - output["predicted_traffic_density"], 4
    )
    output["model_name"] = model_name
    if split is not None:
        output["dataset_split"] = split.values
    return output


def save_predictions(df: pd.DataFrame, path: Path) -> None:
    """Write forecast predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved forecast predictions to %s", path)


def print_model_comparison(metrics: list[EvaluationMetrics], best_model_name: str) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n=== Traffic Density Forecast — Model Comparison ===")
    print(f"{'Model':<28} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 62)
    for item in sorted(metrics, key=lambda m: m.rmse):
        marker = " *" if item.model_name == best_model_name else ""
        print(
            f"{item.model_name:<28} {item.mae:10.4f} {item.rmse:10.4f} {item.r2:10.4f}{marker}"
        )
    print("-" * 62)
    print(f"Best model (lowest RMSE): {best_model_name}\n")


def print_feature_importance(importance_df: pd.DataFrame, model_name: str, top_n: int = 10) -> None:
    """Print ranked feature importances for the best model."""
    print(f"=== Feature Importance — {model_name} (Top {top_n}) ===")
    print(f"{'Rank':<6} {'Feature':<35} {'Importance':>12}")
    print("-" * 55)
    for row in importance_df.head(top_n).itertuples(index=False):
        print(f"{row.rank:<6} {row.feature:<35} {row.importance:12.6f}")
    print("-" * 55 + "\n")


def run_forecast_training_pipeline(
    input_path: Path = DEFAULT_INPUT,
    model_path: Path = DEFAULT_MODEL_PATH,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    metrics_supplement_path: Path = DEFAULT_METRICS_SUPPLEMENT,
    test_size: float = 0.2,
) -> ForecastTrainingResult:
    """
    End-to-end prototype forecasting pipeline with external weather/event features.

    Steps:
        1. Load congestion feature records
        2. Engineer encodings, lag, and rolling density features
        3. Train RandomForest and GradientBoosting regressors
        4. Evaluate on a hold-out split
        5. Retrain the best model on all data
        6. Persist model bundle, predictions, and feature importance
    """
    raw_df = load_feature_data(input_path)
    warn_if_small_dataset(len(raw_df))

    featured_df, encoders = engineer_features(raw_df, metrics_path=metrics_supplement_path)
    X, y = build_feature_matrix(featured_df)

    if len(featured_df) < 5:
        raise ForecastTrainingError(
            f"Need at least 5 rows to train/test split; got {len(featured_df)}."
        )

    X_train, X_test, y_train, y_test, _train_idx, test_idx = train_test_split(
        X,
        y,
        featured_df.index,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    _fitted_models, metrics = train_and_compare_models(X_train, y_train, X_test, y_test)
    best_model_name = select_best_model_name(metrics)
    print_model_comparison(metrics, best_model_name)

    logger.info("Retraining best model (%s) on full dataset.", best_model_name)
    best_model = build_models()[best_model_name]
    best_model.fit(X, y)

    feature_importance = extract_feature_importance(best_model, best_model_name, FEATURE_COLUMNS)
    print_feature_importance(feature_importance, best_model_name)

    split_labels = pd.Series("train", index=featured_df.index)
    split_labels.loc[test_idx] = "test"

    full_predictions = best_model.predict(X)
    predictions_df = build_predictions_frame(
        featured_df,
        full_predictions,
        best_model_name,
        split=split_labels,
    )
    save_predictions(predictions_df, predictions_path)

    bundle = ForecastModelBundle(
        model=best_model,
        model_name=best_model_name,
        encoders=encoders,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        feature_importance=feature_importance,
    )
    save_model_bundle(bundle, model_path)

    return ForecastTrainingResult(
        best_model_name=best_model_name,
        metrics=metrics,
        feature_importance=feature_importance,
        model_path=model_path.resolve(),
        predictions_path=predictions_path.resolve(),
        rows_used=len(featured_df),
    )
