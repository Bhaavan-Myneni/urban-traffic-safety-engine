"""Serve predictions from persisted sklearn models."""

from pathlib import Path

import joblib
import pandas as pd


def load_model(model_path: Path):
    return joblib.load(model_path)


def predict_risk(model, features: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(features), index=features.index, name="risk_label")
