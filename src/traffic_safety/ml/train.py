"""Train and evaluate safety risk classification models."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def train_risk_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_path: Path | None = None,
) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    _ = classification_report(y_test, model.predict(X_test))
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
    return model
