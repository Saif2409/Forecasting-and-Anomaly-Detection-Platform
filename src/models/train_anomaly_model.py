from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories, settings
from src.features.build_features import LABEL_COLUMN, TARGET_COLUMN, build_feature_datasets
from src.data.data_loader import load_sales_data
from src.models.evaluate import classification_metrics, save_metrics
from src.models.model_registry import write_model_metadata

PREFERRED_ANOMALY_FEATURES = [
    "units_sold",
    "revenue",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "rolling_std_14",
    "price_ratio_vs_competitor",
    "inventory_level",
    "marketing_spend",
    "competitor_price_index",
    "economic_index",
    "promotion_flag",
    "holiday_flag",
]


def load_or_build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DATA_DIR / "train_features.csv"
    validation_path = PROCESSED_DATA_DIR / "validation_features.csv"
    test_path = PROCESSED_DATA_DIR / "test_features.csv"
    if all(path.exists() for path in [train_path, validation_path, test_path]):
        return pd.read_csv(train_path), pd.read_csv(validation_path), pd.read_csv(test_path)
    raw_df = load_sales_data()
    train_df, validation_df, test_df, _ = build_feature_datasets(raw_df)
    return train_df, validation_df, test_df


def choose_anomaly_features(df: pd.DataFrame) -> list[str]:
    return [col for col in PREFERRED_ANOMALY_FEATURES if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


def convert_isolation_predictions(predictions: np.ndarray) -> np.ndarray:
    return np.where(predictions == -1, 1, 0)


def train_anomaly_model() -> dict[str, object]:
    ensure_directories()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("enterprise_anomaly_detection")

    train_df, validation_df, test_df = load_or_build_features()
    anomaly_features = choose_anomaly_features(train_df)
    normal_train = train_df[train_df[LABEL_COLUMN] == 0]
    contamination = float(max(0.01, min(0.08, train_df[LABEL_COLUMN].mean())))
    params = {
        "n_estimators": 250,
        "contamination": contamination,
        "random_state": settings.default_random_seed,
        "n_jobs": -1,
    }
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", IsolationForest(**params)),
        ]
    )

    with mlflow.start_run(run_name="isolation_forest_kpi_anomaly_detection"):
        pipeline.fit(normal_train[anomaly_features])
        validation_pred = convert_isolation_predictions(pipeline.predict(validation_df[anomaly_features]))
        test_pred = convert_isolation_predictions(pipeline.predict(test_df[anomaly_features]))
        validation_metrics = classification_metrics(validation_df[LABEL_COLUMN].to_numpy(), validation_pred)
        test_metrics = classification_metrics(test_df[LABEL_COLUMN].to_numpy(), test_pred)
        metrics = {"validation": validation_metrics, "test": test_metrics}

        mlflow.log_params(params)
        mlflow.log_param("feature_count", len(anomaly_features))
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(f"{split_name}_{metric_name}", value)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "anomaly_model.joblib"
        feature_path = MODELS_DIR / "anomaly_feature_columns.json"
        joblib.dump(pipeline, model_path)
        feature_path.write_text(json.dumps(anomaly_features, indent=2), encoding="utf-8")
        metrics_path = save_metrics(metrics, REPORTS_DIR / "anomaly_metrics.json")
        mlflow.sklearn.log_model(pipeline, artifact_path="anomaly_model")
        mlflow.log_artifact(str(feature_path))
        mlflow.log_artifact(str(metrics_path))
        metadata_path = write_model_metadata("anomaly_model", metrics, model_path, {"feature_count": len(anomaly_features)})
        mlflow.log_artifact(str(metadata_path))

    print("Anomaly detection training complete")
    print(json.dumps(metrics, indent=2))
    return {"model_path": model_path, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest anomaly detection model.")
    parser.parse_args()
    train_anomaly_model()


if __name__ == "__main__":
    main()
