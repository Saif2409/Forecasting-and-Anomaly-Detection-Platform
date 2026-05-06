from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR
from src.features.build_features import build_feature_datasets, prepare_feature_frame


class ModelNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_forecasting_artifacts() -> tuple[Any, list[str]]:
    model_path = MODELS_DIR / "forecasting_model.joblib"
    feature_path = MODELS_DIR / "feature_columns.json"
    if not model_path.exists() or not feature_path.exists():
        raise ModelNotFoundError("Forecasting model artifacts not found. Run training first.")
    model = joblib.load(model_path)
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, feature_columns


@lru_cache(maxsize=1)
def load_anomaly_artifacts() -> tuple[Any, list[str]]:
    model_path = MODELS_DIR / "anomaly_model.joblib"
    feature_path = MODELS_DIR / "anomaly_feature_columns.json"
    if not model_path.exists() or not feature_path.exists():
        raise ModelNotFoundError("Anomaly model artifacts not found. Run anomaly training first.")
    model = joblib.load(model_path)
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, feature_columns


def prepare_inference_features(records: list[dict[str, Any]] | pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records).copy()
    if "units_sold" not in df.columns:
        df["units_sold"] = 0
    if "revenue" not in df.columns:
        df["revenue"] = df.get("unit_price", 0) * df["units_sold"]
    if "anomaly_label" not in df.columns:
        df["anomaly_label"] = 0
    if "anomaly_type" not in df.columns:
        df["anomaly_type"] = "normal"
    if "quarter" not in df.columns:
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    featured = prepare_feature_frame(df)
    encoded = pd.get_dummies(featured, columns=["product_id", "product_category", "region", "market", "channel"], drop_first=False)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def predict_demand(records: list[dict[str, Any]] | pd.DataFrame) -> np.ndarray:
    model, feature_columns = load_forecasting_artifacts()
    X = prepare_inference_features(records, feature_columns)
    predictions = model.predict(X)
    return np.maximum(predictions, 0)


def detect_anomalies(records: list[dict[str, Any]] | pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model, feature_columns = load_anomaly_artifacts()
    X = prepare_inference_features(records, feature_columns)
    raw_predictions = model.predict(X)
    anomaly_flags = np.where(raw_predictions == -1, 1, 0)
    if hasattr(model, "decision_function"):
        scores = -model.decision_function(X)
    else:
        scores = anomaly_flags.astype(float)
    return anomaly_flags, scores


def model_status() -> dict[str, bool]:
    return {
        "forecasting_model_loaded": (MODELS_DIR / "forecasting_model.joblib").exists(),
        "forecasting_features_loaded": (MODELS_DIR / "feature_columns.json").exists(),
        "anomaly_model_loaded": (MODELS_DIR / "anomaly_model.joblib").exists(),
        "anomaly_features_loaded": (MODELS_DIR / "anomaly_feature_columns.json").exists(),
    }
