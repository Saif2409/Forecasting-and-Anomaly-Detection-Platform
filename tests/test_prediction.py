import json

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.inference import predict as predict_module
from src.inference.predict import detect_anomalies, load_anomaly_artifacts, load_forecasting_artifacts, predict_demand


def sample_record():
    return {
        "date": "2024-01-01",
        "product_id": "P001",
        "product_category": "Electronics",
        "region": "Europe",
        "market": "Germany",
        "channel": "Online",
        "unit_price": 120.0,
        "promotion_flag": 1,
        "holiday_flag": 0,
        "marketing_spend": 1000.0,
        "inventory_level": 800.0,
        "competitor_price_index": 1.0,
        "economic_index": 101.0,
        "day_of_week": 0,
        "month": 1,
        "quarter": 1,
        "units_sold": 80,
        "revenue": 9600.0,
        "anomaly_label": 0,
        "anomaly_type": "normal",
    }


def test_model_prediction_returns_numeric_output(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_columns = ["unit_price", "promotion_flag", "holiday_flag", "marketing_spend", "inventory_level", "competitor_price_index", "economic_index", "day_of_week", "month"]
    regressor = DummyRegressor(strategy="constant", constant=100)
    regressor.fit(np.zeros((3, len(feature_columns))), [100, 100, 100])
    joblib.dump(regressor, tmp_path / "forecasting_model.joblib")
    (tmp_path / "feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    load_forecasting_artifacts.cache_clear()
    prediction = predict_demand([sample_record()])[0]
    assert isinstance(float(prediction), float)


def test_anomaly_detector_returns_expected_output_format(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "MODELS_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_columns = ["units_sold", "revenue", "inventory_level", "marketing_spend"]
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(random_state=42, contamination=0.1))])
    pipeline.fit(np.array([[80, 9600, 800, 1000], [82, 9840, 790, 950], [78, 9360, 810, 990], [400, 48000, 50, 4000]]))
    joblib.dump(pipeline, tmp_path / "anomaly_model.joblib")
    (tmp_path / "anomaly_feature_columns.json").write_text(json.dumps(feature_columns), encoding="utf-8")
    load_anomaly_artifacts.cache_clear()
    flags, scores = detect_anomalies([sample_record()])
    assert int(flags[0]) in [0, 1]
    assert isinstance(float(scores[0]), float)
