from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, ensure_directories, settings
from src.features.build_features import TARGET_COLUMN, build_feature_datasets
from src.data.data_loader import load_sales_data
from src.models.evaluate import regression_metrics, save_metrics
from src.models.model_registry import write_model_metadata


def load_or_build_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_path = PROCESSED_DATA_DIR / "train_features.csv"
    validation_path = PROCESSED_DATA_DIR / "validation_features.csv"
    test_path = PROCESSED_DATA_DIR / "test_features.csv"
    feature_columns_path = PROCESSED_DATA_DIR / "feature_columns.json"

    if all(path.exists() for path in [train_path, validation_path, test_path, feature_columns_path]):
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)
        feature_columns = json.loads(feature_columns_path.read_text(encoding="utf-8"))
        return train_df, validation_df, test_df, feature_columns

    raw_df = load_sales_data()
    return build_feature_datasets(raw_df)


def baseline_predictions(df: pd.DataFrame) -> pd.Series:
    if "units_sold_lag_1" in df.columns:
        return df["units_sold_lag_1"].fillna(df[TARGET_COLUMN].median())
    return pd.Series([df[TARGET_COLUMN].median()] * len(df), index=df.index)


def train_forecasting_model() -> dict[str, object]:
    ensure_directories()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("enterprise_forecasting")

    train_df, validation_df, test_df, feature_columns = load_or_build_features()
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    X_validation = validation_df[feature_columns]
    y_validation = validation_df[TARGET_COLUMN]
    X_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    params = {
        "objective": "regression",
        "n_estimators": 450,
        "learning_rate": 0.04,
        "num_leaves": 48,
        "max_depth": -1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "random_state": settings.default_random_seed,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    with mlflow.start_run(run_name="lightgbm_demand_forecasting"):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_validation, y_validation)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
        )
        validation_pred = model.predict(X_validation)
        test_pred = model.predict(X_test)
        baseline_validation_pred = baseline_predictions(validation_df)
        baseline_test_pred = baseline_predictions(test_df)

        metrics = {
            "validation_model": regression_metrics(y_validation.to_numpy(), validation_pred),
            "test_model": regression_metrics(y_test.to_numpy(), test_pred),
            "validation_baseline": regression_metrics(y_validation.to_numpy(), baseline_validation_pred.to_numpy()),
            "test_baseline": regression_metrics(y_test.to_numpy(), baseline_test_pred.to_numpy()),
        }

        mlflow.log_params(params)
        mlflow.log_param("feature_count", len(feature_columns))
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", value)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / "forecasting_model.joblib"
        feature_columns_path = MODELS_DIR / "feature_columns.json"
        joblib.dump(model, model_path)
        feature_columns_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
        metrics_path = save_metrics(metrics, REPORTS_DIR / "forecasting_metrics.json")
        metadata_path = write_model_metadata("forecasting_model", metrics, model_path, {"feature_count": len(feature_columns)})
        try:
            mlflow.lightgbm.log_model(model, artifact_path="forecasting_model")
            mlflow.log_artifact(str(feature_columns_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(metadata_path))
        except Exception as exc:
            print(f"MLflow artifact logging skipped: {exc}")

    print("Forecasting training complete")
    print(json.dumps(metrics, indent=2))
    return {"model_path": model_path, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM demand forecasting model.")
    parser.parse_args()
    train_forecasting_model()


if __name__ == "__main__":
    main()
