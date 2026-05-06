from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import PREDICTIONS_DIR, RAW_DATA_DIR, ensure_directories
from src.data.data_loader import load_sales_data
from src.inference.predict import detect_anomalies, predict_demand


def run_batch_inference(input_path: Path = RAW_DATA_DIR / "sales_data.csv", output_path: Path = PREDICTIONS_DIR / "batch_predictions.csv") -> pd.DataFrame:
    ensure_directories()
    df = load_sales_data(input_path)
    predictions = predict_demand(df)
    anomaly_flags, anomaly_scores = detect_anomalies(df)

    output = pd.DataFrame(
        {
            "date": df["date"],
            "product_id": df["product_id"],
            "product_category": df["product_category"],
            "region": df["region"],
            "market": df["market"],
            "channel": df["channel"],
            "unit_price": df["unit_price"],
            "promotion_flag": df["promotion_flag"],
            "holiday_flag": df["holiday_flag"],
            "marketing_spend": df["marketing_spend"],
            "inventory_level": df["inventory_level"],
            "competitor_price_index": df["competitor_price_index"],
            "economic_index": df["economic_index"],
            "day_of_week": df["day_of_week"],
            "month": df["month"],
            "quarter": df["quarter"],
            "actual_units_sold": df["units_sold"] if "units_sold" in df.columns else np.nan,
            "predicted_units_sold": predictions.round(2),
            "prediction_error": (df["units_sold"] - predictions).round(2) if "units_sold" in df.columns else np.nan,
            "anomaly_score": np.asarray(anomaly_scores).round(6),
            "anomaly_flag": anomaly_flags.astype(int),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Batch inference complete: {output_path}")
    print(f"Rows scored: {len(output):,}")
    print(f"Predicted anomaly rate: {output['anomaly_flag'].mean() * 100:.2f}%")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch demand forecasting and anomaly detection.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "sales_data.csv")
    parser.add_argument("--output", type=Path, default=PREDICTIONS_DIR / "batch_predictions.csv")
    args = parser.parse_args()
    run_batch_inference(args.input, args.output)


if __name__ == "__main__":
    main()
