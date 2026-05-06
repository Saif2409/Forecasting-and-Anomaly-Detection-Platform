from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import MONITORING_DIR, PREDICTIONS_DIR, ensure_directories
from src.monitoring.drift_detection import generate_drift_report


def summarize_predictions(predictions_path: Path) -> dict[str, Any]:
    if not predictions_path.exists():
        return {"status": "predictions_not_found"}
    df = pd.read_csv(predictions_path)
    summary: dict[str, Any] = {
        "prediction_volume": int(len(df)),
        "average_prediction": float(df["predicted_units_sold"].mean()) if "predicted_units_sold" in df else None,
        "anomaly_rate": float(df["anomaly_flag"].mean()) if "anomaly_flag" in df else None,
    }
    if "prediction_error" in df and df["prediction_error"].notna().any():
        summary["mean_absolute_error"] = float(df["prediction_error"].abs().mean())
        summary["mean_prediction_error"] = float(df["prediction_error"].mean())
    return summary


def create_monitoring_report(
    predictions_path: Path = PREDICTIONS_DIR / "batch_predictions.csv",
    output_path: Path = MONITORING_DIR / "monitoring_summary.json",
) -> dict[str, Any]:
    ensure_directories()
    prediction_summary = summarize_predictions(predictions_path)
    drift_report = {"status": "not_generated"}
    try:
        drift_report = generate_drift_report(recent_path=predictions_path)
    except FileNotFoundError as exc:
        drift_report = {"status": "skipped", "reason": str(exc)}

    report = {
        "prediction_summary": prediction_summary,
        "drift_summary": {
            "status": drift_report.get("status", "generated"),
            "features_checked": drift_report.get("features_checked", 0),
            "drifted_feature_count": drift_report.get("drifted_feature_count", 0),
            "drifted_features": drift_report.get("drifted_features", []),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Monitoring report saved to {output_path}")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate monitoring summary report.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_DIR / "batch_predictions.csv")
    parser.add_argument("--output", type=Path, default=MONITORING_DIR / "monitoring_summary.json")
    args = parser.parse_args()
    create_monitoring_report(args.predictions, args.output)


if __name__ == "__main__":
    main()
