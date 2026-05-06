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

from src.config.settings import MONITORING_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR, ensure_directories


def compare_numeric_drift(
    reference_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    threshold: float = 0.20,
    min_std: float = 1e-6,
) -> dict[str, Any]:
    common_numeric = sorted(
        set(reference_df.select_dtypes(include="number").columns)
        & set(recent_df.select_dtypes(include="number").columns)
    )
    feature_reports = []
    for column in common_numeric:
        reference_mean = float(reference_df[column].mean())
        recent_mean = float(recent_df[column].mean())
        reference_std = float(reference_df[column].std())
        recent_std = float(recent_df[column].std())
        denominator = max(abs(reference_mean), abs(reference_std), min_std)
        relative_mean_change = abs(recent_mean - reference_mean) / denominator
        std_change = abs(recent_std - reference_std) / max(abs(reference_std), min_std)
        drifted = relative_mean_change > threshold or std_change > threshold * 2
        feature_reports.append(
            {
                "feature": column,
                "reference_mean": round(reference_mean, 5),
                "recent_mean": round(recent_mean, 5),
                "reference_std": round(reference_std, 5),
                "recent_std": round(recent_std, 5),
                "relative_mean_change": round(relative_mean_change, 5),
                "relative_std_change": round(std_change, 5),
                "drifted": bool(drifted),
            }
        )
    drifted_features = [item["feature"] for item in feature_reports if item["drifted"]]
    return {
        "threshold": threshold,
        "features_checked": len(feature_reports),
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features,
        "feature_reports": feature_reports,
    }


def generate_drift_report(
    reference_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    recent_path: Path = PREDICTIONS_DIR / "batch_predictions.csv",
    output_path: Path = MONITORING_DIR / "drift_report.json",
    threshold: float = 0.20,
) -> dict[str, Any]:
    ensure_directories()
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
    if not recent_path.exists():
        raise FileNotFoundError(f"Recent inference data not found at {recent_path}")

    reference_df = pd.read_csv(reference_path)
    recent_df = pd.read_csv(recent_path)
    report = compare_numeric_drift(reference_df, recent_df, threshold=threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Drift report saved to {output_path}")
    print(f"Drifted features: {report['drifted_features']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple statistical drift report.")
    parser.add_argument("--reference", type=Path, default=PROCESSED_DATA_DIR / "train_features.csv")
    parser.add_argument("--recent", type=Path, default=PREDICTIONS_DIR / "batch_predictions.csv")
    parser.add_argument("--output", type=Path, default=MONITORING_DIR / "drift_report.json")
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    generate_drift_report(args.reference, args.recent, args.output, args.threshold)


if __name__ == "__main__":
    main()
