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

from src.config.settings import RAW_DATA_DIR, REPORTS_DIR, ensure_directories
from src.data.data_loader import load_sales_data
from src.data.generate_synthetic_data import ANOMALY_TYPES, CHANNELS, PRODUCT_CATEGORIES, REGION_MARKETS, REQUIRED_COLUMNS

EXPECTED_COLUMNS = REQUIRED_COLUMNS
VALID_CATEGORIES = set(PRODUCT_CATEGORIES)
VALID_REGIONS = set(REGION_MARKETS)
VALID_MARKETS = {market for markets in REGION_MARKETS.values() for market in markets}
VALID_CHANNELS = set(CHANNELS)
NON_NEGATIVE_COLUMNS = ["unit_price", "marketing_spend", "inventory_level", "units_sold", "revenue"]


def validate_sales_data(df: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"passed": True, "checks": {}}
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    report["checks"]["missing_columns"] = missing_columns

    if missing_columns:
        report["passed"] = False
        return report

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    missing_values = df[EXPECTED_COLUMNS].isna().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    negative_values = {col: int((df[col] < 0).sum()) for col in NON_NEGATIVE_COLUMNS}
    invalid_categories = sorted(set(df["product_category"].dropna()) - VALID_CATEGORIES)
    invalid_regions = sorted(set(df["region"].dropna()) - VALID_REGIONS)
    invalid_markets = sorted(set(df["market"].dropna()) - VALID_MARKETS)
    invalid_region_market_rows = int(df.apply(lambda row: row["market"] not in REGION_MARKETS.get(row["region"], []), axis=1).sum())
    invalid_channels = sorted(set(df["channel"].dropna()) - VALID_CHANNELS)
    invalid_anomaly_types = sorted(set(df["anomaly_type"].dropna()) - ANOMALY_TYPES)
    normal_type_mismatches = int((df.loc[df["anomaly_label"] == 0, "anomaly_type"] != "normal").sum())
    anomaly_type_mismatches = int((df.loc[df["anomaly_label"] == 1, "anomaly_type"] == "normal").sum())
    invalid_flags = {
        "promotion_flag": sorted(set(df["promotion_flag"].dropna()) - {0, 1}),
        "holiday_flag": sorted(set(df["holiday_flag"].dropna()) - {0, 1}),
        "anomaly_label": sorted(set(df["anomaly_label"].dropna()) - {0, 1}),
    }
    anomaly_distribution = df["anomaly_label"].value_counts(normalize=True).round(4).to_dict()
    anomaly_type_counts = df["anomaly_type"].value_counts().to_dict()
    target_summary = df["units_sold"].describe().round(3).to_dict()
    date_range = {
        "min": None if df["date"].isna().all() else str(df["date"].min().date()),
        "max": None if df["date"].isna().all() else str(df["date"].max().date()),
        "invalid_dates": int(df["date"].isna().sum()),
    }
    revenue_consistency_error_rate = float((abs(df["revenue"] - (df["units_sold"] * df["unit_price"])) > 1.5).mean())

    report["checks"] = {
        "row_count": int(len(df)),
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "negative_values": negative_values,
        "invalid_categories": invalid_categories,
        "invalid_regions": invalid_regions,
        "invalid_markets": invalid_markets,
        "invalid_region_market_rows": invalid_region_market_rows,
        "invalid_channels": invalid_channels,
        "invalid_anomaly_types": invalid_anomaly_types,
        "normal_type_mismatches": normal_type_mismatches,
        "anomaly_type_mismatches": anomaly_type_mismatches,
        "invalid_flags": invalid_flags,
        "date_range": date_range,
        "anomaly_distribution": anomaly_distribution,
        "anomaly_type_counts": anomaly_type_counts,
        "target_summary": target_summary,
        "revenue_consistency_error_rate": round(revenue_consistency_error_rate, 5),
    }

    failure_conditions = [
        any(value > 0 for value in missing_values.values()),
        duplicate_rows > 0,
        any(value > 0 for value in negative_values.values()),
        bool(invalid_categories),
        bool(invalid_regions),
        bool(invalid_markets),
        invalid_region_market_rows > 0,
        bool(invalid_channels),
        bool(invalid_anomaly_types),
        normal_type_mismatches > 0,
        anomaly_type_mismatches > 0,
        any(bool(values) for values in invalid_flags.values()),
        date_range["invalid_dates"] > 0,
        not df["quarter"].between(1, 4).all(),
        df["units_sold"].max() <= 0,
    ]
    report["passed"] = not any(failure_conditions)
    return report


def print_validation_report(report: dict[str, Any]) -> None:
    print("Data validation report")
    print(f"Status: {'PASSED' if report['passed'] else 'FAILED'}")
    print(json.dumps(report["checks"], indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate enterprise sales data.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "sales_data.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "data_validation_report.json")
    args = parser.parse_args()

    ensure_directories()
    df = load_sales_data(args.input)
    report = validate_sales_data(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print_validation_report(report)


if __name__ == "__main__":
    main()
