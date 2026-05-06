from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories
from src.data.data_loader import load_sales_data, save_processed_data

IDENTIFIER_COLUMNS = ["date", "product_id", "product_category", "region", "market", "channel", "anomaly_type"]
TARGET_COLUMN = "units_sold"
LABEL_COLUMN = "anomaly_label"
CATEGORICAL_COLUMNS = ["product_id", "product_category", "region", "market", "channel"]
BASE_NUMERIC_FEATURES = [
    "unit_price",
    "promotion_flag",
    "holiday_flag",
    "marketing_spend",
    "inventory_level",
    "competitor_price_index",
    "economic_index",
    "day_of_week",
    "month",
]


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["product_id", "region", "market", "channel", "date"]).reset_index(drop=True)
    group_cols = ["product_id", "region", "market", "channel"]
    grouped_units = df.groupby(group_cols)[TARGET_COLUMN]
    grouped_revenue = df.groupby(group_cols)["revenue"]

    for lag in [1, 7, 14, 28]:
        df[f"units_sold_lag_{lag}"] = grouped_units.shift(lag)
        df[f"revenue_lag_{lag}"] = grouped_revenue.shift(lag)

    shifted_units = grouped_units.shift(1)
    df["rolling_mean_7"] = shifted_units.groupby([df[col] for col in group_cols]).rolling(7, min_periods=1).mean().reset_index(level=group_cols, drop=True)
    df["rolling_mean_14"] = shifted_units.groupby([df[col] for col in group_cols]).rolling(14, min_periods=1).mean().reset_index(level=group_cols, drop=True)
    df["rolling_std_7"] = shifted_units.groupby([df[col] for col in group_cols]).rolling(7, min_periods=2).std().reset_index(level=group_cols, drop=True)
    df["rolling_std_14"] = shifted_units.groupby([df[col] for col in group_cols]).rolling(14, min_periods=2).std().reset_index(level=group_cols, drop=True)

    df["revenue_per_unit"] = df["revenue_lag_1"] / df["units_sold_lag_1"].replace(0, np.nan)
    df["price_ratio_vs_competitor"] = df["unit_price"] / df["competitor_price_index"].replace(0, np.nan)
    df["inventory_to_lag_demand_ratio"] = df["inventory_level"] / df["units_sold_lag_7"].replace(0, np.nan)
    df["promotion_holiday_interaction"] = df["promotion_flag"] * df["holiday_flag"]
    df["marketing_promotion_interaction"] = df["marketing_spend"] * df["promotion_flag"]
    df["quarter"] = df["date"].dt.quarter
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["time_index"] = (df["date"] - df["date"].min()).dt.days
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_time_series_features(df)
    featured = featured.sort_values("date").reset_index(drop=True)
    numeric_columns = featured.select_dtypes(include=["number", "bool"]).columns.tolist()
    fill_columns = [col for col in numeric_columns if col != LABEL_COLUMN]
    featured[fill_columns] = featured[fill_columns].replace([np.inf, -np.inf], np.nan)
    featured[fill_columns] = featured[fill_columns].fillna(featured[fill_columns].median(numeric_only=True))
    featured[fill_columns] = featured[fill_columns].fillna(0)
    return featured


def chronological_split(df: pd.DataFrame, train_size: float = 0.70, validation_size: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    train_end = int(len(df) * train_size)
    validation_end = int(len(df) * (train_size + validation_size))
    return df.iloc[:train_end].copy(), df.iloc[train_end:validation_end].copy(), df.iloc[validation_end:].copy()


def encode_categoricals(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_columns: Iterable[str] = CATEGORICAL_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(train_df, columns=list(categorical_columns), drop_first=False)
    validation_encoded = pd.get_dummies(validation_df, columns=list(categorical_columns), drop_first=False)
    test_encoded = pd.get_dummies(test_df, columns=list(categorical_columns), drop_first=False)
    validation_encoded = validation_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    return train_encoded, validation_encoded, test_encoded


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(IDENTIFIER_COLUMNS + [TARGET_COLUMN, "revenue", LABEL_COLUMN])
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def build_feature_datasets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    featured = prepare_feature_frame(df)
    train_df, validation_df, test_df = chronological_split(featured)
    train_encoded, validation_encoded, test_encoded = encode_categoricals(train_df, validation_df, test_df)
    feature_columns = get_feature_columns(train_encoded)
    return train_encoded, validation_encoded, test_encoded, feature_columns


def save_feature_outputs(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path = PROCESSED_DATA_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_processed_data(train_df, output_dir / "train_features.csv")
    save_processed_data(validation_df, output_dir / "validation_features.csv")
    save_processed_data(test_df, output_dir / "test_features.csv")
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe forecasting and anomaly features.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "sales_data.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    ensure_directories()
    df = load_sales_data(args.input)
    train_df, validation_df, test_df, feature_columns = build_feature_datasets(df)
    save_feature_outputs(train_df, validation_df, test_df, feature_columns, args.output_dir)
    print("Feature engineering complete")
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(validation_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Feature columns: {len(feature_columns)}")


if __name__ == "__main__":
    main()
