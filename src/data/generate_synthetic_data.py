from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import RAW_DATA_DIR, ensure_directories, settings
from src.data.database import write_dataframe_to_postgres

PRODUCT_CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home", "Beauty", "Automotive", "Office Supplies"]
REGION_MARKETS = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe": ["Germany", "France", "United Kingdom", "Netherlands"],
    "Middle East": ["UAE", "Saudi Arabia", "Qatar", "Kuwait"],
    "Asia Pacific": ["China", "India", "Japan", "Singapore", "Australia"],
    "Latin America": ["Brazil", "Argentina", "Chile"],
    "Africa": ["South Africa", "Egypt", "Kenya"],
}
REGIONS = list(REGION_MARKETS)
CHANNELS = ["Online", "Retail", "Wholesale", "Partner"]
ANOMALY_TYPES = {"normal", "demand_spike", "demand_drop", "revenue_anomaly", "inventory_stockout"}
REQUIRED_COLUMNS = [
    "date",
    "product_id",
    "product_category",
    "region",
    "market",
    "channel",
    "unit_price",
    "promotion_flag",
    "holiday_flag",
    "marketing_spend",
    "inventory_level",
    "competitor_price_index",
    "economic_index",
    "day_of_week",
    "month",
    "quarter",
    "units_sold",
    "revenue",
    "anomaly_label",
    "anomaly_type",
]


def _calculate_rows_per_combination(rows: int, n_products: int, n_regions: int, n_channels: int) -> int:
    combinations = n_products * n_regions * n_channels
    return max(1, int(np.ceil(rows / combinations)))


def _choose_market(region: str, rng: np.random.Generator) -> str:
    market_weights = {
        "United States": 0.50,
        "Canada": 0.25,
        "Mexico": 0.25,
        "Germany": 0.30,
        "France": 0.25,
        "United Kingdom": 0.30,
        "Netherlands": 0.15,
        "UAE": 0.30,
        "Saudi Arabia": 0.35,
        "Qatar": 0.18,
        "Kuwait": 0.17,
        "China": 0.28,
        "India": 0.28,
        "Japan": 0.22,
        "Singapore": 0.10,
        "Australia": 0.12,
        "Brazil": 0.45,
        "Argentina": 0.25,
        "Chile": 0.30,
        "South Africa": 0.45,
        "Egypt": 0.35,
        "Kenya": 0.20,
    }
    markets = REGION_MARKETS[region]
    weights = np.asarray([market_weights[market] for market in markets], dtype=float)
    weights = weights / weights.sum()
    return str(rng.choice(markets, p=weights))


def generate_sales_data(
    rows: int = 50_000,
    start_date: str = "2022-01-01",
    n_products: int = 40,
    random_seed: int = settings.default_random_seed,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    products = [f"P{idx:03d}" for idx in range(1, n_products + 1)]
    category_map = {product: rng.choice(PRODUCT_CATEGORIES) for product in products}
    base_product_demand = {product: rng.uniform(45, 240) for product in products}
    base_product_price = {product: rng.uniform(8, 900) for product in products}
    category_multiplier = {
        "Electronics": 1.12,
        "Apparel": 1.08,
        "Grocery": 1.45,
        "Home": 0.95,
        "Beauty": 1.18,
        "Automotive": 0.72,
        "Office Supplies": 0.88,
    }
    region_multiplier = {
        "North America": 1.24,
        "Europe": 1.10,
        "Middle East": 0.88,
        "Asia Pacific": 1.28,
        "Latin America": 0.82,
        "Africa": 0.70,
    }
    market_multiplier = {
        "United States": 1.35,
        "Canada": 0.92,
        "Mexico": 0.82,
        "Germany": 1.05,
        "France": 0.96,
        "United Kingdom": 1.02,
        "Netherlands": 0.78,
        "UAE": 0.88,
        "Saudi Arabia": 1.02,
        "Qatar": 0.62,
        "Kuwait": 0.58,
        "China": 1.35,
        "India": 1.25,
        "Japan": 1.04,
        "Singapore": 0.70,
        "Australia": 0.86,
        "Brazil": 1.00,
        "Argentina": 0.68,
        "Chile": 0.63,
        "South Africa": 0.78,
        "Egypt": 0.68,
        "Kenya": 0.52,
    }
    channel_multiplier = {
        "Online": 1.05,
        "Retail": 1.00,
        "Wholesale": 1.35,
        "Partner": 0.88,
    }

    days_needed = _calculate_rows_per_combination(rows, n_products, len(REGIONS), len(CHANNELS))
    dates = pd.date_range(start=start_date, periods=days_needed, freq="D")
    records = []

    for date in dates:
        day_of_week = date.dayofweek
        month = date.month
        quarter = int(date.quarter)
        week_of_year = int(date.isocalendar().week)
        year_end_effect = 1.16 if month in [11, 12] else 1.0
        holiday_flag = int((month == 12 and date.day >= 18) or (month == 1 and date.day <= 5) or (month in [6, 7] and rng.random() < 0.03))
        economic_index = 100 + 5 * np.sin(2 * np.pi * week_of_year / 52) + rng.normal(0, 1.5)

        for product in products:
            category = category_map[product]
            for region in REGIONS:
                for channel in CHANNELS:
                    market = _choose_market(region, rng)
                    weekend = day_of_week in [5, 6]
                    promotion_probability = 0.18 if channel == "Online" else 0.13 if channel == "Retail" else 0.07
                    promotion_flag = int(rng.random() < promotion_probability)
                    marketing_base = {"Online": 1700, "Retail": 1300, "Wholesale": 850, "Partner": 950}[channel]
                    marketing_spend = max(0, rng.normal(marketing_base, marketing_base * 0.22)) * (1.75 if promotion_flag else 1.0)
                    competitor_price_index = max(0.70, min(1.35, rng.normal(1.0, 0.075)))
                    price_noise = rng.normal(1.0, 0.045)
                    unit_price = base_product_price[product] * price_noise * (0.93 if promotion_flag else 1.0)
                    inventory_level = max(0, rng.normal(base_product_demand[product] * 8.5, base_product_demand[product] * 1.9))
                    channel_weekly_effect = 1.0
                    if channel in ["Online", "Retail"] and weekend:
                        channel_weekly_effect = 1.18 if channel == "Online" else 1.24
                    elif channel == "Wholesale" and not weekend:
                        channel_weekly_effect = 1.14
                    elif channel == "Wholesale" and weekend:
                        channel_weekly_effect = 0.82

                    category_month_multiplier = {
                        "Electronics": 1.20 if month in [11, 12] else 1.06 if month in [8, 9] else 1.0,
                        "Apparel": 1.18 if month in [3, 4, 9, 10] else 1.10 if month in [6, 7] else 1.0,
                        "Grocery": 1.14 if holiday_flag else 1.0,
                        "Home": 1.15 if month in [4, 5, 11] else 1.0,
                        "Beauty": 1.16 if month in [2, 5, 12] else 1.0,
                        "Automotive": 1.12 if quarter in [2, 3] else 0.96,
                        "Office Supplies": 1.22 if month in [8, 9, 1] else 0.98,
                    }[category]

                    demand = (
                        base_product_demand[product]
                        * category_multiplier[category]
                        * region_multiplier[region]
                        * market_multiplier[market]
                        * channel_multiplier[channel]
                        * channel_weekly_effect
                        * category_month_multiplier
                        * year_end_effect
                    )
                    promotion_lift = {"Online": 1.30, "Retail": 1.20, "Wholesale": 1.08, "Partner": 1.12}[channel]
                    demand *= promotion_lift if promotion_flag else 1.0
                    demand *= 1.12 if holiday_flag and category in ["Electronics", "Apparel", "Grocery", "Beauty"] else 1.0
                    demand *= 1.0 + min(marketing_spend / 24_000, 0.20)
                    demand *= 1.0 + ((economic_index - 100) / 150)
                    demand *= 1.0 + ((competitor_price_index - 1.0) * 0.42)
                    demand *= max(0.68, min(1.22, (base_product_price[product] / max(unit_price, 1)) ** 0.35))

                    inventory_pressure = inventory_level / max(demand, 1)
                    if inventory_pressure < 2.0:
                        demand *= max(0.62, inventory_pressure / 2.0)

                    units_sold = max(0, rng.normal(demand, demand * 0.13))
                    anomaly_label = 0
                    anomaly_type = "normal"

                    if rng.random() < 0.04:
                        anomaly_label = 1
                        anomaly_type = str(rng.choice(["demand_spike", "demand_drop", "revenue_anomaly", "inventory_stockout"]))
                        if anomaly_type == "demand_spike":
                            units_sold *= rng.uniform(1.45, 2.25)
                            marketing_spend *= rng.uniform(1.05, 1.35)
                        elif anomaly_type == "demand_drop":
                            units_sold *= rng.uniform(0.35, 0.70)
                        elif anomaly_type == "revenue_anomaly":
                            unit_price *= rng.uniform(0.72, 1.38)
                            units_sold *= rng.uniform(0.85, 1.15)
                        else:
                            inventory_level = rng.uniform(0, max(25, demand * 0.55))
                            units_sold = min(units_sold * rng.uniform(0.45, 0.78), inventory_level * rng.uniform(0.75, 1.05))

                    units_sold = int(round(max(units_sold, 0)))
                    revenue = round(units_sold * unit_price, 2)
                    records.append(
                        {
                            "date": date.date().isoformat(),
                            "product_id": product,
                            "product_category": category,
                            "region": region,
                            "market": market,
                            "channel": channel,
                            "unit_price": round(unit_price, 2),
                            "promotion_flag": promotion_flag,
                            "holiday_flag": holiday_flag,
                            "marketing_spend": round(marketing_spend, 2),
                            "inventory_level": round(inventory_level, 2),
                            "competitor_price_index": round(competitor_price_index, 3),
                            "economic_index": round(economic_index, 3),
                            "day_of_week": day_of_week,
                            "month": month,
                            "quarter": quarter,
                            "units_sold": units_sold,
                            "revenue": revenue,
                            "anomaly_label": anomaly_label,
                            "anomaly_type": anomaly_type,
                        }
                    )
                    if len(records) >= rows:
                        df = pd.DataFrame(records)
                        validate_generated_data(df)
                        return df
    df = pd.DataFrame(records).head(rows)
    validate_generated_data(df)
    return df


def validate_generated_data(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Generated data is missing required columns: {missing_columns}")
    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    missing_value_columns = missing_values[missing_values > 0].to_dict()
    if missing_value_columns:
        raise ValueError(f"Generated data contains missing values: {missing_value_columns}")
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Generated data contains unparseable date values.")
    if (df["units_sold"] < 0).any():
        raise ValueError("Generated data contains negative units_sold values.")
    if (df["revenue"] < 0).any():
        raise ValueError("Generated data contains negative revenue values.")
    if set(df["anomaly_label"].unique()) - {0, 1}:
        raise ValueError("Generated data contains invalid anomaly_label values.")
    if set(df["anomaly_type"].unique()) - ANOMALY_TYPES:
        raise ValueError("Generated data contains invalid anomaly_type values.")
    if not (df.loc[df["anomaly_label"] == 0, "anomaly_type"] == "normal").all():
        raise ValueError("Normal rows must have anomaly_type='normal'.")
    if not (df.loc[df["anomaly_label"] == 1, "anomaly_type"] != "normal").all():
        raise ValueError("Anomaly rows must have anomaly_type different from 'normal'.")
    if not df["quarter"].between(1, 4).all():
        raise ValueError("Generated data contains quarter values outside 1..4.")
    invalid_market_rows = df.apply(lambda row: row["market"] not in REGION_MARKETS.get(row["region"], []), axis=1)
    if invalid_market_rows.any():
        raise ValueError("Generated data contains market values that do not belong to their region.")


def save_sales_data(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_preview_data(df: pd.DataFrame, output_path: Path) -> Path:
    preview_path = output_path.parent / "sales_data_preview.csv"
    df.head(100).to_csv(preview_path, index=False)
    return preview_path


def print_generation_summary(df: pd.DataFrame, output_path: Path, preview_path: Path) -> None:
    date_values = pd.to_datetime(df["date"])
    anomaly_rate = df["anomaly_label"].mean() * 100
    anomaly_counts = df["anomaly_type"].value_counts().reindex(["normal", "demand_spike", "demand_drop", "revenue_anomaly", "inventory_stockout"], fill_value=0)
    print("Synthetic enterprise sales data generated successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Output: {output_path}")
    print(f"Preview: {preview_path}")
    print(f"Date range: {date_values.min().date()} to {date_values.max().date()}")
    print(f"Products: {df['product_id'].nunique()}")
    print(f"Categories: {df['product_category'].nunique()}")
    print(f"Regions: {df['region'].nunique()}")
    print(f"Markets: {df['market'].nunique()}")
    print(f"Channels: {df['channel'].nunique()}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print("Anomaly types:")
    for anomaly_type, count in anomaly_counts.items():
        print(f"- {anomaly_type}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic enterprise sales data.")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--products", type=int, default=40)
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR / "sales_data.csv")
    parser.add_argument("--load-postgres", action="store_true")
    parser.add_argument("--table-name", type=str, default="sales_data")
    parser.add_argument("--seed", type=int, default=settings.default_random_seed)
    args = parser.parse_args()

    ensure_directories()
    df = generate_sales_data(
        rows=args.rows,
        start_date=args.start_date,
        n_products=args.products,
        random_seed=args.seed,
    )
    output_path = save_sales_data(df, args.output)
    preview_path = save_preview_data(df, output_path)
    print_generation_summary(df, output_path, preview_path)

    if args.load_postgres:
        loaded = write_dataframe_to_postgres(df, table_name=args.table_name)
        print(f"PostgreSQL load status: {'success' if loaded else 'skipped'}")


if __name__ == "__main__":
    main()
