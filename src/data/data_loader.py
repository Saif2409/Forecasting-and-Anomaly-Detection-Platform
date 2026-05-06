from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import RAW_DATA_DIR
from src.data.database import read_table_from_postgres


def load_sales_data(csv_path: Optional[Path] = None, source: str = "csv", table_name: str = "sales_data") -> pd.DataFrame:
    if source == "postgres":
        return read_table_from_postgres(table_name)

    path = csv_path or RAW_DATA_DIR / "sales_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Sales data not found at {path}. Run data generation first.")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
