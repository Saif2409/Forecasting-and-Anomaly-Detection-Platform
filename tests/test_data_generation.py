import pandas as pd

from src.data.generate_synthetic_data import ANOMALY_TYPES, REGION_MARKETS, generate_sales_data, save_preview_data, save_sales_data
from src.data.validation import EXPECTED_COLUMNS, validate_sales_data


def test_data_generation_creates_expected_columns():
    df = generate_sales_data(rows=3000, n_products=8, random_seed=42)
    assert len(df) == 3000
    assert set(EXPECTED_COLUMNS).issubset(df.columns)
    assert "market" in df.columns
    assert "quarter" in df.columns
    assert "anomaly_type" in df.columns
    assert df["anomaly_label"].isin([0, 1]).all()
    assert set(df["anomaly_type"]).issubset(ANOMALY_TYPES)
    anomaly_rate = df["anomaly_label"].mean()
    assert anomaly_rate > 0
    assert 0.03 <= anomaly_rate <= 0.06
    assert (df["units_sold"] >= 0).all()
    assert (df["revenue"] >= 0).all()
    assert df.apply(lambda row: row["market"] in REGION_MARKETS[row["region"]], axis=1).all()


def test_data_generation_saves_full_and_preview_files(tmp_path):
    df = generate_sales_data(rows=3000, n_products=8, random_seed=42)
    output_path = tmp_path / "sales_data.csv"
    preview_path = save_preview_data(df, save_sales_data(df, output_path))
    assert output_path.exists()
    assert preview_path.exists()
    assert len(pd.read_csv(output_path)) == 3000
    assert len(pd.read_csv(preview_path)) == 100


def test_generated_data_passes_validation():
    df = generate_sales_data(rows=3000, n_products=8, random_seed=42)
    report = validate_sales_data(df)
    assert report["passed"] is True
