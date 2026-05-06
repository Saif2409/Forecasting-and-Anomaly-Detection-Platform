from src.data.generate_synthetic_data import generate_sales_data
from src.features.build_features import build_feature_datasets


def test_feature_pipeline_creates_required_features_without_missing_values():
    df = generate_sales_data(rows=1200, n_products=4, random_seed=42)
    train_df, validation_df, test_df, feature_columns = build_feature_datasets(df)
    required = {"units_sold_lag_1", "rolling_mean_7", "rolling_std_7", "price_ratio_vs_competitor", "quarter", "time_index"}
    assert required.issubset(train_df.columns)
    assert len(feature_columns) > 0
    assert "anomaly_label" not in feature_columns
    assert "anomaly_type" not in feature_columns
    assert any(column.startswith("market_") for column in feature_columns)
    assert train_df[feature_columns].isna().sum().sum() == 0
    assert len(train_df) > len(validation_df) > 0
    assert len(test_df) > 0
