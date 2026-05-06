from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import RAW_DATA_DIR, ensure_directories
from src.data.data_loader import load_sales_data
from src.data.generate_synthetic_data import generate_sales_data, print_generation_summary, save_preview_data, save_sales_data
from src.data.validation import print_validation_report, validate_sales_data
from src.features.build_features import build_feature_datasets, save_feature_outputs
from src.inference.batch_inference import run_batch_inference
from src.models.train_anomaly_model import train_anomaly_model
from src.models.train_forecasting_model import train_forecasting_model
from src.monitoring.generate_monitoring_report import create_monitoring_report


if __name__ == "__main__":
    ensure_directories()
    output_path = RAW_DATA_DIR / "sales_data.csv"
    if output_path.exists():
        df = load_sales_data(output_path)
        preview_path = save_preview_data(df, output_path)
        print(f"Existing dataset found and reused: {output_path}")
        print(f"Preview refreshed: {preview_path}")
        print(f"Rows: {len(df):,}")
    else:
        df = generate_sales_data(rows=50_000)
        save_sales_data(df, output_path)
        preview_path = save_preview_data(df, output_path)
        print_generation_summary(df, output_path, preview_path)
    validation_report = validate_sales_data(df)
    print_validation_report(validation_report)
    train_df, validation_df, test_df, feature_columns = build_feature_datasets(df)
    save_feature_outputs(train_df, validation_df, test_df, feature_columns)
    train_forecasting_model()
    train_anomaly_model()
    run_batch_inference()
    create_monitoring_report()
