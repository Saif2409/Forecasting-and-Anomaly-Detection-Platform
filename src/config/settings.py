from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DIR = REPORTS_DIR / "monitoring"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_name: str = "Enterprise Forecasting & Anomaly Detection MLOps Platform"
    environment: str = "local"
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_mlops"
    mlflow_tracking_uri: str = (ROOT_DIR / "mlruns").as_uri()
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_random_seed: int = 42


settings = Settings()


def ensure_directories() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PREDICTIONS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        MONITORING_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
