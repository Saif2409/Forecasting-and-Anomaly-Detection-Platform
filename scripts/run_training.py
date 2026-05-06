from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.train_anomaly_model import train_anomaly_model
from src.models.train_forecasting_model import train_forecasting_model


if __name__ == "__main__":
    train_forecasting_model()
    train_anomaly_model()
