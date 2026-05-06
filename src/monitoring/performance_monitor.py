from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from src.config.settings import MONITORING_DIR
from src.models.model_registry import read_model_metadata


class PerformanceMonitor:
    def __init__(self) -> None:
        self._lock = Lock()
        self.request_count = 0
        self.total_latency_ms = 0.0
        self.predictions: list[float] = []
        self.anomaly_flags: list[int] = []
        self.last_prediction_timestamp: str | None = None

    def record(self, latency_ms: float, predictions: list[float], anomaly_flags: list[int] | None = None) -> None:
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            self.predictions.extend([float(value) for value in predictions])
            if anomaly_flags is not None:
                self.anomaly_flags.extend([int(value) for value in anomaly_flags])
            self.last_prediction_timestamp = datetime.now(timezone.utc).isoformat()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_latency = self.total_latency_ms / self.request_count if self.request_count else 0.0
            prediction_array = np.asarray(self.predictions, dtype=float) if self.predictions else np.asarray([])
            anomaly_rate = float(np.mean(self.anomaly_flags)) if self.anomaly_flags else 0.0
            forecasting_metadata = read_model_metadata("forecasting_model")
            return {
                "number_of_requests": self.request_count,
                "average_latency_ms": round(average_latency, 3),
                "prediction_count": len(self.predictions),
                "average_prediction_value": round(float(prediction_array.mean()), 3) if prediction_array.size else 0.0,
                "prediction_min": round(float(prediction_array.min()), 3) if prediction_array.size else 0.0,
                "prediction_max": round(float(prediction_array.max()), 3) if prediction_array.size else 0.0,
                "anomaly_rate": round(anomaly_rate, 4),
                "model_version": forecasting_metadata.get("created_at_utc", "not_available"),
                "last_prediction_timestamp": self.last_prediction_timestamp,
            }

    def save_snapshot(self, output_path: Path = MONITORING_DIR / "performance_metrics.json") -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return output_path


monitor = PerformanceMonitor()
