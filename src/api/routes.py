from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from src.api.schemas import AnomalyResponse, BatchPredictionResponse, BusinessRecord, HealthResponse, MetricsResponse, PredictionResponse
from src.config.settings import settings
from src.inference.predict import ModelNotFoundError, detect_anomalies, model_status, predict_demand
from src.monitoring.performance_monitor import monitor

router = APIRouter()


@router.get("/")
def root() -> dict[str, str]:
    return {"service": settings.project_name, "status": "running"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = model_status()
    service_status = "healthy" if all(status.values()) else "degraded_models_not_loaded"
    return HealthResponse(status=service_status, models=status)


@router.post("/predict", response_model=PredictionResponse)
def predict(record: BusinessRecord) -> PredictionResponse:
    start = time.perf_counter()
    try:
        prediction = float(predict_demand([record.model_dump()])[0])
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    monitor.record(latency_ms=latency_ms, predictions=[prediction])
    return PredictionResponse(predicted_units_sold=round(prediction, 2))


@router.post("/detect-anomaly", response_model=AnomalyResponse)
def detect_anomaly(record: BusinessRecord) -> AnomalyResponse:
    start = time.perf_counter()
    try:
        flags, scores = detect_anomalies([record.model_dump()])
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    monitor.record(latency_ms=latency_ms, predictions=[], anomaly_flags=[int(flags[0])])
    return AnomalyResponse(anomaly_flag=int(flags[0]), anomaly_score=round(float(scores[0]), 6))


@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(records: list[BusinessRecord]) -> BatchPredictionResponse:
    if not records:
        raise HTTPException(status_code=400, detail="At least one record is required.")
    start = time.perf_counter()
    try:
        predictions = predict_demand([record.model_dump() for record in records])
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - start) * 1000
    prediction_values = [float(value) for value in predictions]
    monitor.record(latency_ms=latency_ms, predictions=prediction_values)
    return BatchPredictionResponse(predictions=[PredictionResponse(predicted_units_sold=round(value, 2)) for value in prediction_values])


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**monitor.snapshot())
