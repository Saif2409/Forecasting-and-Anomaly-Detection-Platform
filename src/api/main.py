from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router
from src.config.settings import settings

app = FastAPI(
    title=settings.project_name,
    description="Forecast enterprise sales demand and detect anomalous KPI behavior.",
    version="1.0.0",
)
app.include_router(router)
