from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BusinessRecord(BaseModel):
    date: date
    product_id: str = Field(..., min_length=1)
    product_category: Literal["Electronics", "Apparel", "Grocery", "Home", "Beauty", "Automotive", "Office Supplies"]
    region: Literal["North America", "Europe", "Middle East", "Asia Pacific", "Latin America", "Africa"]
    market: Literal[
        "United States",
        "Canada",
        "Mexico",
        "Germany",
        "France",
        "United Kingdom",
        "Netherlands",
        "UAE",
        "Saudi Arabia",
        "Qatar",
        "Kuwait",
        "China",
        "India",
        "Japan",
        "Singapore",
        "Australia",
        "Brazil",
        "Argentina",
        "Chile",
        "South Africa",
        "Egypt",
        "Kenya",
    ]
    channel: Literal["Online", "Retail", "Wholesale", "Partner"]
    unit_price: float = Field(..., gt=0)
    promotion_flag: int = Field(..., ge=0, le=1)
    holiday_flag: int = Field(..., ge=0, le=1)
    marketing_spend: float = Field(..., ge=0)
    inventory_level: float = Field(..., ge=0)
    competitor_price_index: float = Field(..., gt=0)
    economic_index: float = Field(..., gt=0)
    day_of_week: int = Field(..., ge=0, le=6)
    month: int = Field(..., ge=1, le=12)
    quarter: int | None = Field(default=None, ge=1, le=4)
    units_sold: float | None = Field(default=None, ge=0)
    revenue: float | None = Field(default=None, ge=0)
    anomaly_label: int | None = Field(default=0, ge=0, le=1)
    anomaly_type: Literal["normal", "demand_spike", "demand_drop", "revenue_anomaly", "inventory_stockout"] | None = "normal"

    @model_validator(mode="after")
    def fill_optional_values(self) -> "BusinessRecord":
        region_markets = {
            "North America": {"United States", "Canada", "Mexico"},
            "Europe": {"Germany", "France", "United Kingdom", "Netherlands"},
            "Middle East": {"UAE", "Saudi Arabia", "Qatar", "Kuwait"},
            "Asia Pacific": {"China", "India", "Japan", "Singapore", "Australia"},
            "Latin America": {"Brazil", "Argentina", "Chile"},
            "Africa": {"South Africa", "Egypt", "Kenya"},
        }
        if self.market not in region_markets[self.region]:
            raise ValueError("market must belong to the selected region")
        if self.quarter is None:
            self.quarter = ((self.month - 1) // 3) + 1
        if self.units_sold is not None and self.revenue is None:
            self.revenue = self.units_sold * self.unit_price
        return self


class PredictionResponse(BaseModel):
    predicted_units_sold: float


class AnomalyResponse(BaseModel):
    anomaly_flag: int
    anomaly_score: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    models: dict[str, bool]


class MetricsResponse(BaseModel):
    number_of_requests: int
    average_latency_ms: float
    prediction_count: int
    average_prediction_value: float
    prediction_min: float
    prediction_max: float
    anomaly_rate: float
    model_version: str
    last_prediction_timestamp: str | None
