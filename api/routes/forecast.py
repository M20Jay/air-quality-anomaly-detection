# =============================================================
# api/routes/forecast.py
# =============================================================
# Purpose: Forecast endpoint — returns future PM2.5 predictions
#          using the best performing model (ARIMA).
# =============================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class ForecastRequest(BaseModel):
    steps: int = 24

class ForecastResponse(BaseModel):
    model: str
    steps: int
    forecast: list
    unit: str
    message: str

@router.post("/forecast", response_model=ForecastResponse)
def forecast_pm25(request: ForecastRequest):
    """Forecast future PM2.5 values using ARIMA model."""

    if request.steps < 1 or request.steps > 168:
        raise HTTPException(
            status_code=400,
            detail="Steps must be between 1 and 168 (1 week)"
        )

    try:
        with open("models/arima_model.pkl", "rb") as f:
            model = pickle.load(f)

        forecast = model.forecast(steps=request.steps)
        forecast_values = [round(float(v), 2) for v in forecast.values]

        logger.info(f"Forecast generated → {request.steps} steps")

        return ForecastResponse(
            model="ARIMA",
            steps=request.steps,
            forecast=forecast_values,
            unit="µg/m³",
            message=f"PM2.5 forecast for next {request.steps} hours"
        )

    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))