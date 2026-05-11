# =============================================================
# api/routes/anomaly.py
# =============================================================
# Purpose: Anomaly detection endpoint.
#          Flags dangerous PM2.5 readings using
#          Isolation Forest model.
# =============================================================

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

class AnomalyRequest(BaseModel):
    pm25: float

class AnomalyResponse(BaseModel):
    pm25: float
    is_anomaly: bool
    anomaly_score: float
    risk_level: str
    message: str

@router.post("/anomaly", response_model=AnomalyResponse)
def detect_anomaly(request: AnomalyRequest):
    """Detect if a PM2.5 reading is anomalous."""

    if request.pm25 < 0:
        raise HTTPException(
            status_code=400,
            detail="PM2.5 value cannot be negative"
        )

    try:
        model = joblib.load("models/isolation_forest.pkl")

        reading = np.array([[request.pm25]])
        prediction = model.predict(reading)[0]
        score = model.decision_function(reading)[0]

        is_anomaly = prediction == -1

        if request.pm25 > 150:
            risk_level = "HAZARDOUS"
        elif request.pm25 > 55:
            risk_level = "UNHEALTHY"
        elif request.pm25 > 35:
            risk_level = "UNHEALTHY FOR SENSITIVE GROUPS"
        elif request.pm25 > 15:
            risk_level = "MODERATE"
        else:
            risk_level = "GOOD"

        message = (
            f"ANOMALY DETECTED — {risk_level}" 
            if is_anomaly 
            else f"Normal reading — {risk_level}"
        )

        logger.info(f"Anomaly check → PM2.5: {request.pm25} → {message}")

        return AnomalyResponse(
            pm25=request.pm25,
            is_anomaly=bool(is_anomaly),
            anomaly_score=round(float(score), 4),
            risk_level=risk_level,
            message=message
        )

    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))