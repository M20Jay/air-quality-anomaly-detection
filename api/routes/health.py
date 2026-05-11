# =============================================================
# api/routes/health.py
# =============================================================
# Purpose: Health check endpoint.
#          Returns API status and model availability.
# =============================================================

from fastapi import APIRouter
from src.utils.logger import get_logger
import os

logger = get_logger(__name__)

router = APIRouter()

@router.get("/health")
def health_check():
    """Check if the API and models are available."""
    
    models = {
        "arima": os.path.exists("models/arima_model.pkl"),
        "prophet": os.path.exists("models/prophet_model.pkl"),
        "lstm": os.path.exists("models/lstm_model.pt"),
        "isolation_forest": os.path.exists("models/isolation_forest.pkl")
    }
    
    all_healthy = all(models.values())
    logger.info(f"Health check → {'healthy' if all_healthy else 'degraded'}")
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "models": models,
        "version": "1.0.0"
    }