# =============================================================
# api/main.py
# =============================================================
# Purpose: FastAPI application entry point.
#          Registers all routes and starts the server.
#
# Endpoints:
#   GET  /health   → service health check
#   POST /forecast → forecast future PM2.5 values
#   POST /anomaly  → detect anomalous readings
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import forecast, anomaly, health
from src.utils.logger import get_logger

logger =get_logger()

app = FastAPI(
    title="Air Quality Anomaly Detection API",
    description="Real-time PM2.5 forecasting and anomaly detection for Nairobi air quality monitoring.",
    version="1.0.0")

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_orgins["*"],
    allow_methods["*"],
    allow_headers["*"]
)

# Register routes
app.include_router(health.router)
app.include_router(forecast.router)
app.include_router(anomaly.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Air Quality API starting up...")
    logger.info("Models will load on first request")

async def  shutdown_event():
    logger.info("Air Quality API shutting down...")