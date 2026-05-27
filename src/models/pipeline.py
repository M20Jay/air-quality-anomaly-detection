# =============================================================
# src/models/pipeline.py
# =============================================================
# Purpose: Prefect workflow orchestration for the full
#          air quality ML pipeline
#
# Pipeline steps:
#   1. Ingest data from OpenAQ
#   2. Preprocess and feature engineer
#   3. Train all 4 models
#   4. Detect data drift with Evidently AI
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================
import subprocess
import sys
import os
from prefect import flow, task
from prefect.logging import get_run_logger
from datetime import datetime


@task(name="Ingest Data", retries=2, retry_delay_seconds=30)
def ingest_data():
    """Fetch fresh PM2.5 data from OpenAQ API"""
    logger = get_run_logger()
    logger.info("Starting data ingestion...")

    result = subprocess.run(
        [sys.executable, "-m", "src.data.ingestion"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        logger.error(f"Ingestion failed: {result.stderr}")
        raise Exception(f"Data ingestion failed: {result.stderr}")

    logger.info("Data ingestion complete ✅")
    return True


@task(name="Preprocess Data", retries=2, retry_delay_seconds=30)
def preprocess_data():
    """Clean, resample and engineer features"""
    logger = get_run_logger()
    logger.info("Starting preprocessing...")

    result = subprocess.run(
        [sys.executable, "-m", "src.data.preprocessing"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        logger.error(f"Preprocessing failed: {result.stderr}")
        raise Exception(f"Preprocessing failed: {result.stderr}")

    logger.info("Preprocessing complete ✅")
    return True


@task(name="Train Models", retries=1, retry_delay_seconds=60)
def train_models():
    """Train ARIMA, Prophet, LSTM and Isolation Forest"""
    logger = get_run_logger()
    logger.info("Starting model training...")

    result = subprocess.run(
        [sys.executable, "-m", "src.models.train"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        raise Exception(f"Model training failed: {result.stderr}")

    logger.info("Model training complete ✅")
    return True


@task(name="Detect Drift", retries=1, retry_delay_seconds=30)
def detect_drift():
    """Run Evidently drift detection"""
    logger = get_run_logger()
    logger.info("Starting drift detection...")

    result = subprocess.run(
        [sys.executable, "-m", "src.models.drift_detection"],
        capture_output=True,
        text=True,
        cwd=os.getcwd()
    )

    if result.returncode != 0:
        logger.error(f"Drift detection failed: {result.stderr}")
        raise Exception(f"Drift detection failed: {result.stderr}")

    logger.info("Drift detection complete ✅")
    return True


@flow(name="air-quality-pipeline", log_prints=True)
def air_quality_pipeline(run_ingestion: bool = True):
    """
    Full air quality ML pipeline:
    Ingest → Preprocess → Train → Detect Drift
    """
    logger = get_run_logger()
    start_time = datetime.now()
    logger.info(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1 — Ingest data
    if run_ingestion:
        ingest_data()
        preprocess_data()
    else:
        logger.info("Skipping ingestion — using existing data")

    # Step 2 — Train models
    train_models()

    # Step 3 — Detect drift
    detect_drift()

    end_time = datetime.now()
    duration = (end_time - start_time).seconds
    logger.info(f"Pipeline complete ✅ Duration: {duration} seconds")


if __name__ == "__main__":
    # Full end to end pipeline — fetch fresh data
    air_quality_pipeline(run_ingestion=True)