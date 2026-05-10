# =============================================================
# src/models/evaluate.py
# =============================================================
# Purpose: Evaluate all trained models on test data.
#          Calculate RMSE, MAE for forecasting models.
#          Calculate anomaly detection metrics for
#          Isolation Forest.
#
# Models evaluated:
#   1. ARIMA     — RMSE, MAE, MAPE
#   2. Prophet   — RMSE, MAE, MAPE
#   3. LSTM      — RMSE, MAE, MAPE
#   4. Isolation Forest — anomaly scores, flagged readings
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import yaml
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
CONFIG_PATH = "configs/model.yaml"

def load_config() -> dict:
    """Load model configuration"""
    with open(CONFIG_PATH,'r') as f:
        config = yaml.safe_load(f)
    logger.info("Config Loaded")
    return config

def load_test_data() -> pd.Series:
    """Load processed data and return test set"""
    filepath = os.path.join(PROCESSED_PATH,"nairobi_pm25_hourly.csv")
    df = pd.read_csv(filepath, index_col =0, parse_dates=True)
    series=df.squeeze()
    series.index.freq('h')
    splt_idx = int(len(series)*0.8)
    test = series[splt_idx:]
    logger.info(f"Test set: {len(test)} rows")
    return series, test




