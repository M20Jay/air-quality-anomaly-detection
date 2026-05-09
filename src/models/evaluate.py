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