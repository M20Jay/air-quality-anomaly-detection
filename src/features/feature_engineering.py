# =============================================================
# src/features/feature_engineering.py
# =============================================================
# Purpose: Create features from clean hourly PM2.5 data
#          for time series modelling.
#
# Features created:
#   - Time features: hour, dayofweek, is_weekend
#   - Lag features: lag_1h, lag_24h, lag_48h
#   - Rolling stats: rolling_mean_6h, rolling_mean_24h,
#                    rolling_std_24h
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/"


def load_processed(filename: str) -> pd.DataFrame:
    """Load processed hourly data from data/processed/"""
    filepath = os.path.join(PROCESSED_PATH, filename)
    logger.info(f"Loading processed data from {filepath}")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = ['pm25']
    logger.info(f"Loaded {len(df)} rows")
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, dayofweek, is_weekend features"""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    logger.info("Time features created")
    return df


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features — past PM2.5 values"""
    df['lag_1h'] = df['pm25'].shift(1)
    df['lag_24h'] = df['pm25'].shift(24)
    df['lag_48h'] = df['pm25'].shift(48)
    logger.info("Lag features created")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and std features"""
    df['rolling_mean_6h'] = df['pm25'].rolling(window=6).mean()
    df['rolling_mean_24h'] = df['pm25'].rolling(window=24).mean()
    df['rolling_std_24h'] = df['pm25'].rolling(window=24).std()
    logger.info("Rolling features created")
    return df


def save_features(df: pd.DataFrame, filename: str) -> None:
    """Save feature engineered data to data/processed/"""
    filepath = os.path.join(PROCESSED_PATH, filename)
    df.to_csv(filepath)
    logger.info(f"Saved features to {filepath}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    logger.info("Starting feature engineering pipeline...")
    df = load_processed("nairobi_pm25_hourly.csv")
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = df.dropna()
    save_features(df, "nairobi_pm25_features.csv")
    logger.info("Feature engineering complete ✅")