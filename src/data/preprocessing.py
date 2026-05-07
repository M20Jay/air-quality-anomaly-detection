# =============================================================
# src/data/preprocessing.py
# =============================================================
# Purpose: Clean and prepare raw OpenAQ air quality data
#          for time series modelling.
#
# Pipeline:
#   1. Load raw CSV from data/raw/
#   2. Filter to PM2.5 parameter only
#   3. Convert to datetime index
#   4. Resample to consistent hourly frequency
#   5. Interpolate missing values
#   6. Save clean data to data/processed/
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"

def load_raw_data(filename: str) -> pd.DataFrame:
    """Load raw CSV from data/raw/"""
    filepath = os.path.join(RAW_PATH, filename)
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df

def filter_pm25(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to PM2.5 parameter only"""
    df_pm25 = df[df['parameter.name'] == 'pm25'].copy()
    logger.info(f"PM2.5 rows: {len(df_pm25)}")
    return df_pm25

def prepare_timeseries(df: pd.DataFrame) -> pd.Series:
    """Convert to datetime index and resample to hourly"""
    df['datetime'] = pd.to_datetime(df['period.datetimeFrom.utc'])
    df = df.set_index('datetime')
    df = df.sort_index()
    
    # Filter to 2024 — most complete and continuous data
    df = df[df.index.year == 2024]
    logger.info(f"Rows after 2024 filter: {len(df)}")
    
    df_hourly = df['value'].resample('h').mean()
    df_hourly = df_hourly.interpolate(method='linear')
    logger.info(f"Hourly series: {len(df_hourly)} rows")
    logger.info(f"Missing values: {df_hourly.isnull().sum()}")
    return df_hourly

def save_processed(df: pd.Series, filename: str) -> None:
    """Save processed data to data/processed/"""
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    filepath = os.path.join(PROCESSED_PATH, filename)
    df.to_csv(filepath)
    logger.info(f"Saved processed data to {filepath}")

if __name__ == "__main__":
    logger.info("Starting preprocessing pipeline...")
    raw_file = "nairobi_aqi_20260506.csv"
    df = load_raw_data(raw_file)
    df_pm25 = filter_pm25(df)
    df_hourly = prepare_timeseries(df_pm25)
    save_processed(df_hourly, "nairobi_pm25_hourly.csv")
    logger.info("Preprocessing complete ✅")