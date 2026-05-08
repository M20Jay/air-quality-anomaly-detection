# =============================================================
# src/models/train.py
# =============================================================
# Purpose: Train ARIMA, Prophet, LSTM and Isolation Forest
#          models on Nairobi PM2.5 hourly data.
#
# Models:
#   1. ARIMA     — statistical baseline forecast
#   2. Prophet   — Facebook time series forecast
#   3. LSTM      — deep learning sequence forecast
#   4. Isolation Forest — anomaly detection
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================
import pandas as pd
import numpy as np
import os
import pickle
import joblib
import yaml
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/"
MODELS_PATH = 'models/'
CONFIG_PATH = 'configs/model.yaml'

def load_config() -> dict:
    """Load model parameters from configs/model.yaml"""
    with open(CONFIG_PATH,'r') as f:
        config = yaml.safe_load(f)
        logger.info("Config loaded")
        return config
    
def load_data() -> pd.Series:
    """Load processed hourly PM2.5 data"""
    filepath=os.path.join(PROCESSED_PATH, "nairobi_pm25_hourly.csv")
    df=pd.read_csv(filepath, index_col=0, parse_dates=True)
    series= df.squeeze()
    logger.info(f" Loaded {len(series)} rows")
    return series

def split_data(series:pd.Series,train_size: float=0.8):
    """Split time series chronologically into train and test"""
    split_idx = int(len(series)*train_size)
    train = series[:split_idx]
    test = series[split_idx:]
    logger.info(f" Train : {len(train)} rows | Test: {len(test)} rows")
    return train, test


def train_arima(train: pd.Series, config: dict) -> object:
    """Train ARIMA model on training data"""
    p = config['arima']['p']
    d = config['arima']['d']
    q = config['arima']['q']
    logger.info(f"Train ARIMA ({p}, {d}), {q})...")
    model = ARIMA(train, order=(p,d,q))
    fitted_model =model.fit()
    logger.info(f" AIC: {fitted_model.aic:.2f}")
    os.makedirs(MODELS_PATH,exist_ok=True)
    with open(f"{MODELS_PATH} arima_model.pkl", 'wb') as f:
        pickle.dump(fitted_model,f)
    logger.info(f"ARIMA model saved to {MODELS_PATH} arima_model.pkl")
    return fitted_model


def train_prophet(train: pd.Series,  config: dict) -> object:
    """Train Prophet model on training data"""
    logger.info("Training Prophet model...")


    # Prophet requires specific column names
    df_prophet = pd.DataFrame({
        'ds' : train.index,
        'y'  : train.values
    })

    # Remove timezone from datetime
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

    model = Prophet(
        seasonality_mode = config['prophet']['seaonality_mode'],
        yearly_seasonality = config['prophet']['yearly_seasonalit'],
        weekly_seasonality=config['prophet']['weekly_seasonality'],
        daily_seasonality=config['prophet']['daily_seasonality']
    )
    model.fit(df_prophet)
    logger.info("Prophet trained successfully")

    with open(f" {MODELS_PATH}prophet_model.pkl", 'wb') as f:
        pickle.dump(model,f)
        logger.info(f"Prophet model saved to {MODELS_PATH}prophet_model.pkl")
        return model
    
def create_sequences(data: np.ndarray, lookback:int) -> tuple:
    """Create input sequences for LSTM"""
    X,y = [], []
    for i in range(lookback,len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i,0])
        return np.array(X), np.array(y)
    
def train_lstm(train:pd.DataFrame, config:dict)  -> object:
    """Train LSTM model on training data"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

        