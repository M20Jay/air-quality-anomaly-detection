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
from sklearn.preprocessing import MinMaxScaler
from src.utils.logger import get_logger
import mlflow
import mlflow.sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    filepath = os.path.join(PROCESSED_PATH, "nairobi_pm25_hourly.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    series = df.squeeze()
    series.index.freq = 'h'  # ← add this line
    logger.info(f"Loaded {len(series)} rows")
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

    with mlflow.start_run(run_name="ARIMA"):

        # Log parameters
        mlflow.log_param("p", p)
        mlflow.log_param("d", d)
        mlflow.log_param("q", q)

        logger.info(f"Train Arima {p}, {d}, {q}....")
        model = ARIMA(train, order =(p,d,q))
        fitted_model = model.fit()
        logger.info(f"AIC: {fitted_model.aic:.2f}")

        mlflow.log_metric("aic", round(fitted_model.aic, 2))

        os.makedirs(MODELS_PATH, exist_ok=True)
        with open(f"{MODELS_PATH}arima_model.pkl", 'wb') as f:
            pickle.dump(fitted_model, f)

        
        mlflow.log_artifact(f"{MODELS_PATH}arima_model.pkl")
        logger.info("ARIMA model saved and tracked in MLflow")

    return fitted_model


def train_prophet(train: pd.Series, config: dict) -> object:
    """Train Prophet model on training data"""

    with mlflow.start_run(run_name="Prophet"):

        mlflow.log_param("seasonality_mode", config['prophet']['seasonality_mode'])
        mlflow.log_param("yearly_seasonality", config['prophet']['yearly_seasonality'])
        mlflow.log_param("weekly_seasonality", config['prophet']['weekly_seasonality'])
        mlflow.log_param("daily_seasonality", config['prophet']['daily_seasonality'])

        logger.info("Training Prophet model...")

        df_prophet = pd.DataFrame({
            'ds': train.index,
            'y': train.values
        })
        df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

        model = Prophet(
            seasonality_mode=config['prophet']['seasonality_mode'],
            yearly_seasonality=config['prophet']['yearly_seasonality'],
            weekly_seasonality=config['prophet']['weekly_seasonality'],
            daily_seasonality=config['prophet']['daily_seasonality']
        )
        model.fit(df_prophet)
        logger.info("Prophet trained successfully")

        os.makedirs(MODELS_PATH, exist_ok=True)
        with open(f"{MODELS_PATH}prophet_model.pkl", 'wb') as f:
            pickle.dump(model, f)

        mlflow.log_artifact(f"{MODELS_PATH}prophet_model.pkl")
        logger.info("Prophet model saved and tracked in MLflow")

    return model
    
def create_sequences(data:np.ndarray, lookback: int) -> tuple:
    X,y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_lstm(train: pd.DataFrame, config: dict) -> object:
    """Train LSTM model using PyTorch"""

    lookback = config['lstm']['lookback_window']
    units = config['lstm']['units']
    epochs = config['lstm']['epochs']
    batch_size = config['lstm']['batch_size']

    with mlflow.start_run(run_name="LSTM"):

        mlflow.log_param("lookback_window", lookback)
        mlflow.log_param("units", units)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        logger.info(f"Training LSTM (PyTorch) — lookback={lookback}, units={units}, epochs={epochs}, batch_size={batch_size}")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(train.values)

        X, y = create_sequences(scaled, lookback)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        split = int(len(X_tensor) * 0.9)
        X_train = X_tensor[:split]
        y_train = y_tensor[:split]

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(0.2)
                self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
                self.fc = nn.Linear(hidden_size // 2, 1)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout(out)
                out, _ = self.lstm2(out)
                out = self.fc(out[:, -1, :])
                return out

        input_size = X.shape[2]
        model = LSTMModel(input_size, units)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")
                mlflow.log_metric("loss", round(total_loss, 4), step=epoch+1)

        os.makedirs(MODELS_PATH, exist_ok=True)
        torch.save(model.state_dict(), f"{MODELS_PATH}lstm_model.pt")
        joblib.dump(scaler, f"{MODELS_PATH}lstm_scaler.pkl")

        mlflow.log_artifact(f"{MODELS_PATH}lstm_model.pt")
        mlflow.log_artifact(f"{MODELS_PATH}lstm_scaler.pkl")
        logger.info("LSTM model saved and tracked in MLflow")

    return model, scaler


def train_isolation_forest(df: pd.DataFrame, config: dict) -> object:
    """Train Isolation Forest for anomaly detection"""

    with mlflow.start_run(run_name="IsolationForest"):

        contamination = config['isolation_forest']['contamination']
        n_estimators = config['isolation_forest']['n_estimators']
        random_state = config['isolation_forest']['random_state']

        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        logger.info("Training Isolation Forest...")

        filepath = os.path.join(PROCESSED_PATH, "nairobi_pm25_features.csv")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state
        )

        model.fit(df[['pm25']])
        logger.info("Isolation Forest trained successfully")

        os.makedirs(MODELS_PATH, exist_ok=True)
        joblib.dump(model, f"{MODELS_PATH}isolation_forest.pkl")

        mlflow.log_artifact(f"{MODELS_PATH}isolation_forest.pkl")
        logger.info("Isolation Forest saved and tracked in MLflow")

    return model


if __name__ =="__main__":
    logger.info("Starting model training pipeline...")
    mlflow.set_experiment("air-quality-nairobi")
    # Load config and data
    config = load_config()
    series = load_data()
    train, test = split_data(series)
    features_path = os.path.join(PROCESSED_PATH, "nairobi_pm25_features.csv")
    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    split_idx = int(len(df_features) * 0.8)
    df_train = df_features.iloc[:split_idx]

    # Log data metadata to MLflow tags
    mlflow.set_experiment_tags({
        "data_start_date": str(df_features.index.min().date()),
        "data_end_date": str(df_features.index.max().date()),
        "total_rows": str(len(df_features)),
        "reference_mean_pm25": str(round(df_features['pm25'].mean(), 2)),
        "training_rows": str(split_idx),
        "test_rows": str(len(df_features) - split_idx),
    })
    logger.info(f"Data range: {df_features.index.min().date()} → {df_features.index.max().date()}")
    logger.info(f"Total rows: {len(df_features)} | Training: {split_idx} | Test: {len(df_features) - split_idx}")

    # Train all models
    train_arima(train, config)
    train_prophet(train, config)
    train_lstm(df_train, config)
    train_isolation_forest(df_features, config)
    
    logger.info("All models trained successfully ✅")