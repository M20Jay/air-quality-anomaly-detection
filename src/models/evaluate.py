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
    series.index.freq= 'h'
    splt_idx = int(len(series)*0.8)
    test = series[splt_idx:]
    logger.info(f"Test set: {len(test)} rows")
    return series, test

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    """Calculate RMSE, MAE and MAPE"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual,predicted)
    mape = np.mean(np.abs(actual-predicted)/actual) *100
    logger.info(f"{model_name} → RMSE: {rmse:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}%")
    return {'model': model_name, 'rmse': rmse, 'mae': mae, 'mape': mape}


def evaluate_arima(series:pd.Series, test: pd.Series) -> dict:
    """Evaluate ARIMA model on test data"""
    logger.info("Evaluating ARIMA...")
    with open(f"{MODELS_PATH}arima_model.pkl", 'rb') as f:
        fitted_model = pickle.load(f)

    # Forecast for test period
    forecast = fitted_model.forecast(steps=len(test))

    # Calculate metrics
    metrics = calculate_metrics(test.values, forecast.values, "ARIMA")
    return metrics

def evaluate_prophet(series: pd.Series, test: pd.Series) -> dict:
    """Evaluate Prophet model on test data"""
    logger.info("Evaluating Prophet...")
    with open(f"{MODELS_PATH}prophet_model.pkl", 'rb') as f:
        model =pickle.load(f)

    # Create future dataframe for test period
    future = pd.DataFrame({'ds': test.index.tz_localize(None)})

    # Predict
    forecast = model.predict(future)
    predicted = forecast['yhat'].values

    # Calculate metrics
    metrics = calculate_metrics(
        test.values,
        predicted,
        "Prophet"
        
    )
    return metrics

def evaluate_lstm(config: dict) -> dict:
    """Evaluate LSTM model on test data"""
    import torch
    import torch.nn as nn

    logger.info("Evaluating LSTM...")

    lookback = config['lstm']['lookback_window']

    # Load features X = np.array(X)
    filepath = os.path.join(PROCESSED_PATH,"nairobi_pm25_features.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Load scaler
    scaler = joblib.load(f"{MODELS_PATH}lstm_scaler.pkl")
    scaled = scaler.transform(df.values)

    # Create sequences
    X, y = [], []
    for i in range (lookback,len(scaled)):
        X.append(scaled[i-lookback:i])
        y.append(scaled[i, 0])
    X = np.array(X)
    logger.info(f"X shape: {X.shape}")
    y = np.array(y)

    # Split to get test portion
    split_idx = int(len(X) * 0.8)
    X_test = torch.FloatTensor(X[split_idx:])
    y_test = y[split_idx:]
    logger.info(f"X_test shape: {X_test.shape}, y_test length: {len(y_test)}")

    # Define model architecture
    class LSTMModel(nn.Module):
        def __init__(self, input_size,hidden_size):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(hidden_size,hidden_size//2, batch_first=True)
            self.fc = nn.Linear(hidden_size//2,1)

        def forward(self,x):
            out,_ = self.lstm1(x)
            out   = self.dropout(out)
            out,_ = self.lstm2(out)
            out = self.fc(out[:,-1,:])
            return out
        
    # Load saved weights
    units = config['lstm']['units']
    input_size = X.shape[2]
    model = LSTMModel(input_size, units)
    model.load_state_dict(torch.load(f"{MODELS_PATH}lstm_model.pt"))
    model.eval()

    # Predict
    with torch.no_grad():
        predictions = model(X_test).squeeze().numpy()
    
    # Inverse scale
    dummy = np.zeros((len(predictions), df.shape[1]))
    dummy[:, 0] = predictions  # ← missing line
    predictions_rescaled = scaler.inverse_transform(dummy)[:, 0]

    dummy_y = np.zeros((len(y_test), df.shape[1]))
    dummy_y[:, 0] = y_test
    y_test_rescaled = scaler.inverse_transform(dummy_y)[:, 0]

    metrics = calculate_metrics(y_test_rescaled, predictions_rescaled, "LSTM")
    return metrics

def evaluate_isolation_forest() -> dict:
    """Evaluate Isolation Forest anomaly detection"""

    # Load features data:
    filepath = os.path.join(PROCESSED_PATH, "nairobi_pm25_features.csv")
    
    df = pd.read_csv(filepath, index_col =0, parse_dates=True)

    # Split to test set
    split_idx = int(len(df)* 0.8)
    df_test = df.iloc[split_idx:]

    # Load model
    model = joblib.load(f"{MODELS_PATH}isolation_forest.pkl")

    # Predict - returns 1 (normal) or -1 (anomaly)
    predictions = model.predict(df_test[['pm25']])

    # Count anomalies
    n_anomalies = (predictions == -1).sum()
    anomaly_pct = (n_anomalies/len(predictions)) *100

    logger.info(f"Isolation Forest → Anomalies detected: {n_anomalies} ({anomaly_pct:.1f}%)")

    return{
        'model': 'Isolation Forest',
        'anomalies_detected': int(n_anomalies),
        'anomaly_percentage': round(anomaly_pct, 2),
        'total_predictions': len(predictions)
    }

if __name__ == "__main__":
    logger.info("Starting model evaluation pipeline...")

    config = load_config()
    series, test =load_test_data()

    # Evaluate all models
    arima_metrics = evaluate_arima(series, test)
    prophet_metrics = evaluate_prophet(series, test)
    lstm_metrics = evaluate_lstm(config)
    if_metrics = evaluate_isolation_forest()

    # Summary table
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION SUMMARY")
    logger.info("=" * 60)

    for metrics in [arima_metrics, prophet_metrics, lstm_metrics]:
        logger.info(
            f"{metrics['model']:15} → "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"MAE: {metrics['mae']:.2f} | "
            f"MAPE: {metrics['mape']:.2f}%"
        )

    logger.info(
        f"Isolation Forest → "
        f"Anomalies: {if_metrics['anomalies_detected']} "
        f"({if_metrics['anomaly_percentage']}%)"
    )
    logger.info("=" * 60)
    logger.info("Evaluation complete ✅")