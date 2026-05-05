# Air Quality Anomaly Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

Production time series forecasting and anomaly detection system for air quality monitoring.
Built by Martin James Ng'ang'a — MLOps Engineer | Nairobi, Kenya

> 🔗 Live API — added on deployment

## What This System Does

This pipeline ingests real air quality sensor data from OpenAQ, forecasts future pollution levels using three models, and automatically flags dangerous anomalies for immediate action.

Three questions answered for every sensor reading:
- **Will pollution levels rise?** — ARIMA + Prophet + LSTM forecasting
- **Is this reading abnormal?** — Isolation Forest anomaly detection
- **Why did this spike happen?** — Feature attribution and context

## Tech Stack

| Layer | Technology |
|---|---|
| Forecasting | ARIMA · Prophet · LSTM |
| Anomaly Detection | Isolation Forest |
| API | FastAPI · Uvicorn · Pydantic |
| Dashboard | Streamlit · Plotly |
| Database | PostgreSQL |
| Monitoring | Grafana |
| Containerisation | Docker · docker-compose |
| Deployment | Render |
| Versioning | DVC · Git |

## Project Structure

```
air-quality-anomaly-detection/
├── configs/             # Model parameters
├── data/                # Raw and processed data
├── notebooks/           # EDA and experiments
├── src/
│   ├── data/            # Ingestion and preprocessing
│   ├── features/        # Feature engineering
│   ├── models/          # Training and evaluation
│   └── utils/           # Logging and utilities
├── api/
│   ├── routes/          # Forecast and anomaly endpoints
│   └── schemas/         # Request and response models
├── tests/               # Unit tests
└── streamlit_app.py     # Interactive dashboard
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health check |
| `/forecast` | POST | Predict future AQI values |
| `/anomaly` | POST | Detect anomalous readings |

## Dataset

Real air quality sensor data from OpenAQ — open-source platform aggregating government air quality data from cities worldwide.

## Results

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| ARIMA | TBD | TBD | TBD |
| Prophet | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |

*Results updated as models are trained*

## Running Locally

```
git clone https://github.com/M20Jay/air-quality-anomaly-detection.git
cd air-quality-anomaly-detection
pip install -r requirements.txt
uvicorn api.main:app --reload
```

## Running with Docker

```
docker-compose up --build
```

---
*Building from Nairobi. For the environment and the world. 🇰🇪*