# Week 6 — Environmental Anomaly Detection + Time Series Forecasting

**Project:** Air Quality Anomaly Detection Pipeline
**Dataset:** OpenAQ API — Real Nairobi PM2.5 sensor data
**Stack:** ARIMA · Prophet · LSTM (PyTorch) · Isolation Forest · FastAPI · Streamlit · Docker

---

## Overview

Production air quality monitoring pipeline that answers two questions automatically for every hourly PM2.5 reading:

- **Will pollution levels rise?** — ARIMA + Prophet + LSTM forecasting
- **Is this reading dangerous?** — Isolation Forest anomaly detection

---

## Data Summary

| Metric | Value |
|--------|-------|
| Raw rows fetched | 11,998 |
| Hourly rows after resampling | 1,620 |
| Locations monitored | 5 Nairobi stations |
| Date range | 2024-01-30 → 2024-04-05 |
| Training rows | 1,296 (80%) |
| Test rows | 324 (20%) |
| Maximum PM2.5 spike | 469 µg/m³ — 2024-02-18 04:00 |
| WHO annual safe limit | 5 µg/m³ |
| Spike vs WHO limit | 93x above annual safe limit |
| Dangerous readings | 1.8% exceed US EPA threshold of 55 µg/m³ |

---

## EDA Key Findings

| Finding | Detail |
|---------|--------|
| Stationarity | ADF test p=0.0000 — series is stationary — d=0 for ARIMA |
| Daily peaks | 4am and 4pm — night burning and traffic |
| Worst day of week | Friday — consistently above WHO 24-hour limit of 15 µg/m³ |
| Best day of week | Sunday — lowest PM2.5 levels |
| Seasonal pattern | Dry season (Jan-Mar) higher than wet season |

---

## Model Results

| Model | RMSE | MAE | MAPE | Type | Notes |
|-------|------|-----|------|------|-------|
| **ARIMA (2,0,2)** | **9.93** | **8.35** | **100.64%** | Forecasting | ✅ Best model |
| LSTM (PyTorch) | 19.46 | 17.87 | 155.28% | Deep learning | lookback=24 units=64 epochs=50 |
| Prophet | 22.05 | 19.40 | 187.13% | Forecasting | daily seasonality enabled |
| Isolation Forest | — | — | — | Anomaly detection | 1 anomaly detected (0.32%) |

---

## Architecture Decisions

**Why ARIMA beat LSTM:**
PM2.5 series is stationary (ADF p=0.0000) — ARIMA's ideal condition.
Dataset has only 1,620 rows — LSTM requires significantly more data.
ARIMA is interpretable — critical for public health policy decisions.
RMSE 9.93 vs LSTM 19.46 — clear performance advantage.

**Why Isolation Forest for anomaly detection:**
Unsupervised — no labelled anomaly data required.
Efficient on high-dimensional data.
contamination=0.002 — expects 0.2% anomalies based on domain knowledge.
Detected spike of 469 µg/m³ — 93x the WHO annual safe limit.

**Why MAPE is high (100%+):**
MAPE divides error by actual value.
When PM2.5 is near zero (2 µg/m³) even small errors become 100%+.
RMSE and MAE are the reliable metrics for this dataset.
High MAPE does not mean poor model — context matters.

---

## Key Concepts

**Stationarity:**
A time series is stationary when mean and variance
do not change over time — no trend, no seasonality.
ADF (Augmented Dickey-Fuller) test checks this statistically.
Null hypothesis: series has a unit root (non-stationary).
p < 0.05 → reject null → series IS stationary → d=0 for ARIMA.
p > 0.05 → non-stationary → difference the series → d=1 or d=2.

**ARIMA parameters (p, d, q):**
p = autoregressive order  → how many past values to use
d = differencing order    → how many times to difference the series
q = moving average order  → how many past forecast errors to use
Our model: ARIMA(2, 0, 2)
d=0 because series is already stationary.
Selected via ACF and PACF plots + AIC minimisation.

**Isolation Forest algorithm:**
Randomly selects a feature and a random split value.
Recursively partitions the data into subsets.
Anomalies are isolated in fewer splits than normal points.
Path length = number of splits needed to isolate a point.
Short path = anomaly. Long path = normal observation.

**Lag features:**
pm25_lag1  → PM2.5 value 1 hour ago
pm25_lag24 → PM2.5 value 24 hours ago (yesterday same hour)
pm25_rolling_mean_24 → average over last 24 hours
These capture temporal dependencies for LSTM and Isolation Forest.

---

## Pipeline Architecture
OpenAQ API
↓
src/data/ingestion.py          → data/raw/nairobi_pm25_raw.csv
↓
src/data/preprocessing.py      → data/processed/nairobi_pm25_hourly.csv
↓
src/features/feature_engineering.py → data/processed/nairobi_pm25_features.csv
↓
src/models/train.py            → models/ (ARIMA · Prophet · LSTM · IsolationForest)
↓
src/models/evaluate.py         → RMSE · MAE · MAPE comparison
↓
api/main.py                    → /forecast · /anomaly · /health
↓
streamlit_app.py               → interactive dashboard
↓
Docker + AWS EC2               → production deployment

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Data source | OpenAQ API | Real Nairobi PM2.5 sensor readings |
| Forecasting | ARIMA · Prophet · LSTM (PyTorch) | Time series prediction |
| Anomaly detection | Isolation Forest (scikit-learn) | Spike detection |
| Feature engineering | Lag features · rolling averages · time features | Model inputs |
| API | FastAPI · Uvicorn · Pydantic | Production inference endpoints |
| Dashboard | Streamlit · Plotly | Interactive visualisation |
| Testing | pytest | 10/10 tests passing |
| Containerisation | Docker · docker-compose | Environment consistency |
| Deployment | AWS EC2 | Production hosting |
| Logging | Python logging · rotating file handler | Observability |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check — model availability |
| `/forecast` | POST | Forecast next N hours of PM2.5 (1-168 hours) |
| `/anomaly` | POST | Detect if a PM2.5 reading is anomalous |

---

## CLI Reference

**Setup:**
```bash
git clone https://github.com/M20Jay/air-quality-anomaly-detection.git
cd air-quality-anomaly-detection
pip install -r requirements.txt
```

**Run pipeline manually:**
```bash
python -m src.data.ingestion
python -m src.data.preprocessing
python -m src.features.feature_engineering
python -m src.models.train
python -m src.models.evaluate
```

**Start API:**
```bash
uvicorn api.main:app --reload
# Open http://127.0.0.1:8000/docs
```

**Run with Docker:**
```bash
docker-compose up --build
```

**Run tests:**
```bash
pytest tests/ -v
```

**Run individual scripts with PYTHONPATH:**
```bash
PYTHONPATH=. python src/data/ingestion.py
PYTHONPATH=. python src/data/preprocessing.py
PYTHONPATH=. python src/features/feature_engineering.py
PYTHONPATH=. python src/models/train.py
PYTHONPATH=. python src/models/evaluate.py
```

---

*Week 6 · 15-Week MLOps Programme · Built in Nairobi, Kenya 🇰🇪*