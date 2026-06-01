
## Overview

Week 8 applies production MLOps tooling to the Week 6 air quality pipeline,
transforming a manually-run script collection into a fully automated,
monitored, and versioned ML system.

The five tools work together as a complete MLOps stack:
- **MLflow** — tracks every experiment automatically
- **DVC** — versions data and model files like Git
- **Evidently AI** — monitors data drift in production
- **Prefect** — orchestrates the full pipeline with retry logic
- **GitHub Actions** — automates testing, drift detection and alerting on every push

---

## MLOps Stack Overview

| Tool | Category | Purpose | What it Automates |
|------|----------|---------|-------------------|
| **MLflow** | Experiment tracking | Records params, metrics, artifacts per run | Manual logging of training results |
| **DVC** | Data versioning | Tracks CSV and model file versions | Manual file management and backup |
| **Evidently AI** | Drift monitoring | Compares training vs production distributions | Manual data quality checks |
| **Prefect** | Orchestration | Runs all pipeline steps in order with retry | Manual script execution in correct order |
| **GitHub Actions** | CI/CD | Tests, detects drift, sends alerts on every push | Manual testing and deployment |

---

## Day 1 — MLflow Experiment Tracking

### What MLflow Does
```
Records every training run automatically:
→ Parameters  — ARIMA(p,d,q) · LSTM epochs/units/batch · Prophet seasonality
→ Metrics     — ARIMA AIC · LSTM loss per epoch
→ Artifacts   — saved model .pkl and .pt files
→ Tags        — data_start_date · data_end_date · total_rows · reference_mean_pm25
→ Run name    — ARIMA · Prophet · LSTM · IsolationForest
→ Status      — FINISHED · FAILED · RUNNING
→ Timestamp   — exact start and end time of each run
```

### MLflow Results
```
Experiment: air-quality-nairobi
Run 1: ARIMA    → AIC: 10613.49  → FINISHED
Run 2: Prophet  → no metric      → FINISHED
Run 3: LSTM     → loss: 0.0302   → FINISHED
Run 4: IsolationForest           → FINISHED
Data range logged: 2024-01-30 → 2024-04-05
Total rows: 1,572 | Training: 1,257 | Test: 315
```

### Key Code Pattern
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("air-quality-nairobi")

with mlflow.start_run(run_name="ARIMA"):
    mlflow.log_param("p", p)
    mlflow.log_param("d", d)
    mlflow.log_param("q", q)
    mlflow.log_metric("aic", round(fitted_model.aic, 2))
    mlflow.log_artifact("models/arima_model.pkl")

# Log data metadata at experiment level
mlflow.set_experiment_tags({
    "data_start_date": str(df.index.min().date()),
    "data_end_date": str(df.index.max().date()),
    "total_rows": str(len(df)),
    "reference_mean_pm25": str(round(df["pm25"].mean(), 2)),
})
```

### MLflow CLI Reference
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open http://127.0.0.1:5001

# Fix protobuf conflict if UI shows INTERNAL_ERROR
pip install protobuf==4.25.3

# Check runs from terminal
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
runs = client.search_runs('1', order_by=['start_time DESC'], max_results=4)
for run in runs:
    print(f'{run.info.run_name} | {run.info.status} | {run.data.metrics}')
"
```

---

## Day 2 — DVC Data Version Control

### What DVC Does
```
Tracks large data and model files outside Git.
Creates a small pointer file (.dvc) that Git tracks.
Actual file stored in a remote (local path or S3).
Enables: which data produced which model?
```

### How It Works
```
Original file: data/processed/nairobi_pm25_features.csv  (245KB)
DVC pointer:   data/processed/nairobi_pm25_features.csv.dvc  (107 bytes)

Git tracks    → the 107-byte pointer file
DVC tracks    → the actual 245KB CSV
Remote stores → the actual file content

Result: repo stays small, data is versioned
```

### Files Tracked by DVC
```
data/processed/nairobi_pm25_features.csv
data/processed/nairobi_pm25_hourly.csv
models/arima_model.pkl
models/prophet_model.pkl
models/lstm_model.pt
models/isolation_forest.pkl
```

### DVC CLI Reference
```bash
# Initialise DVC in project
dvc init

# Add file to DVC tracking
dvc add data/processed/nairobi_pm25_features.csv

# Push files to remote storage
dvc push

# Pull files from remote storage
dvc pull

# Check what has changed
dvc status

# List configured remotes
dvc remote list

# Add local remote
dvc remote add myremote /tmp/dvc-storage

# Set default remote
dvc remote default myremote
```

---

## Day 3 — Evidently AI Drift Detection

### Three Types of Drift
```
Data drift:
→ Input feature distribution changes
→ PM2.5 mean dropped from 19.02 to 12.50 µg/m³
→ DETECTED in this pipeline ⚠️
→ Caused by: seasonal change · new road · sensor recalibration

Target drift:
→ Output/prediction distribution changes
→ High pollution hours dropped from 30% to 10% of readings
→ Monitor actual PM2.5 value distribution over time

Concept drift:
→ Relationship between input and output changes
→ 4am no longer predicts high pollution — traffic patterns changed
→ Most dangerous — model learns wrong patterns
→ Detected via rising RMSE trend in MLflow over time
```

### Drift Detection Results
```
Reference data: 1,257 rows (training period)
Current data:   315 rows  (test period)
Columns monitored: pm25 · hour

pm25 → DRIFTED ⚠️
       Wasserstein distance: 0.273595
       Reference mean: 19.02 µg/m³
       Current mean:   12.50 µg/m³

hour → No drift ✅
```

### Key Code Pattern
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("reports/drift/drift_report.html")

# Check drift programmatically
result = report.as_dict()
drift_detected = result["metrics"][0]["result"]["dataset_drift"]
# True = drift found → alert
# False = no drift → continue
```

### Evidently CLI Reference
```bash
# Run drift detection
PYTHONPATH=. python src/models/drift_detection.py

# Open signed drift report
open reports/drift/drift_report_signed.html

# Live drift report
# https://m20jay.github.io/air-quality-anomaly-detection
```

---

## Day 4 — Prefect Pipeline Orchestration

### Key Concepts
```
@task  → one tracked step in the pipeline
         has retry logic, logging, state tracking

@flow  → the complete pipeline
         connects all tasks in correct order
         manages execution and error handling

retries=2              → retry failed task 2 times automatically
retry_delay_seconds=30 → wait 30 seconds between retries
log_prints=True        → show all logs in Prefect UI dashboard
run_ingestion=True     → fetch fresh data from OpenAQ API
run_ingestion=False    → use existing data on disk (faster for testing)
```

### Pipeline Run Results
```
Flow: air-quality-pipeline
Runs completed: 3
Run names: bizarre-oyster · pompous-chameleon · dynamic-wren
Average duration: ~35 seconds (without ingestion)
Full pipeline with ingestion: ~93 seconds
All runs: COMPLETED ✅
```

### Key Code Pattern
```python
from prefect import flow, task
from prefect.logging import get_run_logger
import subprocess, sys, os

@task(name="Train Models", retries=1, retry_delay_seconds=60)
def train_models():
    logger = get_run_logger()
    logger.info("Starting model training...")
    result = subprocess.run(
        [sys.executable, "-m", "src.models.train"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if result.returncode != 0:
        raise Exception(f"Training failed: {result.stderr}")
    logger.info("Model training complete ✅")
    return True

@flow(name="air-quality-pipeline", log_prints=True)
def air_quality_pipeline(run_ingestion: bool = True):
    logger = get_run_logger()
    if run_ingestion:
        ingest_data()
        preprocess_data()
    else:
        logger.info("Skipping ingestion — using existing data")
    train_models()
    detect_drift()
```

### subprocess.run explained
```python
subprocess.run(
    [sys.executable, "-m", "src.models.train"],  # command to run
    capture_output=True,   # capture stdout and stderr
    text=True,             # return as strings not bytes
    cwd=os.getcwd()        # run from project root directory
)
# result.returncode → 0 = success, non-zero = failure
# result.stdout     → normal output
# result.stderr     → error output
```

### Prefect CLI Reference
```bash
# Start Prefect server
prefect server start

# Connect pipeline to running server
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Run full pipeline with fresh data from OpenAQ
PYTHONPATH=. python src/models/pipeline.py

# Open Prefect UI
# http://127.0.0.1:4200

# List all flows
prefect flow ls

# List all deployments
prefect deployment ls
```

---

## Day 5 — GitHub Actions CI/CD

### What Happens on Every Push
```
Trigger: push to main branch (src/ or data/ files changed)
Also runs: every Monday at 6am (scheduled drift check)

Steps:
1. Ubuntu machine spins up in GitHub cloud
2. Python 3.12 installed
3. pip install -r requirements.txt
4. pytest tests/ -v --tb=short
5. PYTHONPATH=. python src/models/drift_detection.py
6. Drift report uploaded as downloadable artifact
7. Drift check → if detected → ⚠️ warning banner in GitHub UI
8. Email alert sent if drift detected
```

### Workflow Structure
```yaml
name: Air Quality ML Pipeline

on:
  push:
    branches: [ main ]
    paths:
      - 'src/**'
      - 'data/**'
      - 'requirements.txt'
  pull_request:
    branches: [ main ]
  workflow_dispatch:
  schedule:
    - cron: '0 6 * * 1'   # Every Monday at 6am

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
```

### Key yml Concepts
```
runs-on: ubuntu-latest    → fresh Ubuntu VM in GitHub cloud
continue-on-error: true   → do not stop pipeline if tests fail
if: always()              → run this step even if previous steps failed
workflow_dispatch:        → allows manual trigger from GitHub UI
paths:                    → only trigger if these files changed
cron: '0 6 * * 1'        → minute hour day month weekday
                            0=minute 0, 6=6am, *=any day,
                            *=any month, 1=Monday
```

### GitHub Secrets Setup
```
Location: Repository → Settings → Secrets and variables → Actions
Required secrets:
→ EMAIL_ADDRESS  → sender Gmail address
→ EMAIL_PASSWORD → Gmail app password (16 chars, no spaces)

Generate app password: myaccount.google.com/apppasswords
Important: store EMAIL_ADDRESS under Secrets not Variables
```

### Email Alert Code Pattern
```python
import smtplib, os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender   = os.environ["EMAIL_ADDRESS"]   # from GitHub Secret
password = os.environ["EMAIL_PASSWORD"]  # from GitHub Secret

msg = MIMEMultipart()
msg["From"]    = sender
msg["To"]      = receiver
msg["Subject"] = "⚠️ MLOps Alert: PM2.5 Data Drift Detected"
msg.attach(MIMEText(body, "plain"))

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()          # encrypt the connection
server.login(sender, password)
server.send_message(msg)
server.quit()
```

---

## Full Automated Pipeline Flow

```
Developer pushes code to GitHub (src/ or data/ changed)
    ↓
GitHub Actions triggers automatically
    ↓
Fresh Ubuntu 22.04 machine spins up in GitHub cloud
    ↓
Python 3.12 installed + pip install -r requirements.txt
    ↓
pytest tests/ -v — runs all 10 tests
    ↓
PYTHONPATH=. python src/models/drift_detection.py
    ↓
Drift report uploaded as downloadable artifact (999KB)
    ↓
Drift check: PM2.5 mean 19.02 → 12.50 µg/m³ — DRIFTED ⚠️
    ↓
Email alert sent automatically
    ↓
Email received: ⚠️ MLOps Alert: PM2.5 Data Drift Detected
```

---

## Tool Comparison

### Prefect vs Airflow vs GitHub Actions
```
Prefect:
→ Pipeline orchestration — runs ML steps in order
→ Modern Python-first approach
→ Retry logic and logging built in
→ Best for small to medium pipelines
→ UI: http://127.0.0.1:4200

Airflow:
→ Pipeline orchestration — DAG-based visual graph
→ Industry standard at large companies (Google · Airbnb · Twitter)
→ More complex setup — more powerful for parallel workflows
→ DAGs define task dependencies explicitly
→ Week 9 of this programme

GitHub Actions:
→ CI/CD automation — not a pipeline orchestrator
→ Runs on every git push automatically
→ Tests + drift detection + alerts + deployment
→ Complements Prefect and Airflow
→ Free for public repositories
```

### MLflow vs DVC
```
MLflow:
→ Tracks WHAT happened in training
→ Parameters, metrics, model performance
→ Answers: which experiment gave best RMSE?
→ UI: http://127.0.0.1:5001

DVC:
→ Tracks WHAT FILES were used
→ Data and model file versions
→ Answers: which data produced this model?
→ Works like Git for binary files
→ Integrates with S3, GCS, Azure Blob

They complement each other:
MLflow → experiment results
DVC    → files that produced those results
```

---

## Pipeline Architecture

```
OpenAQ API
    ↓
src/data/ingestion.py                → data/raw/
    ↓
src/data/preprocessing.py            → data/processed/nairobi_pm25_hourly.csv
    ↓
src/features/feature_engineering.py  → data/processed/nairobi_pm25_features.csv
    ↓
src/models/train.py                  → models/ + MLflow tracking
    ↓
src/models/drift_detection.py        → reports/drift/drift_report.html
    ↓
src/models/pipeline.py               → Prefect orchestration
    ↓
.github/workflows/ml-pipeline.yml    → GitHub Actions CI/CD
    ↓
Email alert if drift detected        → automated monitoring
```

---

## CLI Reference

**Start all services:**
```bash
# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Prefect server
prefect server start

# Connect to Prefect server
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Run full pipeline with fresh data
PYTHONPATH=. python src/models/pipeline.py

# Run drift detection only
PYTHONPATH=. python src/models/drift_detection.py

# Run training only
PYTHONPATH=. python src/models/train.py
```

**Git workflow:**
```bash
git status
git add .
git commit -m "feat: description"
git push origin main
git log --oneline -5
git restore --staged filename
```

**Search and navigate files:**
```bash
grep -n "search term" src/models/train.py      # find text with line numbers in one file
grep -r "search term" src/                     # find text recursively in all files
grep -i "search term" src/models/train.py      # case insensitive search
grep -c "INFO" logs/app.log                    # count matching lines
grep -v "DEBUG" logs/app.log                   # show lines NOT matching
grep "mlflow" src/models/train.py | head -20   # first 20 matching lines
sed -n '70,80p' src/models/train.py            # show lines 70 to 80
sed -n '255,275p' src/models/train.py          # show lines 255 to 275
tail -5 src/models/pipeline.py                 # show last 5 lines
ls -lh data/processed/                         # list files with sizes
```

**Check MLflow runs from terminal:**
```bash
python -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
client = mlflow.tracking.MlflowClient()
runs = client.search_runs('1', order_by=['start_time DESC'], max_results=4)
for run in runs:
    print(f'{run.info.run_name} | {run.info.status} | {run.data.metrics}')
"
```

**DVC commands:**
```bash
dvc init
dvc add data/processed/nairobi_pm25_features.csv
dvc push
dvc pull
dvc status
dvc remote list
```

**Fix common errors:**
```bash
# MLflow protobuf conflict
pip install protobuf==4.25.3

# Module not found
PYTHONPATH=. python src/models/train.py

# Port already in use
lsof -i :5001
kill -9 <PID>

# DVC files not found after clone
dvc pull
```

---

*Week 8 · 15-Week MLOps Programme · Built in Nairobi, Kenya 🇰🇪*
---

## Week 8 Bonus — AWS EC2 Migration

Migrated all 6 production APIs from Render free tier to AWS EC2 t3.small Frankfurt.

### Key commands used

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start all services after reboot
cd ~/recommendation-system && docker compose up -d
cd ~/churn-prediction-pipeline && docker compose up -d churn-api
cd ~/fraud-detection-pipeline && docker compose -f docker-compose-api.yml up -d
cd ~/customer-segmentation && docker compose up -d seg_fastapi
cd ~/credit-risk-scoring-pipeline && docker compose up -d api
cd ~/air-quality-anomaly-detection && docker compose up -d

# ngrok tunnel (auto-starts via systemd)
sudo systemctl status ngrok

# Cloudflare tunnel (auto-starts via systemd)
sudo systemctl status cloudflared-airquality

# Check all containers
docker ps

# Expand disk
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /
```

### sed commands for README updates

```bash
# Update IP across all repos
sed -i '' 's|18.199.241.52|18.184.3.203|g' README.md

# Fix port conflict — credit risk postgres
sed -i 's/"5432:5432"/"5435:5432"/' docker-compose.yml

# Fix fraud API model path
sed -i 's/restart: unless-stopped/environment:\n      - MODEL_PATH=\/app\/src\/fraud_pipeline.pkl\n    restart: unless-stopped/' docker-compose-api.yml

# Fix recommendation logger bug
sed -i 's/            logger.warning(f"DB save skipped: {e}")/            print(f"DB save skipped: {e}")/' api/routes/recommend.py
```

### Instance details
- Instance: i-0b3c0fda7da6ccb95
- IP: 18.184.3.203
- Type: t3.small (upgraded from t3.micro — OOM crashes)
- Region: eu-central-1 Frankfurt
- Disk: 30GB (expanded from 20GB)
- Memory: 2GB · Usage ~64%

---

## Week 8 Bonus — AWS EC2 Migration

Migrated all 6 production APIs from Render free tier to AWS EC2 t3.small Frankfurt.

### Key commands used

```bash
# SSH to server
ssh -i ~/Documents/GitHub/mlops-key.pem ubuntu@18.184.3.203

# Start all services after reboot
cd ~/recommendation-system && docker compose up -d
cd ~/churn-prediction-pipeline && docker compose up -d churn-api
cd ~/fraud-detection-pipeline && docker compose -f docker-compose-api.yml up -d
cd ~/customer-segmentation && docker compose up -d seg_fastapi
cd ~/credit-risk-scoring-pipeline && docker compose up -d api
cd ~/air-quality-anomaly-detection && docker compose up -d

# ngrok tunnel (auto-starts via systemd)
sudo systemctl status ngrok

# Cloudflare tunnel (auto-starts via systemd)
sudo systemctl status cloudflared-airquality

# Check all containers
docker ps

# Expand disk
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
df -h /
```

### sed commands for README updates

```bash
# Update IP across all repos
sed -i '' 's|18.199.241.52|18.184.3.203|g' README.md

# Fix port conflict — credit risk postgres
sed -i 's/"5432:5432"/"5435:5432"/' docker-compose.yml

# Fix fraud API model path
sed -i 's/restart: unless-stopped/environment:\n      - MODEL_PATH=\/app\/src\/fraud_pipeline.pkl\n    restart: unless-stopped/' docker-compose-api.yml

# Fix recommendation logger bug
sed -i 's/            logger.warning(f"DB save skipped: {e}")/            print(f"DB save skipped: {e}")/' api/routes/recommend.py
```

### Instance details
- Instance: i-0b3c0fda7da6ccb95
- IP: 18.184.3.203
- Type: t3.small (upgraded from t3.micro — OOM crashes)
- Region: eu-central-1 Frankfurt
- Disk: 30GB (expanded from 20GB)
- Memory: 2GB · Usage ~64%
