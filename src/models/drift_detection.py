# =============================================================
# src/models/drift_detection.py
# =============================================================
# Purpose: Detect data drift between training and test data
#          using Evidently AI
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = "data/processed/"
REPORTS_PATH = "reports/drift/"


def load_data():
    """Load training and test data for drift detection"""
    filepath = os.path.join(PROCESSED_PATH, "nairobi_pm25_features.csv")
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    split_idx = int(len(df) * 0.8)
    reference_data = df.iloc[:split_idx]
    current_data = df.iloc[split_idx:]

    logger.info(f"Reference data: {len(reference_data)} rows")
    logger.info(f"Current data: {len(current_data)} rows")

    return reference_data, current_data


def detect_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> str:
    """Run Evidently drift detection report"""

    logger.info("Running drift detection...")

    # Select columns to monitor
    columns = ['pm25', 'hour', 'day_of_week', 'month',
               'pm25_lag1', 'pm25_lag24', 'pm25_rolling_mean_24']

    # Use only columns that exist in both datasets
    available_columns = [col for col in columns if col in reference_data.columns]
    logger.info(f"Monitoring columns: {available_columns}")

    ref = reference_data[available_columns].reset_index(drop=True)
    cur = current_data[available_columns].reset_index(drop=True)

    # Create report
    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    # Run report
    report.run(reference_data=ref, current_data=cur)

    # Save HTML report
    os.makedirs(REPORTS_PATH, exist_ok=True)
    report_path = os.path.join(REPORTS_PATH, "drift_report.html")
    report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")

    # Create wrapper with author signature
    wrapper_path = os.path.join(REPORTS_PATH, "drift_report_signed.html")
    with open(wrapper_path, 'w') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Air Quality Drift Report — Martin James Ng'ang'a</title>
</head>
<body style="margin:0; padding:0; font-family:Arial;">
  <div style="text-align:center; padding:15px; background:#4169E1; color:white;">
    <h2 style="margin:0;">🌍 Nairobi Air Quality — Data Drift Report</h2>
    <p style="margin:5px 0;">
      Built by <strong>Martin James Ng'ang'a</strong> | MLOps Engineer | Nairobi, Kenya 🇰🇪
    </p>
    <p style="margin:5px 0;">
      Week 8 — MLOps Automation · Evidently AI Drift Detection &nbsp;|&nbsp;
      <a href="https://github.com/M20Jay" style="color:white;">github.com/M20Jay</a>
      &nbsp;·&nbsp;
      <a href="https://www.linkedin.com/in/martin-james-nganga" style="color:white;">LinkedIn</a>
    </p>
  </div>
  <iframe src="drift_report.html"
          style="width:100%; height:calc(100vh - 100px); border:none;">
  </iframe>
</body>
</html>""")
    logger.info(f"Signed report saved to {wrapper_path}")

    return wrapper_path


if __name__ == "__main__":
    logger.info("Starting drift detection...")

    reference_data, current_data = load_data()
    report_path = detect_drift(reference_data, current_data)

    logger.info("Drift detection complete ✅")
    logger.info(f"Report saved to {report_path}")