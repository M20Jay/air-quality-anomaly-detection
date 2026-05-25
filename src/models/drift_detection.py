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
    
    # Inject author signature
    with open(report_path, 'r') as f:
        html = f.read()
    
    signature = """
    <div style="text-align:center; padding:20px; font-family:Arial; 
                color:#555; border-top:2px solid #4169E1; margin-top:30px;">
      <p style="font-size:16px; font-weight:bold;">
        🌍 Built by Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya 🇰🇪
      </p>
      <p style="font-size:14px;">Week 8 — MLOps Automation · Evidently AI Drift Detection</p>
      <p style="font-size:14px;">
        <a href="https://github.com/M20Jay" style="color:#4169E1;">github.com/M20Jay</a> · 
        <a href="https://www.linkedin.com/in/martin-james-nganga" style="color:#4169E1;">LinkedIn</a>
      </p>
    </div>
    """
    
    html = html.replace('</body>', signature + '</body>')
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    logger.info(f"Drift report saved to {report_path}")

    return report_path


if __name__ == "__main__":
    logger.info("Starting drift detection...")

    reference_data, current_data = load_data()
    report_path = detect_drift(reference_data, current_data)

    logger.info("Drift detection complete ✅")
    logger.info(f"Report saved to {report_path}")