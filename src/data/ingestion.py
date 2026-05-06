import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from src.utils.logger import get_logger

# Load environment variables from .env
load_dotenv()

logger = get_logger(__name__)

API_KEY = os.getenv("OPENAQ_API_KEY")
BASE_URL = "https://api.openaq.org/v3"

def fetch_nairobi_locations() -> list:
    """Fetch air quality sensor location IDs for Nairobi."""
    url = f"{BASE_URL}/locations"
    headers = {"X-API-Key": API_KEY}
    params = {
        "bbox": "36.6,-1.5,37.1,-1.0",
        "limit": 100
    }
    logger.info("Fetching Nairobi sensor locations...")
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    locations = data.get("results", [])
    logger.info(f"Found {len(locations)} locations in Kenya")
    return locations

def fetch_measurements(location_id: int, days: int = 90) -> pd.DataFrame:
    """Fetch measurements for a specific location."""
    url = f"{BASE_URL}/locations/{location_id}/sensors"
    headers = {"X-API-Key": API_KEY}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.warning(f"Failed to fetch sensors for location {location_id}")
        return pd.DataFrame()
    
    data = response.json()
    results = data.get("results", [])
    if not results:
        logger.warning(f"No sensors found for location {location_id}")
        return pd.DataFrame()
    
    all_measurements = []
    for sensor in results:
        sensor_id = sensor.get("id")
        sensor_url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
        params = {"limit": 1000}
        r = requests.get(sensor_url, headers=headers, params=params)
        if r.status_code == 200:
            measurements = r.json().get("results", [])
            if measurements:
                df = pd.json_normalize(measurements)
                df["sensor_id"] = sensor_id
                all_measurements.append(df)
    
    if all_measurements:
        return pd.concat(all_measurements, ignore_index=True)
    return pd.DataFrame()

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """Save dataframe to data/raw/ folder."""
    os.makedirs("data/raw", exist_ok=True)
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")

if __name__ == "__main__":
    logger.info("Starting OpenAQ data ingestion...")
    locations = fetch_nairobi_locations()
    all_data = []
    for loc in locations[:5]:
        loc_id = loc.get("id")
        loc_name = loc.get("name", "unknown")
        logger.info(f"Fetching measurements for: {loc_name}")
        df = fetch_measurements(loc_id)
        if not df.empty:
            df["location_name"] = loc_name
            all_data.append(df)
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        filename = f"nairobi_aqi_{datetime.now().strftime('%Y%m%d')}.csv"
        save_raw_data(combined, filename)
        logger.info(f"Ingestion complete. Total rows: {len(combined)}")
    else:
        logger.error("No data fetched. Check API key and connection.")