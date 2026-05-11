# =============================================================
# tests/test_forecast.py
# =============================================================
# Purpose: Tests for the /forecast endpoint.
#
# Run with: pytest tests/ -v
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_forecast_default():
    """Test forecast with default 24 steps."""
    response = client.post("/forecast", json={"steps": 24})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "ARIMA"
    assert data["steps"] == 24
    assert len(data["forecast"]) == 24
    assert data["unit"] == "µg/m³"

def test_forecast_custom_steps():
    """Test forecast with custom steps."""
    response = client.post("/forecast", json={"steps": 48})
    assert response.status_code == 200
    data = response.json()
    assert len(data["forecast"]) == 48

def test_forecast_invalid_steps_zero():
    """Test forecast rejects zero steps."""
    response = client.post("/forecast", json={"steps": 0})
    assert response.status_code == 400

def test_forecast_invalid_steps_too_large():
    """Test forecast rejects steps over 168."""
    response = client.post("/forecast", json={"steps": 200})
    assert response.status_code == 400

def test_forecast_values_are_positive():
    """Test all forecast values are positive PM2.5."""
    response = client.post("/forecast", json={"steps": 24})
    data = response.json()
    assert all(v > 0 for v in data["forecast"])