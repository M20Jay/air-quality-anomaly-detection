# =============================================================
# tests/test_anomaly.py
# =============================================================
# Purpose: Tests for the /anomaly endpoint.
#
# Run with: pytest tests/ -v
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_anomaly_normal_reading():
    """Test normal PM2.5 reading returns not anomaly."""
    response = client.post("/anomaly", json={"pm25": 8.5})
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] == False
    assert data["risk_level"] == "GOOD"

def test_anomaly_dangerous_reading():
    """Test dangerous PM2.5 reading returns anomaly."""
    response = client.post("/anomaly", json={"pm25": 469.23})
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] == True
    assert data["risk_level"] == "HAZARDOUS"

def test_anomaly_negative_rejected():
    """Test negative PM2.5 is rejected."""
    response = client.post("/anomaly", json={"pm25": -5.0})
    assert response.status_code == 400

def test_anomaly_returns_score():
    """Test anomaly score is returned."""
    response = client.post("/anomaly", json={"pm25": 12.0})
    data = response.json()
    assert "anomaly_score" in data
    assert isinstance(data["anomaly_score"], float)

def test_anomaly_risk_levels():
    """Test different PM2.5 values return correct risk levels."""
    tests = [
        (8.0, "GOOD"),
        (20.0, "MODERATE"),
        (40.0, "UNHEALTHY FOR SENSITIVE GROUPS"),
        (100.0, "UNHEALTHY"),
        (200.0, "HAZARDOUS"),
    ]
    for pm25, expected_risk in tests:
        response = client.post("/anomaly", json={"pm25": pm25})
        data = response.json()
        assert data["risk_level"] == expected_risk