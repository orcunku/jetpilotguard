"""Tests for the FastAPI serving layer.

These use FastAPI's TestClient, which runs the app in-process (no network, no
server to start), so they are fast and CI-friendly.
"""

import pytest
from fastapi.testclient import TestClient

from jetpilotguard.io.service import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_assess_nominal_cruise(client):
    r = client.post("/assess", json={
        "altitude": 5000, "airspeed": 350, "pitch": 0, "roll": 0,
        "g_force": 1.1, "ppg_amplitude": 0.42, "gaze_offset_deg": 2.0,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "NOMINAL"


def test_assess_collision_dive_triggers_override(client):
    r = client.post("/assess", json={
        "altitude": 400, "airspeed": 500, "pitch": -45, "roll": 0,
        "g_force": 2.5, "ppg_amplitude": 0.30, "gaze_offset_deg": 4.0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "CRITICAL_OVERRIDE"
    assert body["auto_gcas_engaged"] is True


def test_assess_rejects_out_of_range(client):
    r = client.post("/assess", json={
        "altitude": 5000, "airspeed": 350, "pitch": 999, "roll": 0,
        "g_force": 1.1, "ppg_amplitude": 0.42, "gaze_offset_deg": 2.0,
    })
    assert r.status_code == 422  # Pydantic validation error


def test_metrics_endpoint_exposes_prometheus(client):
    client.post("/assess", json={
        "altitude": 5000, "airspeed": 350, "pitch": 0, "roll": 0,
        "g_force": 1.1, "ppg_amplitude": 0.42, "gaze_offset_deg": 2.0,
    })
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "jetpilotguard_packets_total" in r.text
    assert "jetpilotguard_process_latency_seconds" in r.text
