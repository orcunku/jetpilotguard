"""FastAPI service exposing the JetPilotGuard engine over HTTP.

Endpoints
---------
GET  /health   -> liveness check (is the service up?)
POST /assess   -> score one telemetry packet, return the full assessment
GET  /metrics  -> Prometheus metrics (scraped by Prometheus, not for humans)

Run locally (no Docker needed):
    uvicorn jetpilotguard.io.service:app --reload
Then open http://127.0.0.1:8000/docs for an auto-generated, interactive API UI.

Teaching note
-------------
FastAPI + Pydantic give us three things for free:
  1. Request validation -- the TelemetryIn model below rejects malformed input
     with a clear 422 error before our engine ever sees it.
  2. Auto-generated docs at /docs (try it -- you can call the API from a form).
  3. Type-safe responses.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from jetpilotguard import metrics
from jetpilotguard.engine import JetPilotGuardEngine
from jetpilotguard.telemetry import TelemetryPacket

# The engine is created once at startup and reused (loading the model per
# request would be slow). We stash it on the app state.
_engine: JetPilotGuardEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = JetPilotGuardEngine(stateful_filtering=False)
    yield
    _engine = None


app = FastAPI(
    title="JetPilotGuard",
    description="Simulated human-autonomy flight-safety watchdog (advisory + Auto-GCAS).",
    version="0.1.0",
    lifespan=lifespan,
)


class TelemetryIn(BaseModel):
    """Validated telemetry input. Bounds mirror the engine's own limits."""

    altitude: float = Field(..., ge=0, le=60000, description="AGL feet")
    airspeed: float = Field(..., ge=0, le=900, description="knots")
    pitch: float = Field(..., ge=-90, le=90, description="degrees")
    roll: float = Field(..., ge=-180, le=180, description="degrees")
    g_force: float = Field(..., ge=-3, le=12, description="load factor")
    ppg_amplitude: float = Field(..., ge=0, le=1, description="0-1 perfusion")
    gaze_offset_deg: float = Field(..., ge=0, le=90, description="degrees")


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/assess")
def assess(packet_in: TelemetryIn) -> dict:
    """Assess one telemetry packet and update monitoring metrics."""
    assert _engine is not None
    packet = TelemetryPacket(**packet_in.model_dump())

    start = time.perf_counter()
    assessment = _engine.process(packet)
    metrics.PROCESS_LATENCY.observe(time.perf_counter() - start)

    metrics.record_assessment(
        status=assessment.status.value,
        impairment=assessment.ml_impairment_probability,
        tti_seconds=assessment.time_to_impact_s,
    )
    return assessment.to_dict()


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Expose metrics in Prometheus text format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
