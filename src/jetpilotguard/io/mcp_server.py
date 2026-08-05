"""Model Context Protocol (MCP) server exposing JetPilotGuard to AI copilots.

Lets an agent (Claude, Cursor, etc.) query the safety engine programmatically:
score a telemetry sample, or fetch the static operational envelope.

Run:
    python -m jetpilotguard.io.mcp_server
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from jetpilotguard.engine import JetPilotGuardEngine
from jetpilotguard.telemetry import TelemetryPacket

mcp = FastMCP("JetPilotGuard")
_engine = JetPilotGuardEngine(stateful_filtering=False)


@mcp.tool
def evaluate_flight_safety(
    altitude: float,
    airspeed: float,
    pitch: float,
    roll: float,
    g_force: float,
    ppg_amplitude: float,
    gaze_offset_deg: float,
) -> dict[str, Any]:
    """Assess one telemetry sample for pilot impairment and collision risk.

    Returns the fused status (NOMINAL / ADVISORY / CRITICAL_OVERRIDE), the ML
    impairment probability, time-to-impact, and the filtered signal values.
    """
    packet = TelemetryPacket(
        altitude=altitude,
        airspeed=airspeed,
        pitch=pitch,
        roll=roll,
        g_force=g_force,
        ppg_amplitude=ppg_amplitude,
        gaze_offset_deg=gaze_offset_deg,
    )
    return _engine.process(packet).to_dict()


@mcp.resource("jetpilotguard://operational-envelope")
def operational_envelope() -> str:
    """Static advisory operational envelope for the simulated airframe.

    NOTE: illustrative values for a fast-jet-like simulation, not certified
    limits for any real aircraft.
    """
    envelope = {
        "airframe": "Simulated fast-jet (illustrative)",
        "altitude_ft": {"min": 100.0, "max": 50000.0},
        "airspeed_kt": {"min": 80.0, "max": 650.0},
        "pitch_deg": {"min": -45.0, "max": 45.0},
        "roll_deg": {"min": -90.0, "max": 90.0},
        "g_force": {"min": 0.0, "max": 9.0},
        "ppg_amplitude": {"min": 0.0, "max": 0.6},
        "gaze_offset_deg": {"min": 0.0, "max": 30.0},
    }
    return json.dumps(envelope, indent=2)


if __name__ == "__main__":
    mcp.run()
