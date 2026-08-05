"""Telemetry data model for JetPilotGuard.

A ``TelemetryPacket`` is one time-step of fused aircraft-state and pilot
bio-sensor data. Fields carry explicit physical units in their names or
docstrings so that downstream code never has to guess.

Note on scope: these are *simulated* signals. JetPilotGuard is a research and
demonstration platform, not certified avionics. See docs/DESIGN.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# Physical/plausibility bounds used for input validation. These are deliberately
# generous (they describe "a sensor could report this", not "this is safe").
_BOUNDS = {
    "altitude_ft": (0.0, 60_000.0),
    "airspeed_kt": (0.0, 900.0),
    "pitch_deg": (-90.0, 90.0),
    "roll_deg": (-180.0, 180.0),
    "g_force": (-3.0, 12.0),
    "ppg_amplitude": (0.0, 1.0),
    "gaze_offset_deg": (0.0, 90.0),
}


@dataclass(frozen=True, slots=True)
class TelemetryPacket:
    """One fused sample of aircraft kinematics and pilot physiology.

    Attributes:
        altitude: Height above ground level, in feet (AGL).
        airspeed: True airspeed, in knots.
        pitch: Nose-up positive, in degrees.
        roll: Right-wing-down positive, in degrees.
        g_force: Vertical load factor, in g.
        ppg_amplitude: Cranial photoplethysmography amplitude, normalised 0-1.
            Low values indicate reduced cerebral blood flow (G-LOC precursor).
        gaze_offset_deg: Angular deviation of gaze from boresight, in degrees.
            High sustained values indicate spatial disorientation / task
            saturation.
    """

    altitude: float
    airspeed: float
    pitch: float
    roll: float
    g_force: float
    ppg_amplitude: float
    gaze_offset_deg: float

    def __post_init__(self) -> None:
        self._check("altitude_ft", self.altitude)
        self._check("airspeed_kt", self.airspeed)
        self._check("pitch_deg", self.pitch)
        self._check("roll_deg", self.roll)
        self._check("g_force", self.g_force)
        self._check("ppg_amplitude", self.ppg_amplitude)
        self._check("gaze_offset_deg", self.gaze_offset_deg)

    @staticmethod
    def _check(name: str, value: float) -> None:
        lo, hi = _BOUNDS[name]
        if not (lo <= value <= hi):
            raise ValueError(
                f"{name}={value} outside plausible sensor range [{lo}, {hi}]"
            )
