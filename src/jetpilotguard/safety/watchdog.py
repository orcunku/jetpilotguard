"""Deterministic ground-collision safety watchdog.

Design principle: the layer that can *override the pilot* must be simple,
transparent, and auditable. It is pure rule-based kinematics with no machine
learning in the loop, so its behaviour can be fully enumerated and tested. ML
is used only for *advisory* warnings (see jetpilotguard.ml), never to trigger an
override on its own.

This mirrors how real Automatic Ground Collision Avoidance Systems (Auto-GCAS)
are architected: a deterministic, formally-analysable trigger, not a learned
model. JetPilotGuard is a simulation of that pattern, not certified avionics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from jetpilotguard.telemetry import TelemetryPacket

_KNOTS_TO_FPS = 1.68781


@dataclass(frozen=True, slots=True)
class WatchdogConfig:
    """Tunable thresholds for the collision watchdog.

    Defaults are chosen for a fast-jet-like envelope. They are intentionally
    explicit and documented so the trigger surface is auditable.
    """

    # Time-to-impact (s) at or below which an override is commanded, provided
    # the aircraft is actually descending toward terrain.
    tti_override_s: float = 2.5
    # Only treat a descent as a collision threat below this flight-path angle.
    descending_pitch_deg: float = -10.0
    # Hard floor: any steep dive below this altitude triggers regardless of TTI.
    hard_floor_altitude_ft: float = 500.0
    hard_floor_pitch_deg: float = -20.0
    # Vertical-speed floor (ft/s) below which TTI is treated as effectively
    # infinite (avoids divide-by-noise on near-level flight).
    min_vertical_speed_fps: float = 0.1


@dataclass(frozen=True, slots=True)
class WatchdogResult:
    """Outcome of one watchdog evaluation."""

    override: bool
    time_to_impact_s: float
    reason: str


class CollisionWatchdog:
    """Rule-based Auto-GCAS trigger."""

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self.config = config or WatchdogConfig()

    def vertical_speed_fps(self, packet: TelemetryPacket) -> float:
        """Downward vertical speed in ft/s (0 if not descending).

        Uses airspeed projected onto the flight-path angle. Only negative
        pitch (nose-down) contributes to closure with the ground.
        """
        if packet.pitch >= 0.0:
            return 0.0
        airspeed_fps = packet.airspeed * _KNOTS_TO_FPS
        return airspeed_fps * math.sin(math.radians(abs(packet.pitch)))

    def time_to_impact(self, packet: TelemetryPacket) -> float:
        """Seconds until ground impact at current vertical speed.

        Returns ``math.inf`` when the aircraft is not meaningfully descending.
        """
        v_down = self.vertical_speed_fps(packet)
        if v_down <= self.config.min_vertical_speed_fps:
            return math.inf
        return packet.altitude / v_down

    def evaluate(self, packet: TelemetryPacket) -> WatchdogResult:
        """Decide whether an autonomous override is warranted."""
        cfg = self.config
        tti = self.time_to_impact(packet)

        imminent_tti = (
            tti <= cfg.tti_override_s
            and packet.pitch <= cfg.descending_pitch_deg
        )
        hard_floor = (
            packet.altitude < cfg.hard_floor_altitude_ft
            and packet.pitch <= cfg.hard_floor_pitch_deg
        )
        override = imminent_tti or hard_floor

        if not override:
            reason = "NOMINAL: within terrain-clearance margins"
        elif hard_floor and not math.isfinite(tti):
            reason = (
                f"OVERRIDE (hard floor): altitude {packet.altitude:.0f} ft, "
                f"pitch {packet.pitch:.1f} deg"
            )
        else:
            reason = (
                f"OVERRIDE: TTI {tti:.2f} s, altitude {packet.altitude:.0f} ft, "
                f"pitch {packet.pitch:.1f} deg"
            )

        return WatchdogResult(override=override, time_to_impact_s=tti, reason=reason)
