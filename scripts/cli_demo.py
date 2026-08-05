"""Command-line demo cycling through representative flight scenarios.

Run:
    python -m scripts.cli_demo
"""

from __future__ import annotations

from jetpilotguard.engine import JetPilotGuardEngine, StatusLevel
from jetpilotguard.telemetry import TelemetryPacket

SCENARIOS = [
    ("Straight & level cruise",
     dict(altitude=5000, airspeed=350, pitch=0, roll=0,
          g_force=1.0, ppg_amplitude=0.42, gaze_offset_deg=2.0)),
    ("High-G turn, low cerebral perfusion",
     dict(altitude=4000, airspeed=450, pitch=3, roll=45,
          g_force=7.5, ppg_amplitude=0.06, gaze_offset_deg=6.0)),
    ("Spatial disorientation (gaze wandering)",
     dict(altitude=3500, airspeed=300, pitch=0, roll=50,
          g_force=1.8, ppg_amplitude=0.30, gaze_offset_deg=22.0)),
    ("Low-altitude steep dive (collision course)",
     dict(altitude=400, airspeed=450, pitch=-40, roll=0,
          g_force=2.5, ppg_amplitude=0.30, gaze_offset_deg=4.0)),
]

_ICON = {
    StatusLevel.NOMINAL: "[ OK ]",
    StatusLevel.ADVISORY: "[WARN]",
    StatusLevel.CRITICAL_OVERRIDE: "[ !! ]",
}


def main() -> None:
    engine = JetPilotGuardEngine(stateful_filtering=False)
    print("=" * 60)
    print(" JETPILOTGUARD CLI DEMO")
    print("=" * 60)
    for name, kw in SCENARIOS:
        a = engine.process(TelemetryPacket(**kw))
        print(f"\n{_ICON[a.status]} {name}")
        print(f"    Status        : {a.status.value}")
        print(f"    ML impairment : {a.ml_impairment_probability:.1%}")
        tti = a.time_to_impact_s
        print(f"    Time-to-impact: {'inf' if tti == float('inf') else f'{tti:.2f} s'}")
        print(f"    Auto-GCAS     : {'ENGAGED' if a.auto_gcas_engaged else 'inactive'}")
        print(f"    Reason        : {a.reason}")
    print()


if __name__ == "__main__":
    main()
