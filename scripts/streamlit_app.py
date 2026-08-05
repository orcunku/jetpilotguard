"""Streamlit cockpit HUD for JetPilotGuard.

Run:
    streamlit run scripts/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from jetpilotguard.engine import JetPilotGuardEngine, StatusLevel
from jetpilotguard.telemetry import TelemetryPacket

st.set_page_config(
    page_title="JetPilotGuard Cockpit HUD",
    page_icon="\u2708\ufe0f",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading JetPilotGuard safety engine...")
def get_engine() -> JetPilotGuardEngine:
    # One-shot scoring in a UI: filters should not carry state between the
    # user's independent slider positions.
    return JetPilotGuardEngine(stateful_filtering=False)


engine = get_engine()

st.title("\u2708\ufe0f JetPilotGuard \u2014 Cockpit HUD")
st.caption(
    "Simulated human-autonomy safety watchdog. Deterministic Auto-GCAS + ML "
    "pilot-impairment advisory. All data is synthetic."
)

with st.sidebar:
    st.header("Flight kinematics")
    altitude = st.slider("Altitude (ft)", 100.0, 10000.0, 2500.0, 50.0)
    airspeed = st.slider("Airspeed (kt)", 100.0, 800.0, 350.0, 10.0)
    pitch = st.slider("Pitch (deg)", -90.0, 90.0, 0.0, 1.0)
    roll = st.slider("Roll (deg)", -180.0, 180.0, 0.0, 1.0)
    g_force = st.slider("G-force", 0.5, 9.0, 1.2, 0.1)

    st.header("Pilot physiology")
    ppg = st.slider("Cranial PPG amplitude", 0.0, 1.0, 0.35, 0.01)
    gaze = st.slider("Gaze offset (deg)", 0.0, 40.0, 2.0, 0.5)

assessment = engine.process(
    TelemetryPacket(altitude, airspeed, pitch, roll, g_force, ppg, gaze)
)

_BANNER = {
    StatusLevel.CRITICAL_OVERRIDE: ("#D32F2F", "CRITICAL OVERRIDE \u2014 AUTO-GCAS ENGAGED"),
    StatusLevel.ADVISORY: ("#F57C00", "ADVISORY \u2014 pilot impairment likely"),
    StatusLevel.NOMINAL: ("#388E3C", "NOMINAL \u2014 all clear"),
}
color, text = _BANNER[assessment.status]
st.markdown(
    f"<div style='background:{color};padding:14px;border-radius:8px;"
    f"text-align:center;color:white;font-size:20px;font-weight:700;'>"
    f"{text}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
tti = assessment.time_to_impact_s
c1.metric("Time-to-impact", "\u221e" if tti == float("inf") else f"{tti:.2f} s")
c2.metric("ML impairment probability", f"{assessment.ml_impairment_probability:.1%}")
c3.metric("Auto-GCAS", "ACTIVE" if assessment.auto_gcas_engaged else "inactive")

st.divider()
left, right = st.columns([2, 1])

with left:
    st.subheader("Attitude indicator")
    sky_pct = int(np.clip(50 + (pitch / 90.0) * 50, 0, 100))
    roll_vis = float(np.clip(roll, -90, 90))
    st.markdown(
        f"<div style='border:3px solid #444;border-radius:10px;overflow:hidden;"
        f"height:240px;position:relative;"
        f"background:linear-gradient(180deg,#1e88e5 {sky_pct}%,#8d6e63 {sky_pct}%);'>"
        f"<div style='position:absolute;top:50%;left:20%;width:60%;height:3px;"
        f"background:yellow;transform:translateY(-50%) rotate({-roll_vis}deg);'></div>"
        f"<div style='position:absolute;top:50%;left:50%;width:14px;height:14px;"
        f"border:3px solid #0f0;border-radius:50%;transform:translate(-50%,-50%);'>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

with right:
    st.subheader("Diagnostics")
    st.info(assessment.reason)
    st.write("Watchdog override:", assessment.watchdog_override)
    st.write("ML advisory flag:", assessment.ml_advisory)
    st.write(f"Filtered altitude: {assessment.filtered.altitude:.0f} ft")
    st.write(f"Filtered g-force: {assessment.filtered.g_force:.2f} g")
