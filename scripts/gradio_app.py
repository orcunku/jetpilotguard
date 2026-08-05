"""Gradio demo for JetPilotGuard.

An interactive web page: move the sliders to set flight kinematics and pilot
physiology, and see the safety assessment update -- ML impairment probability,
time-to-impact, and whether the deterministic watchdog engages Auto-GCAS.

Run locally:
    python -m scripts.gradio_app

Deploy free on Hugging Face Spaces (see docs/DEPLOY_HUGGINGFACE.md): this file
plus the package is all a Space needs.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from jetpilotguard.engine import JetPilotGuardEngine, StatusLevel
from jetpilotguard.telemetry import TelemetryPacket

# Train a model on first run if none exists (keeps the Space self-contained).
_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "impairment.joblib"
if not _MODEL_PATH.exists():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "scripts.train_model"], check=True)

_engine = JetPilotGuardEngine(stateful_filtering=False)

_STATUS_STYLE = {
    StatusLevel.NOMINAL: ("#1b5e20", "NOMINAL - system clear"),
    StatusLevel.ADVISORY: ("#e65100", "ADVISORY - pilot impairment likely"),
    StatusLevel.CRITICAL_OVERRIDE: ("#b71c1c", "CRITICAL OVERRIDE - Auto-GCAS engaged"),
}


def assess(altitude, airspeed, pitch, roll, g_force, ppg_amplitude, gaze_offset):
    """Score one telemetry configuration and return display strings."""
    packet = TelemetryPacket(
        altitude=altitude,
        airspeed=airspeed,
        pitch=pitch,
        roll=roll,
        g_force=g_force,
        ppg_amplitude=ppg_amplitude,
        gaze_offset_deg=gaze_offset,
    )
    a = _engine.process(packet)
    color, label = _STATUS_STYLE[a.status]

    banner = (
        f"<div style='background:{color};color:white;padding:16px;"
        f"border-radius:8px;text-align:center;font-size:20px;"
        f"font-weight:700;'>{label}</div>"
    )

    tti = a.time_to_impact_s
    tti_str = "inf (not descending)" if tti == float("inf") else f"{tti:.2f} s"

    details = (
        f"ML impairment probability : {a.ml_impairment_probability:.1%}\n"
        f"Time-to-impact            : {tti_str}\n"
        f"Auto-GCAS engaged         : {'YES' if a.auto_gcas_engaged else 'no'}\n"
        f"Watchdog override         : {a.watchdog_override}\n"
        f"ML advisory flag          : {a.ml_advisory}\n"
        f"Reason                    : {a.reason}"
    )
    return banner, details


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="JetPilotGuard", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# JetPilotGuard\n"
            "**Simulated human-autonomy flight-safety watchdog.** "
            "A deterministic ground-collision override fused with an ML "
            "pilot-impairment advisory. Move the sliders to explore how the "
            "system responds. *All data is synthetic.*"
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Flight kinematics")
                altitude = gr.Slider(100, 10000, value=2500, step=50, label="Altitude (ft)")
                airspeed = gr.Slider(100, 800, value=350, step=10, label="Airspeed (kt)")
                pitch = gr.Slider(-90, 90, value=0, step=1, label="Pitch (deg)")
                roll = gr.Slider(-180, 180, value=0, step=1, label="Roll (deg)")
                g_force = gr.Slider(0.5, 9.0, value=1.2, step=0.1, label="G-force")
                gr.Markdown("### Pilot physiology")
                ppg = gr.Slider(0.0, 1.0, value=0.35, step=0.01, label="Cranial PPG amplitude")
                gaze = gr.Slider(0.0, 40.0, value=2.0, step=0.5, label="Gaze offset (deg)")
            with gr.Column():
                gr.Markdown("### Assessment")
                banner = gr.HTML()
                details = gr.Textbox(label="Details", lines=8, interactive=False)

        gr.Markdown("### Try a preset scenario")
        with gr.Row():
            btn_cruise = gr.Button("Cruise")
            btn_highg = gr.Button("High-G / low perfusion")
            btn_dive = gr.Button("Low-altitude dive")

        inputs = [altitude, airspeed, pitch, roll, g_force, ppg, gaze]
        outputs = [banner, details]

        # Live update on any slider change.
        for comp in inputs:
            comp.change(assess, inputs=inputs, outputs=outputs)

        # Preset buttons set the sliders (which triggers assessment).
        btn_cruise.click(
            lambda: (5000, 350, 0, 0, 1.0, 0.42, 2.0), outputs=inputs
        )
        btn_highg.click(
            lambda: (4000, 450, 3, 45, 7.5, 0.06, 6.0), outputs=inputs
        )
        btn_dive.click(
            lambda: (400, 500, -40, 0, 2.5, 0.30, 4.0), outputs=inputs
        )

        # Initial render.
        demo.load(assess, inputs=inputs, outputs=outputs)

    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
