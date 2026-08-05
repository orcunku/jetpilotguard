"""End-to-end system evaluation for JetPilotGuard.

This deliberately replaces the earlier (flawed) evaluation that generated
"collision" cases by copying the watchdog's own trigger rule and then measured
a 100% trigger rate. That is circular: it proves a rule fires on inputs built
to fire it.

Here the two subsystems are tested honestly and separately:

  1. WATCHDOG (physics): we simulate dives with an *independent* ground-truth
     model -- an object descending at a computed vertical speed, with a
     ground-truth "will hit terrain within the reaction window" label derived
     from kinematics, NOT from the watchdog's thresholds. We then report
     detection rate, false-alarm rate, and the lead time the watchdog provides.

  2. ML ADVISORY: metrics come straight from the held-out test split in
     training (models/metrics.json), which the model never saw.

  3. LATENCY: per-packet processing time, since this is a "real-time" system.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from jetpilotguard.engine import JetPilotGuardEngine
from jetpilotguard.safety.watchdog import CollisionWatchdog
from jetpilotguard.telemetry import TelemetryPacket

_KNOTS_TO_FPS = 1.68781
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _independent_dive_truth(
    altitude_ft: float,
    airspeed_kt: float,
    pitch_deg: float,
    reaction_window_s: float,
) -> bool:
    """Ground truth computed from physics, independent of the watchdog rule.

    True if, continuing on the current ballistic descent, the aircraft reaches
    the ground within ``reaction_window_s``. This is a genuinely separate model
    from the watchdog's TTI/hard-floor thresholds.
    """
    if pitch_deg >= 0:
        return False
    v_down = airspeed_kt * _KNOTS_TO_FPS * math.sin(math.radians(abs(pitch_deg)))
    if v_down <= 0:
        return False
    return (altitude_ft / v_down) <= reaction_window_s


def evaluate_watchdog(
    n: int = 5_000, reaction_window_s: float = 3.0, seed: int = 7
) -> dict:
    """Test the watchdog against independent physics ground truth."""
    rng = np.random.default_rng(seed)
    watchdog = CollisionWatchdog()

    tp = fp = tn = fn = 0
    lead_times: list[float] = []

    for _ in range(n):
        # Broad, uncorrelated sampling across the flight envelope so that
        # positives and negatives are NOT constructed from the trigger rule.
        altitude = float(rng.uniform(100.0, 8000.0))
        airspeed = float(rng.uniform(120.0, 650.0))
        pitch = float(rng.uniform(-80.0, 20.0))
        packet = TelemetryPacket(
            altitude=altitude,
            airspeed=airspeed,
            pitch=pitch,
            roll=float(rng.uniform(-30, 30)),
            g_force=float(rng.uniform(0.8, 6.0)),
            ppg_amplitude=float(rng.uniform(0.1, 0.5)),
            gaze_offset_deg=float(rng.uniform(0.0, 15.0)),
        )
        truth = _independent_dive_truth(
            altitude, airspeed, pitch, reaction_window_s
        )
        fired = watchdog.evaluate(packet).override

        if fired and truth:
            tp += 1
            tti = watchdog.time_to_impact(packet)
            if math.isfinite(tti):
                lead_times.append(tti)
        elif fired and not truth:
            fp += 1
        elif not fired and truth:
            fn += 1
        else:
            tn += 1

    detection_rate = tp / (tp + fn) if (tp + fn) else float("nan")
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "reaction_window_s": reaction_window_s,
        "samples": n,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "median_lead_time_s": (
            statistics.median(lead_times) if lead_times else float("nan")
        ),
    }


def evaluate_latency(n: int = 3_000, seed: int = 11) -> dict:
    """Measure per-packet processing latency of the full engine."""
    rng = np.random.default_rng(seed)
    engine = JetPilotGuardEngine(stateful_filtering=False)

    latencies_ms: list[float] = []
    for _ in range(n):
        packet = TelemetryPacket(
            altitude=float(rng.uniform(200, 8000)),
            airspeed=float(rng.uniform(120, 600)),
            pitch=float(rng.uniform(-60, 20)),
            roll=float(rng.uniform(-40, 40)),
            g_force=float(rng.uniform(0.8, 8.0)),
            ppg_amplitude=float(rng.uniform(0.05, 0.55)),
            gaze_offset_deg=float(rng.uniform(0, 30)),
        )
        start = perf_counter()
        engine.process(packet)
        latencies_ms.append((perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    return {
        "samples": n,
        "mean_ms": statistics.fmean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": latencies_ms[int(0.95 * n)],
        "p99_ms": latencies_ms[int(0.99 * n)],
        "max_ms": latencies_ms[-1],
    }


def _load_ml_metrics() -> dict | None:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    print("=" * 62)
    print(" JETPILOTGUARD SYSTEM EVALUATION")
    print("=" * 62)

    print("\n[1] Deterministic watchdog vs. INDEPENDENT physics ground truth")
    wd = evaluate_watchdog()
    print(f"    Samples            : {wd['samples']}")
    print(f"    Reaction window    : {wd['reaction_window_s']} s")
    print(f"    Detection rate     : {wd['detection_rate']:.1%}")
    print(f"    False-alarm rate   : {wd['false_alarm_rate']:.1%}")
    print(f"    Median lead time   : {wd['median_lead_time_s']:.2f} s")
    print(
        f"    Confusion          : TP={wd['true_positive']} "
        f"FP={wd['false_positive']} TN={wd['true_negative']} "
        f"FN={wd['false_negative']}"
    )

    print("\n[2] ML impairment advisory (held-out test split from training)")
    ml = _load_ml_metrics()
    if ml is None:
        print("    No metrics.json found. Run: python -m scripts.train_model")
    else:
        print(f"    ROC-AUC            : {ml['roc_auc']:.3f}")
        print(f"    Average precision  : {ml['average_precision']:.3f}")
        print(
            f"    Precision/Recall/F1: {ml['precision']:.3f} / "
            f"{ml['recall']:.3f} / {ml['f1']:.3f}"
        )
        print(f"    Brier score        : {ml['brier_score']:.3f}")
        print(
            f"    CV ROC-AUC         : {ml['cv_roc_auc_mean']:.3f} "
            f"+/- {ml['cv_roc_auc_std']:.3f}"
        )

    print("\n[3] Temporal model vs per-frame baseline (trajectory matters)")
    temporal_path = MODELS_DIR / "temporal_comparison.json"
    if temporal_path.exists():
        tc = json.loads(temporal_path.read_text())
        print(f"    Per-frame ROC-AUC  : {tc['frame_roc_auc']:.3f}")
        print(f"    Temporal ROC-AUC   : {tc['temporal_roc_auc']:.3f}")
        print(f"    Uplift             : {tc['auc_uplift']:+.3f}")
    else:
        print("    Not found. Run: python -m scripts.train_temporal")

    print("\n[4] Real-time latency (full engine, per packet)")
    lat = evaluate_latency()
    print(f"    Mean / Median      : {lat['mean_ms']:.3f} / {lat['median_ms']:.3f} ms")
    print(f"    p95 / p99 / max    : {lat['p95_ms']:.3f} / "
          f"{lat['p99_ms']:.3f} / {lat['max_ms']:.3f} ms")

    print("\n" + "=" * 62)
    print(" Note: all figures are on SIMULATED data. See docs/DESIGN.md.")
    print("=" * 62)


if __name__ == "__main__":
    main()
