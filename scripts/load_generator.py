"""Stream synthetic telemetry at the running service so the dashboards move.

Usage (service must be running):
    python -m scripts.load_generator --url http://127.0.0.1:8000 --rate 5

Teaching note
-------------
A monitoring dashboard is boring if nothing is happening. This script plays the
role of "the aircraft", sending a realistic mix of nominal, advisory, and
override scenarios so Prometheus has a live signal to graph. It cycles through
scenario archetypes with some randomness so the graphs look organic.
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

SCENARIOS = [
    # (weight, name, kwargs)
    (60, "cruise", dict(altitude=5000, airspeed=350, pitch=0, roll=0,
                        g_force=1.1, ppg_amplitude=0.42, gaze_offset_deg=2.0)),
    (20, "high_g", dict(altitude=4000, airspeed=450, pitch=3, roll=45,
                        g_force=7.5, ppg_amplitude=0.06, gaze_offset_deg=6.0)),
    (10, "disorientation", dict(altitude=3500, airspeed=300, pitch=0, roll=50,
                                g_force=1.8, ppg_amplitude=0.30, gaze_offset_deg=22.0)),
    (10, "dive", dict(altitude=400, airspeed=500, pitch=-40, roll=0,
                      g_force=2.5, ppg_amplitude=0.30, gaze_offset_deg=4.0)),
]


def _jitter(kwargs: dict) -> dict:
    """Add small random noise so successive packets are not identical."""
    out = dict(kwargs)
    out["altitude"] = max(100.0, out["altitude"] + random.uniform(-200, 200))
    out["g_force"] = max(0.5, out["g_force"] + random.uniform(-0.3, 0.3))
    out["ppg_amplitude"] = min(1.0, max(0.02, out["ppg_amplitude"] + random.uniform(-0.03, 0.03)))
    out["gaze_offset_deg"] = max(0.0, out["gaze_offset_deg"] + random.uniform(-1, 1))
    return out


def _pick() -> dict:
    weights = [w for w, _, _ in SCENARIOS]
    _, _name, kwargs = random.choices(SCENARIOS, weights=weights, k=1)[0]
    return _jitter(kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--rate", type=float, default=5.0, help="packets/sec")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to run (0 = forever)")
    args = parser.parse_args()

    interval = 1.0 / args.rate
    deadline = time.time() + args.duration if args.duration else None
    sent = 0

    print(f"Streaming to {args.url}/assess at {args.rate}/s (Ctrl+C to stop)")
    try:
        with httpx.Client(timeout=5.0) as client:
            while deadline is None or time.time() < deadline:
                try:
                    client.post(f"{args.url}/assess", json=_pick())
                    sent += 1
                    if sent % 20 == 0:
                        print(f"  sent {sent} packets")
                except httpx.HTTPError as exc:
                    print(f"  request failed: {exc}")
                time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\nStopped after {sent} packets.")


if __name__ == "__main__":
    main()
