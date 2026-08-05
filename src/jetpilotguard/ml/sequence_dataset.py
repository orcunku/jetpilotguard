"""Sequence dataset generation for temporal impairment detection.

Motivation
----------
Pilot impairment is a *trajectory*, not a single frame. G-induced loss of
consciousness (G-LOC) has a characteristic onset: sustained high g-load drives
cerebral perfusion down over several seconds before consciousness is lost.
Spatial disorientation builds as gaze wanders over a window of time. A
per-frame classifier is blind to this — it sees a snapshot and cannot know
whether a high-g reading is a momentary spike or the fifth second of a
sustained pull.

This module simulates short windows of telemetry (a few seconds at a fixed
sample rate) with temporally-coherent dynamics, and labels the *window* as
impaired based on the trajectory, not any single frame. That gives a temporal
model something real to exploit that a per-frame model structurally cannot.

Honesty note: still fully synthetic. See docs/DESIGN.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-frame channels present at every timestep of a window.
FRAME_CHANNELS = [
    "g_force",
    "ppg_amplitude",
    "gaze_offset_deg",
    "pitch",
    "roll",
    "altitude",
]
LABEL_COLUMN = "impaired"


def _simulate_window(
    rng: np.random.Generator, window_len: int, dt: float
) -> tuple[np.ndarray, int]:
    """Simulate one coherent telemetry window and its trajectory label.

    Returns:
        frames: array of shape (window_len, len(FRAME_CHANNELS)).
        label: 1 if the *trajectory* indicates impairment, else 0.
    """
    # Pick a scenario archetype with realistic per-window dynamics.
    archetype = rng.choice(
        ["cruise", "sustained_high_g", "transient_g_spike", "disorientation"],
        p=[0.45, 0.22, 0.15, 0.18],
    )

    t = np.arange(window_len)

    # Baselines
    altitude = np.full(window_len, rng.uniform(2000, 8000))
    pitch = np.clip(rng.normal(0, 4, window_len), -30, 30)
    roll = np.clip(rng.normal(0, 8, window_len), -60, 60)

    # Design principle: the FINAL frame is deliberately made ambiguous across
    # archetypes so that a snapshot classifier cannot separate danger from
    # safety. The discriminating information lives in the *path* to that frame
    # (how long high-g was sustained, whether perfusion was already depressed
    # before the pull, whether gaze was trending out). This is what forces a
    # temporal model to win.
    if archetype == "cruise":
        g_force = np.clip(rng.normal(1.4, 0.4, window_len), 0.6, 3.5)
        ppg = np.clip(rng.normal(0.38, 0.05, window_len), 0.1, 0.55)
        gaze = np.clip(np.abs(rng.normal(3.0, 2.0, window_len)), 0, 18)

    elif archetype == "sustained_high_g":
        # High-g held for MOST of the window -> large cumulative exposure.
        # DANGER. Both g and ppg are steered to a COMMON endpoint shared with
        # the transient case, so the final frame is (near) uninformative and
        # the danger is legible only from the sustained middle of the window.
        hold_g = rng.uniform(5.5, 8.0)
        g_force = np.clip(np.full(window_len, hold_g) + rng.normal(0, 0.3, window_len), 1, 9)
        ppg = np.clip(rng.normal(0.22, 0.03, window_len), 0.02, 0.55)
        gaze = np.clip(np.abs(rng.normal(4.0, 2.5, window_len)), 0, 20)

    elif archetype == "transient_g_spike":
        # Low-g for most of the window with a brief spike -> NO sustained
        # exposure. SAFE. But its final frame is forced to match the dangerous
        # case's endpoint below, making the snapshot a genuine coin-flip.
        g_force = np.clip(rng.normal(1.5, 0.3, window_len), 1, 9)
        spike_at = rng.integers(2, window_len - 4)
        g_force[spike_at] = rng.uniform(6.0, 8.0)
        ppg = np.clip(rng.normal(0.34, 0.03, window_len), 0.1, 0.55)
        gaze = np.clip(np.abs(rng.normal(3.0, 2.0, window_len)), 0, 18)

    # Force the two confusable archetypes to share a COMMON final frame so the
    # last-frame snapshot cannot separate them. All discriminating signal is
    # therefore in the trajectory, not the endpoint.
    if archetype in ("sustained_high_g", "transient_g_spike"):
        common_g = rng.uniform(3.0, 4.0)
        common_ppg = rng.uniform(0.24, 0.30)
        common_gaze = rng.uniform(3.0, 6.0)
        g_force[-2:] = np.clip(common_g + rng.normal(0, 0.15, 2), 1, 9)
        ppg[-2:] = np.clip(common_ppg + rng.normal(0, 0.01, 2), 0.02, 0.55)
        gaze[-2:] = np.clip(common_gaze + rng.normal(0, 0.5, 2), 0, 40)

    else:  # disorientation
        # Gaze drifts monotonically outward, but we clip the LAST frame back to
        # a moderate value so the snapshot understates the trend. DANGER lives
        # in the slope, not the final value.
        drift = np.linspace(rng.uniform(3, 6), rng.uniform(20, 32), window_len)
        gaze = np.clip(drift + rng.normal(0, 2, window_len), 0, 40)
        gaze[-2:] = np.clip(rng.uniform(8, 13, 2), 0, 40)  # deceptive late dip
        roll = np.clip(rng.normal(0, 30, window_len), -70, 70)
        g_force = np.clip(rng.normal(1.8, 0.3, window_len), 1, 4)
        ppg = np.clip(rng.normal(0.35, 0.04, window_len), 0.1, 0.5)

    frames = np.stack([g_force, ppg, gaze, pitch, roll, altitude], axis=1)

    # --- Trajectory label: depends on the WHOLE window, not the last frame ---
    # Cumulative G-LOC exposure: time spent at high g while perfusion is low.
    gloc_exposure = np.sum(np.maximum(g_force - 4.5, 0) * np.clip(0.25 - ppg, 0, None)) * dt
    # Sustained large-gaze integral over the window.
    diso_exposure = np.sum(np.maximum(gaze - 12.0, 0)) * dt
    # Gaze trend across the window (rising = disorientation building).
    gaze_trend = float(np.polyfit(t, gaze, 1)[0])

    risk = 4.5 * gloc_exposure + 0.06 * diso_exposure + 2.5 * max(gaze_trend, 0)
    prob = 1.0 / (1.0 + np.exp(-(risk - 1.0)))
    label = int(rng.uniform() < prob)

    return frames, label


def generate_sequence_dataset(
    n_windows: int = 6_000,
    window_len: int = 15,
    dt: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a set of telemetry windows and trajectory labels.

    Args:
        n_windows: number of windows.
        window_len: frames per window (e.g. 15 frames * 0.2 s = 3 s window).
        dt: seconds between frames.
        seed: RNG seed.

    Returns:
        X: array of shape (n_windows, window_len, n_channels).
        y: array of shape (n_windows,).
    """
    rng = np.random.default_rng(seed)
    xs = np.empty((n_windows, window_len, len(FRAME_CHANNELS)), dtype=float)
    ys = np.empty(n_windows, dtype=int)
    for i in range(n_windows):
        frames, label = _simulate_window(rng, window_len, dt)
        xs[i] = frames
        ys[i] = label
    return xs, ys


def last_frames(X: np.ndarray) -> pd.DataFrame:
    """Extract only the final frame of each window as per-frame features.

    Used to give the per-frame baseline model a fair shot on the same data:
    it sees the most recent snapshot, exactly what a frame classifier would
    have at decision time.
    """
    final = X[:, -1, :]
    return pd.DataFrame(final, columns=FRAME_CHANNELS)
