"""Synthetic dataset generation for the pilot-impairment classifier.

Honesty statement (read this before trusting any metric):

    This data is *simulated*. There is no real flight or physiological data
    here. The generator encodes a plausible but hand-designed relationship
    between flight/bio signals and impairment, then adds realistic structure
    that makes the learning problem non-trivial:

      1. Soft, overlapping decision regions (a logistic risk model), not a hard
         if/else rule. Two pilots with identical telemetry can differ.
      2. Label noise (sensors are imperfect; physiology is noisy).
      3. Correlated, non-uniform feature distributions.

    Because the labels are a *probabilistic* function of the features rather
    than a threshold, a classifier cannot achieve 100% accuracy, and reported
    metrics reflect genuine generalisation rather than memorising a rule. This
    is the difference between a demo that fools you and one that doesn't.

The physiological reasoning behind the risk model:
  * G-LOC (G-induced loss of consciousness) risk rises with sustained high g,
    and is worsened by low cranial PPG amplitude (reduced cerebral perfusion).
  * Spatial disorientation risk rises with large sustained gaze offset,
    especially combined with roll/unusual attitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "g_force",
    "ppg_amplitude",
    "gaze_offset_deg",
    "pitch",
    "roll",
    "altitude",
]
LABEL_COLUMN = "impaired"


def _impairment_logit(
    g_force: np.ndarray,
    ppg: np.ndarray,
    gaze: np.ndarray,
    roll: np.ndarray,
) -> np.ndarray:
    """Log-odds of impairment as a smooth function of physiological load.

    Coefficients are hand-tuned to be plausible, not fitted. The point is a
    smooth, overlapping risk surface -- not a crisp boundary.
    """
    # G-LOC channel: interaction between high g and low perfusion.
    g_term = 1.6 * (g_force - 4.5)
    perfusion_term = 8.0 * (0.22 - ppg)          # low ppg -> higher risk
    gloc = g_term + perfusion_term + 3.0 * np.maximum(g_force - 5.0, 0) * (0.22 - ppg)

    # Disorientation channel: sustained gaze offset, amplified at high roll.
    diso = 0.30 * (gaze - 9.0) + 0.05 * np.abs(roll) * np.maximum(gaze - 8.0, 0) / 10.0

    # Combine channels; the baseline keeps most cruise samples safe while
    # leaving a substantial, learnable positive class (~20-30%).
    return -1.3 + gloc + diso


def generate_dataset(
    n_samples: int = 8_000,
    label_noise: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labelled dataset of flight/bio samples.

    Args:
        n_samples: Number of rows.
        label_noise: Fraction of labels randomly flipped, simulating sensor and
            annotation error. Keeps the Bayes error above zero.
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame with FEATURE_COLUMNS + LABEL_COLUMN.
    """
    rng = np.random.default_rng(seed)

    # Feature distributions: broad, overlapping, loosely realistic.
    g_force = rng.gamma(shape=2.2, scale=1.1, size=n_samples) + 0.8
    g_force = np.clip(g_force, 0.5, 9.0)

    ppg = np.clip(rng.normal(0.32, 0.10, n_samples), 0.02, 0.60)
    gaze = np.clip(np.abs(rng.normal(4.0, 5.0, n_samples)), 0.0, 40.0)
    pitch = np.clip(rng.normal(0.0, 18.0, n_samples), -80.0, 40.0)
    roll = np.clip(rng.normal(0.0, 25.0, n_samples), -120.0, 120.0)
    altitude = np.clip(rng.normal(4000.0, 2500.0, n_samples), 200.0, 12000.0)

    logit = _impairment_logit(g_force, ppg, gaze, roll)
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.uniform(size=n_samples) < prob).astype(int)

    # Flip a fraction of labels to inject irreducible (Bayes) error.
    if label_noise > 0:
        flip = rng.uniform(size=n_samples) < label_noise
        labels[flip] = 1 - labels[flip]

    return pd.DataFrame(
        {
            "g_force": g_force,
            "ppg_amplitude": ppg,
            "gaze_offset_deg": gaze,
            "pitch": pitch,
            "roll": roll,
            "altitude": altitude,
            LABEL_COLUMN: labels,
        }
    )
