"""Temporal impairment model: window-level features that a per-frame model
structurally cannot see, plus a head-to-head comparison against the per-frame
baseline on identical held-out windows.

The claim we want to *prove* (not assert): modelling the trajectory beats
scoring the latest frame. The comparison in ``compare_temporal_vs_frame``
trains both on the same windows and evaluates on the same held-out split.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from jetpilotguard.ml.sequence_dataset import (
    FRAME_CHANNELS,
    generate_sequence_dataset,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

# Names of the engineered temporal features, in a fixed, documented order.
TEMPORAL_FEATURE_NAMES: list[str] = []


def _build_feature_names() -> list[str]:
    names: list[str] = []
    for ch in FRAME_CHANNELS:
        names += [
            f"{ch}_mean",
            f"{ch}_max",
            f"{ch}_min",
            f"{ch}_std",
            f"{ch}_last",
            f"{ch}_slope",       # linear trend across the window
            f"{ch}_delta",       # last - first
        ]
    # Cross-channel, physiologically-motivated temporal features.
    names += [
        "gloc_exposure",         # integral of (g-4.5)+ weighted by low perfusion
        "high_g_fraction",       # fraction of window above 5 g
        "gaze_rising",           # positive gaze slope indicator
        "ppg_falling",           # negative ppg slope magnitude
    ]
    return names


TEMPORAL_FEATURE_NAMES = _build_feature_names()


def extract_temporal_features(window: np.ndarray) -> np.ndarray:
    """Turn one (window_len, n_channels) window into a temporal feature vector.

    These features encode *how signals evolve*, which is exactly the
    information a single-frame snapshot discards.
    """
    n_len = window.shape[0]
    t = np.arange(n_len)
    feats: list[float] = []
    for c in range(window.shape[1]):
        col = window[:, c]
        slope = float(np.polyfit(t, col, 1)[0]) if n_len > 1 else 0.0
        feats += [
            float(col.mean()),
            float(col.max()),
            float(col.min()),
            float(col.std()),
            float(col[-1]),
            slope,
            float(col[-1] - col[0]),
        ]

    g = window[:, 0]
    ppg = window[:, 1]
    gaze = window[:, 2]
    gloc_exposure = float(np.sum(np.maximum(g - 4.5, 0) * np.clip(0.25 - ppg, 0, None)))
    high_g_fraction = float(np.mean(g > 5.0))
    gaze_slope = float(np.polyfit(t, gaze, 1)[0]) if n_len > 1 else 0.0
    ppg_slope = float(np.polyfit(t, ppg, 1)[0]) if n_len > 1 else 0.0
    feats += [
        gloc_exposure,
        high_g_fraction,
        max(gaze_slope, 0.0),
        max(-ppg_slope, 0.0),
    ]
    return np.asarray(feats, dtype=float)


def build_temporal_matrix(X_windows: np.ndarray) -> np.ndarray:
    """Vectorise a batch of windows into a temporal feature matrix."""
    return np.stack([extract_temporal_features(w) for w in X_windows])


@dataclass
class ComparisonReport:
    """Head-to-head metrics: temporal model vs per-frame baseline."""

    n_train: int
    n_test: int
    positive_rate: float
    frame_roc_auc: float
    frame_ap: float
    frame_f1: float
    temporal_roc_auc: float
    temporal_ap: float
    temporal_f1: float

    @property
    def auc_uplift(self) -> float:
        return self.temporal_roc_auc - self.frame_roc_auc

    def to_json(self) -> str:
        d = asdict(self)
        d["auc_uplift"] = self.auc_uplift
        return json.dumps(d, indent=2)

    def summary(self) -> str:
        return (
            "Per-frame baseline vs temporal model (same held-out windows)\n"
            f"  positive rate      : {self.positive_rate:.1%}\n"
            f"  ROC-AUC  frame->temporal : {self.frame_roc_auc:.3f} -> "
            f"{self.temporal_roc_auc:.3f}  (+{self.auc_uplift:.3f})\n"
            f"  Avg-prec frame->temporal : {self.frame_ap:.3f} -> "
            f"{self.temporal_ap:.3f}\n"
            f"  F1       frame->temporal : {self.frame_f1:.3f} -> "
            f"{self.temporal_f1:.3f}\n"
            f"  train/test windows : {self.n_train}/{self.n_test}"
        )


class TemporalImpairmentModel:
    """Gradient-boosted model over engineered temporal features."""

    def __init__(self, estimator, threshold: float = 0.5) -> None:
        self._estimator = estimator
        self.threshold = threshold
        self.feature_names = list(TEMPORAL_FEATURE_NAMES)

    def predict_proba_window(self, window: np.ndarray) -> float:
        feats = extract_temporal_features(window).reshape(1, -1)
        return float(self._estimator.predict_proba(feats)[0, 1])

    @property
    def estimator(self):
        return self._estimator

    def save(self, path: str | Path) -> None:
        if joblib is None:  # pragma: no cover
            raise RuntimeError("joblib required")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": self._estimator,
                "threshold": self.threshold,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TemporalImpairmentModel:
        if joblib is None:  # pragma: no cover
            raise RuntimeError("joblib required")
        blob = joblib.load(path)
        m = cls(blob["estimator"], blob.get("threshold", 0.5))
        m.feature_names = blob["feature_names"]
        return m


def _fit_gb(X, y, seed: int) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.9, random_state=seed,
    )
    clf.fit(X, y)
    return clf


def compare_temporal_vs_frame(
    n_windows: int = 6_000,
    window_len: int = 15,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[TemporalImpairmentModel, ComparisonReport]:
    """Train both models on identical windows; evaluate on identical test split.

    The per-frame baseline is given the *final* frame of each window (what a
    snapshot classifier would have at decision time). The temporal model gets
    engineered window features. Same labels, same split -> fair comparison.
    """
    X_win, y = generate_sequence_dataset(
        n_windows=n_windows, window_len=window_len, seed=seed
    )
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )

    # Per-frame baseline: last frame only.
    X_frame = X_win[:, -1, :]
    frame_clf = _fit_gb(X_frame[idx_tr], y[idx_tr], seed)
    frame_proba = frame_clf.predict_proba(X_frame[idx_te])[:, 1]
    frame_pred = (frame_proba >= 0.5).astype(int)

    # Temporal model: engineered window features.
    X_temp = build_temporal_matrix(X_win)
    temp_clf = _fit_gb(X_temp[idx_tr], y[idx_tr], seed)
    temp_proba = temp_clf.predict_proba(X_temp[idx_te])[:, 1]
    temp_pred = (temp_proba >= 0.5).astype(int)

    report = ComparisonReport(
        n_train=len(idx_tr),
        n_test=len(idx_te),
        positive_rate=float(y[idx_te].mean()),
        frame_roc_auc=float(roc_auc_score(y[idx_te], frame_proba)),
        frame_ap=float(average_precision_score(y[idx_te], frame_proba)),
        frame_f1=float(f1_score(y[idx_te], frame_pred, zero_division=0)),
        temporal_roc_auc=float(roc_auc_score(y[idx_te], temp_proba)),
        temporal_ap=float(average_precision_score(y[idx_te], temp_proba)),
        temporal_f1=float(f1_score(y[idx_te], temp_pred, zero_division=0)),
    )
    return TemporalImpairmentModel(temp_clf), report
