"""Explainability for the temporal impairment advisory.

A safety-adjacent advisory that a human must trust cannot be a black box. This
module uses SHAP (SHapley Additive exPlanations) to attribute each advisory to
the temporal features that drove it, so a warning can be presented as, e.g.,
"driven by sustained high-g exposure and falling perfusion" rather than an
opaque probability.

SHAP is optional; the model works without it. Import errors degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jetpilotguard.ml.temporal import (
    TemporalImpairmentModel,
    extract_temporal_features,
)

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


@dataclass
class FeatureAttribution:
    feature: str
    value: float
    shap_value: float

    def direction(self) -> str:
        return "increases risk" if self.shap_value > 0 else "decreases risk"


class AdvisoryExplainer:
    """Wraps a trained temporal model with a SHAP TreeExplainer."""

    def __init__(self, model: TemporalImpairmentModel) -> None:
        if shap is None:  # pragma: no cover
            raise RuntimeError(
                "shap is not installed. Install with: pip install shap"
            )
        self._model = model
        self._explainer = shap.TreeExplainer(model.estimator)
        self._names = model.feature_names

    def explain_window(
        self, window: np.ndarray, top_k: int = 5
    ) -> list[FeatureAttribution]:
        """Return the top-k features driving the advisory for one window."""
        feats = extract_temporal_features(window).reshape(1, -1)
        shap_out = self._explainer.shap_values(feats)
        # Normalise across SHAP versions (some return a list per class).
        values = np.asarray(shap_out)
        if values.ndim == 3:            # (classes, samples, features)
            values = values[1]
        row = values[0]
        order = np.argsort(np.abs(row))[::-1][:top_k]
        return [
            FeatureAttribution(
                feature=self._names[i],
                value=float(feats[0, i]),
                shap_value=float(row[i]),
            )
            for i in order
        ]

    def describe(self, window: np.ndarray, top_k: int = 3) -> str:
        """Human-readable one-line explanation of the advisory."""
        attrs = self.explain_window(window, top_k=top_k)
        parts = [
            f"{a.feature} ({a.direction()})"
            for a in attrs
            if abs(a.shap_value) > 1e-6
        ]
        if not parts:
            return "No dominant driver; advisory near baseline."
        return "Advisory driven by: " + ", ".join(parts)
