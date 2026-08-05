"""Tests for the temporal modelling pipeline.

The headline test asserts that the temporal model beats the per-frame baseline
on identical held-out windows -- the whole justification for the approach.
"""

import numpy as np
import pytest

from jetpilotguard.ml.sequence_dataset import (
    FRAME_CHANNELS,
    generate_sequence_dataset,
)
from jetpilotguard.ml.temporal import (
    TEMPORAL_FEATURE_NAMES,
    compare_temporal_vs_frame,
    extract_temporal_features,
)


def test_sequence_shapes():
    X, y = generate_sequence_dataset(n_windows=100, window_len=15, seed=0)
    assert X.shape == (100, 15, len(FRAME_CHANNELS))
    assert y.shape == (100,)
    assert set(np.unique(y)).issubset({0, 1})


def test_sequence_reproducible():
    a_x, a_y = generate_sequence_dataset(50, seed=1)
    b_x, b_y = generate_sequence_dataset(50, seed=1)
    assert np.array_equal(a_x, b_x)
    assert np.array_equal(a_y, b_y)


def test_feature_vector_length_matches_names():
    X, _ = generate_sequence_dataset(5, window_len=15, seed=0)
    feats = extract_temporal_features(X[0])
    assert feats.shape == (len(TEMPORAL_FEATURE_NAMES),)


def test_temporal_features_capture_slope():
    # A window with rising gaze should have a positive gaze slope feature.
    window = np.zeros((15, len(FRAME_CHANNELS)))
    window[:, 0] = 1.0        # g_force
    window[:, 1] = 0.4        # ppg
    window[:, 2] = np.linspace(0, 30, 15)   # gaze rising
    feats = extract_temporal_features(window)
    gaze_slope_idx = TEMPORAL_FEATURE_NAMES.index("gaze_offset_deg_slope")
    assert feats[gaze_slope_idx] > 0


def test_temporal_beats_frame_baseline_on_average():
    """Core claim: modelling the trajectory beats the latest-frame snapshot.

    Stated honestly: the uplift is real but modest, so on any single small
    sample it can fall within noise. We therefore assert the claim the way it
    should be defended -- as a mean advantage across several seeds at a
    realistic sample size, not a guaranteed win on every tiny run.
    """
    uplifts = []
    for seed in range(3):
        _, report = compare_temporal_vs_frame(n_windows=4000, seed=seed)
        uplifts.append(report.temporal_roc_auc - report.frame_roc_auc)
        # Sanity: both are real classifiers, not degenerate.
        assert report.frame_roc_auc > 0.55
        assert report.temporal_roc_auc > 0.6

    mean_uplift = sum(uplifts) / len(uplifts)
    # The mean advantage should be clearly positive.
    assert mean_uplift > 0.01, f"mean uplift only {mean_uplift:.4f}"
    # And it should win in every run at this sample size.
    assert all(u > 0 for u in uplifts)


def test_model_persistence(tmp_path):
    model, _ = compare_temporal_vs_frame(n_windows=1500, seed=0)
    p = tmp_path / "m.joblib"
    model.save(p)
    from jetpilotguard.ml.temporal import TemporalImpairmentModel

    loaded = TemporalImpairmentModel.load(p)
    X, _ = generate_sequence_dataset(3, seed=0)
    assert 0.0 <= loaded.predict_proba_window(X[0]) <= 1.0


def test_explainer_optional_import():
    """Explainer should work if shap is present, and raise clearly if not."""
    pytest.importorskip("shap")
    from jetpilotguard.ml.explain import AdvisoryExplainer

    model, _ = compare_temporal_vs_frame(n_windows=1500, seed=0)
    explainer = AdvisoryExplainer(model)
    X, _ = generate_sequence_dataset(3, seed=0)
    attrs = explainer.explain_window(X[0], top_k=3)
    assert len(attrs) == 3
    assert all(hasattr(a, "shap_value") for a in attrs)
    assert isinstance(explainer.describe(X[0]), str)
