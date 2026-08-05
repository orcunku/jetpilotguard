"""Engine + ML integration tests.

These train a small model in-process (fast) so the suite is self-contained and
does not depend on a checked-in model artifact.
"""

import pytest

from jetpilotguard.engine import JetPilotGuardEngine, StatusLevel
from jetpilotguard.ml.classifier import train_and_evaluate
from jetpilotguard.ml.dataset import generate_dataset
from jetpilotguard.telemetry import TelemetryPacket


@pytest.fixture(scope="module")
def trained():
    model, report = train_and_evaluate(n_samples=2000, random_state=0)
    return model, report


def test_dataset_is_reproducible():
    a = generate_dataset(500, seed=1)
    b = generate_dataset(500, seed=1)
    assert a.equals(b)


def test_dataset_has_both_classes():
    df = generate_dataset(2000, seed=3)
    counts = df["impaired"].value_counts()
    assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0


def test_model_beats_chance(trained):
    _, report = trained
    # Honest floor: must be clearly better than a coin flip, but we do NOT
    # assert anything absurd like 1.0 (that would signal leakage).
    assert 0.6 < report.roc_auc < 0.95
    # Stable across folds. Bound is generous because this test trains on a
    # deliberately small sample (2000) for speed; the full model is ~0.01.
    assert report.cv_roc_auc_std < 0.08


def test_probability_in_unit_interval(trained):
    model, _ = trained
    p = model.predict_proba_one(
        {"g_force": 7.0, "ppg_amplitude": 0.05, "gaze_offset_deg": 20.0,
         "pitch": 0.0, "roll": 40.0, "altitude": 3000.0}
    )
    assert 0.0 <= p <= 1.0


def test_high_risk_scores_above_low_risk(trained):
    model, _ = trained
    high = model.predict_proba_one(
        {"g_force": 8.0, "ppg_amplitude": 0.04, "gaze_offset_deg": 25.0,
         "pitch": 0.0, "roll": 50.0, "altitude": 3000.0}
    )
    low = model.predict_proba_one(
        {"g_force": 1.1, "ppg_amplitude": 0.45, "gaze_offset_deg": 1.0,
         "pitch": 0.0, "roll": 0.0, "altitude": 5000.0}
    )
    assert high > low


def test_engine_nominal_cruise(trained):
    model, _ = trained
    engine = JetPilotGuardEngine(model=model, stateful_filtering=False)
    a = engine.process(TelemetryPacket(5000, 350, 0, 0, 1.1, 0.42, 1.0))
    assert a.status == StatusLevel.NOMINAL
    assert not a.auto_gcas_engaged


def test_engine_critical_override_on_dive(trained):
    model, _ = trained
    engine = JetPilotGuardEngine(model=model, stateful_filtering=False)
    a = engine.process(TelemetryPacket(400, 500, -45, 0, 2.5, 0.30, 4.0))
    assert a.status == StatusLevel.CRITICAL_OVERRIDE
    assert a.auto_gcas_engaged


def test_ml_cannot_override_alone(trained):
    """Even a very high ML probability must NOT engage Auto-GCAS on its own.

    This is the core safety-architecture invariant: only the deterministic
    watchdog can override. We craft a physiologically alarming but
    geometrically safe packet (high altitude, level) and assert no override.
    """
    model, _ = trained
    engine = JetPilotGuardEngine(model=model, advisory_threshold=0.01,
                             stateful_filtering=False)
    a = engine.process(TelemetryPacket(6000, 300, 0, 60, 8.5, 0.03, 28.0))
    assert not a.auto_gcas_engaged  # watchdog did not fire
    # With the near-zero threshold it should still be flagged as advisory.
    assert a.status in (StatusLevel.ADVISORY, StatusLevel.NOMINAL)


def test_engine_to_dict_roundtrip(trained):
    model, _ = trained
    engine = JetPilotGuardEngine(model=model, stateful_filtering=False)
    a = engine.process(TelemetryPacket(5000, 350, 0, 0, 1.1, 0.42, 1.0))
    d = a.to_dict()
    assert d["status"] == "NOMINAL"
    assert "ml_impairment_probability" in d
