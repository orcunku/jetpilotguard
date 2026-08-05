# JetPilotGuard — Design, Limitations, and Roadmap

This document is deliberately candid about what the system does, what it does
*not* do, and where it would go next. In safety engineering, an honest account
of limitations is worth more than an impressive-looking claim.

## 1. Problem framing

Two distinct failure modes threaten a fast-jet pilot:

1. **Controlled flight into terrain (CFIT)** — the aircraft is flyable but on a
   collision course, often due to task saturation or disorientation.
2. **Pilot incapacitation** — G-induced loss of consciousness (G-LOC) or
   spatial disorientation degrades the pilot before any control limit is hit.

JetPilotGuard addresses (1) with a deterministic override and gives *early
warning* of (2) with an ML advisory.

## 2. Why the override layer is not ML

Anything that can seize control from a human must be **auditable**: every
trigger condition should be enumerable and unit-testable, and its behaviour
must be identical every run. A learned model cannot offer that guarantee — its
decision surface is opaque and can shift with retraining. So the override is
pure kinematics:

- **Time-to-impact (TTI):** altitude ÷ vertical speed, where vertical speed is
  airspeed projected onto the (negative) flight-path angle.
- **Hard floor:** any steep dive below a fixed altitude triggers regardless of
  TTI, guarding against the case where TTI math is unreliable at very low
  altitude.

The ML layer feeds situational awareness but is architecturally barred from the
override path. This invariant is enforced by `test_ml_cannot_override_alone`.

## 3. The data is synthetic — and why the metrics are still meaningful

There is no real flight or physiological data here. The generator
(`ml/dataset.py`) encodes a **probabilistic** risk model:

- Impairment probability is a smooth logistic function of g-load, cranial PPG
  amplitude, gaze offset, and roll, with a documented physiological rationale.
- Labels are *sampled* from that probability, not thresholded, so identical
  telemetry can yield different outcomes.
- 5% of labels are flipped to model sensor/annotation error.

Consequences:

- A classifier **cannot** reach 100% — there is an irreducible Bayes error. An
  AUC near 0.75 therefore reflects real learned structure, not a leaked rule.
- This is the single most important difference from a naive demo, where the
  "model" is trained on rule-generated labels and merely re-learns the rule.

**What synthetic data cannot tell us:** whether these features actually predict
impairment in real pilots, or whether the model would transfer. Those are
empirical questions requiring real (or high-fidelity simulator) data.

## 4. Evaluation methodology

Two subsystems, tested separately and honestly:

- **Watchdog** is scored against an *independent* physics model of whether a
  descending aircraft reaches the ground within a reaction window. The test
  inputs are sampled broadly across the envelope, **not** reverse-engineered
  from the watchdog's own thresholds. This is why detection is 80.6%, not a
  suspicious 100%.
- **ML** metrics come only from a held-out test split created before fitting,
  plus 5-fold cross-validation for stability.

An earlier version of this project generated "collision" cases by copying the
watchdog's trigger rule, then reported a 100% trigger rate. That is circular
and has been removed; `scripts/evaluate_system.py` documents the fix inline.

## 5. Known limitations

- **Shallow-descent misses.** The watchdog under-detects slow, shallow descents
  toward terrain (the ~19% miss rate). A radar-altimeter closure-rate term
  would help.
- **No temporal model.** Each packet is scored independently apart from Kalman
  smoothing. Real impairment (e.g. onset of G-LOC) is a *trajectory*; a
  sequence model (e.g. an LSTM or a sliding-window feature set) would likely
  outperform the per-frame classifier.
- **Single airframe assumption.** Thresholds are hand-tuned for one
  fast-jet-like envelope.
- **Calibration is measured, not guaranteed under shift.** The Brier score is
  good on in-distribution data; behaviour under distribution shift is untested.

## 6. Roadmap

Prioritised by impact-to-effort for a portfolio piece:

1. **Temporal features / sequence model** for impairment onset — the clearest
   ML upgrade and a strong talking point.
2. **Feature attribution** (SHAP) on the advisory, so each warning is
   explainable — important for any human-facing safety tool.
3. **Streaming replay harness** — feed a recorded time-series through the engine
   to show behaviour over a full sortie, not just single frames.
4. **Model card + data card** documenting intended use and failure modes.
5. **Radar-altimeter closure term** in the watchdog to close the shallow-descent
   gap.
