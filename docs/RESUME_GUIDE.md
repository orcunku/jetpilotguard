# Résumé Keywords & Interview Guide

Everything in this project is real and defensible. Only claim what you can
explain — this guide pairs each keyword with a one-line "why", so you can back
it up when probed.

## Keywords you can honestly put on a résumé

**Languages & core:** Python, NumPy, pandas, object-oriented design, packaging
(`pyproject.toml`, editable installs).

**Machine learning:** scikit-learn, gradient boosting, binary classification,
**time-series / temporal modelling**, feature engineering, train/test split,
cross-validation, probability calibration, ROC-AUC / precision-recall / Brier
score, class imbalance handling, **model evaluation**.

**Explainable AI:** SHAP, feature attribution.

**MLOps:** **MLflow** (experiment tracking), **DVC** (data/model versioning),
**Docker** & Docker Compose (containerisation), **Prometheus** & **Grafana**
(monitoring / observability), **FastAPI** (model serving), CI/CD (GitHub
Actions), reproducible pipelines.

**Engineering practice:** unit & integration testing (pytest), linting (ruff),
REST APIs, Pydantic validation, technical documentation, model cards.

**Domain framing:** safety-critical systems design, human-autonomy interaction,
deterministic + ML hybrid architecture.

## Sample résumé bullets

- Built an end-to-end ML system pairing a deterministic safety override with an
  explainable, **temporally-aware** impairment classifier; proved the temporal
  model beat a per-frame baseline by +0.04 ROC-AUC across 5 seeds on held-out
  data.
- Wrapped the model in a **FastAPI** service instrumented with **Prometheus**
  metrics and a **Grafana** dashboard; containerised the full stack with
  **Docker Compose** (one-command spin-up).
- Added **MLflow** experiment tracking and **DVC** pipelines for reproducible
  training; enforced correctness with 40 **pytest** tests and **GitHub Actions**
  CI across Python 3.10–3.12.

## Questions you should be ready for (and honest answers)

**"Is this real data?"** No — it's simulated, and I designed it deliberately so
the metrics are honest (label noise makes a perfect score impossible). I can
explain exactly how the data is generated and why that makes the ~0.75 AUC
meaningful rather than leaked.

**"Why not just use ML for the override too?"** Anything that can take control
from a human must be auditable and fully testable. ML decision surfaces are
opaque and shift on retraining, so the override is deterministic; ML is confined
to advisories. I enforce that with a test.

**"Why did the temporal model only improve things by 0.04?"** Because the
improvement is *real*, not engineered. The scenarios share a near-identical
final frame, so the snapshot baseline is genuinely handicapped; the temporal
model recovers the trajectory signal. A huge gap would suggest leakage.

**"What does Grafana add that your evaluation script doesn't?"** Different jobs.
The eval script measures *model quality* offline on labelled data. Prometheus/
Grafana monitor the *running service's* operational health over time — latency,
throughput, prediction distribution — which is the basis for drift detection.

**"What would you do next?"** A proper sequence model (LSTM/temporal CNN) instead
of engineered window features; real or high-fidelity simulator data to test
transfer; and alerting rules on the override counter.

## What NOT to claim

- Don't call it "certified" or "airworthy" — it's a simulation.
- Don't say you "evaluated the model with Grafana" — that conflates monitoring
  with evaluation.
- Don't list tools you haven't actually run and can't explain.
