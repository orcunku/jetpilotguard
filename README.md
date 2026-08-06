# JetPilotGuard
## 🔗 Live pages

- **[Showcase page](https://orcunku.github.io/jetpilotguard/showcase/)** — project overview
- **[Interactive demo](https://orcunku.github.io/jetpilotguard/demo/)** — try it in your browser

### Interactive cockpit (static demo)
<img width="1307" height="668" alt="index1" src="https://github.com/user-attachments/assets/fd203b5a-a5dc-4b57-9b2f-260e62757546" />

### MlFlow
<img width="1350" height="677" alt="mlflow" src="https://github.com/user-attachments/assets/02058d8c-699a-4d4a-8f00-d2a7afe71400" />

### Prometheus
<img width="1350" height="342" alt="prometheus2" src="https://github.com/user-attachments/assets/803af8f2-c56d-4fa7-ae76-f553fb8be037" />
<img width="1314" height="650" alt="prometheus" src="https://github.com/user-attachments/assets/1b406f94-27f0-4bae-b51f-e4174ebdce0f" />

###  Evaluation Runs
<img width="1311" height="628" alt="evalruns" src="https://github.com/user-attachments/assets/9996d9f5-c8aa-41b0-9799-a0e27f4e7b34" />


### Live monitoring (Prometheus + Grafana)
<img width="1330" height="637" alt="dashboard2" src="https://github.com/user-attachments/assets/c9cd599d-8694-453e-b2a8-2eb7c22a0b6b" />
<img width="1333" height="677" alt="dashboard1" src="https://github.com/user-attachments/assets/fd001d5c-85a8-4343-a17c-0d9323a750c1" />

### Assessment states
<img width="1362" height="637" alt="gradio3" src="https://github.com/user-attachments/assets/d8f7ece3-6d7e-4a58-91c0-7bfe24cb9619" />
<img width="1348" height="652" alt="gradio2" src="https://github.com/user-attachments/assets/1e894b02-564d-4080-8dfd-e2ff8cbf0a8b" />
<img width="1311" height="647" alt="gradio1" src="https://github.com/user-attachments/assets/b37967eb-389e-47b3-8420-e3ac49dbde11" />


**A simulated human-autonomy flight-safety watchdog: a deterministic ground-collision override fused with a machine-learning pilot-impairment advisory.**

> ⚠️ **Scope & honesty statement.** JetPilotGuard is a **research and portfolio demonstration** built on **synthetic data**. It is *not* certified avionics and makes no airworthiness claim. Its value is as a clean, tested example of the architecture real safety systems use: a simple, auditable deterministic core for anything that can override a human, with ML confined to advisory warnings. Every number below is measured on simulated data and reproducible with the commands shown.

---

## Why this design

The central engineering idea is **layered authority**:

| Layer | Type | Can raise | Can override pilot? |
|-------|------|-----------|---------------------|
| Collision watchdog | Deterministic, rule-based kinematics | `CRITICAL_OVERRIDE` | **Yes** |
| Impairment classifier | Calibrated gradient-boosted trees | `ADVISORY` | **No** |

Only the deterministic layer can command an Auto-GCAS override, so the safety-critical trigger is fully enumerable and testable. ML adds earlier, softer warnings about pilot state (G-LOC risk, spatial disorientation) but is never in the override path. This mirrors how real Automatic Ground Collision Avoidance Systems are architected, and it is enforced by a unit test (`test_ml_cannot_override_alone`).

## Results (all on simulated data, reproducible)

**ML impairment advisory** — held-out test split, never seen in training:

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.75 |
| 5-fold CV ROC-AUC | 0.755 ± 0.010 |
| Precision / Recall / F1 | 0.63 / 0.43 / 0.51 |
| Brier score (calibration) | 0.07 |

The dataset injects label noise and uses a smooth, overlapping risk surface, so a perfect score is **impossible by construction** — an AUC near 0.75 reflects genuine learned signal, not a leaked rule.

**Temporal model** — impairment is a *trajectory*, not a snapshot. A windowed model with engineered temporal features (cumulative G-LOC exposure, gaze trend, perfusion rate-of-change) is compared head-to-head against a per-frame baseline on identical held-out windows:

| Model | ROC-AUC |
|-------|---------|
| Per-frame baseline (latest frame only) | ~0.73 |
| **Temporal model** | **~0.77** |

The temporal model wins on all 5 seeds tested (mean uplift **+0.037 AUC**). The scenarios are constructed so the *final frame is deliberately non-discriminative* between the dangerous sustained-g case and the safe transient-spike case — so the advantage can only come from modelling the path, not the endpoint. Each advisory is explainable via SHAP (`jetpilotguard.ml.explain`). See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

**Deterministic watchdog** — tested against an *independent* physics ground-truth (not its own trigger rule):

| Metric | Value |
|--------|-------|
| Detection rate (3 s reaction window) | 80.6% |
| False-alarm rate | 0.1% |
| Median warning lead time | 1.45 s |
| Per-packet latency (p99) | ~3.4 ms |

The ~19% miss rate is a **real, documented limitation** (the current thresholds target steep dives and under-detect shallow descents) rather than a hidden failure — see [docs/DESIGN.md](docs/DESIGN.md).

## Architecture

```
raw telemetry
   │
   ▼  Kalman filtering (altitude, g-force, PPG)
   ├──────────────┐
   ▼              ▼
ML advisory   deterministic watchdog   ← only this can override
(soft warn)   (hard, rule-based)
   └──────┬───────┘
          ▼
   fused status: NOMINAL / ADVISORY / CRITICAL_OVERRIDE
```

## Quickstart

New to this? See **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** for a
complete VS Code walkthrough from zero.

```bash
# 1. Install (editable, with all features)
pip install -e ".[dev,app,mcp,explain,serve,mlops,notebook]"

# 2. Train the models
python -m scripts.train_model      # per-frame advisory + honest metrics
python -m scripts.train_temporal   # temporal model + head-to-head comparison

# 3. See it run
python -m scripts.cli_demo               # scripted scenarios
python -m scripts.evaluate_system        # full honest evaluation report
streamlit run scripts/streamlit_app.py   # interactive cockpit HUD
uvicorn jetpilotguard.io.service:app --reload  # REST API at /docs

# 4. Run the tests
pytest -q
```

## MLOps stack

The project is wrapped in a production-style MLOps toolchain, all free and
runnable on modest hardware:

| Concern | Tool | Entry point |
|---------|------|-------------|
| Model serving | FastAPI + Pydantic | `jetpilotguard.io.service` |
| Observability | Prometheus + Grafana | `docker compose up` |
| Experiment tracking | MLflow | `scripts/train_with_mlflow.py` |
| Data/model versioning | DVC | `dvc.yaml` |
| Containerisation | Docker + Compose | `Dockerfile`, `docker-compose.yml` |
| CI | GitHub Actions | `.github/workflows/ci.yml` |
| Exploration | Jupyter | `notebooks/` |

See **[docs/MONITORING.md](docs/MONITORING.md)** for the observability stack.

## AI-agent integration (MCP)

The engine is exposed over the [Model Context Protocol](https://modelcontextprotocol.io) so an AI copilot can query flight safety programmatically:

```bash
python -m jetpilotguard.io.mcp_server
```

It exposes an `evaluate_flight_safety` tool and an `jetpilotguard://operational-envelope` resource.

## Project layout

```
src/jetpilotguard/
  telemetry.py         # validated data model
  filters.py           # 1-D Kalman filter
  engine.py            # fusion engine (the public API)
  safety/watchdog.py   # deterministic collision override
  ml/dataset.py          # per-frame synthetic data (documented risk model)
  ml/classifier.py       # per-frame train / evaluate / persist / infer
  ml/sequence_dataset.py # temporal windows (trajectory-labelled)
  ml/temporal.py         # temporal features + head-to-head vs baseline
  ml/explain.py          # SHAP attributions for each advisory
  io/mcp_server.py       # MCP integration
scripts/                 # train, train_temporal, evaluate, CLI + Streamlit
tests/                   # 35 tests: units, integration, safety + temporal
docs/DESIGN.md           # design rationale, limitations, roadmap
docs/MODEL_CARD.md       # model card for the temporal advisory
```

## License

MIT.
