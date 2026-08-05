# Getting Started in VS Code — Start to End

This guide takes you from a fresh machine to running every part of JetPilotGuard.
No prior experience with these tools assumed. Everything here is free.

---

## 0. Install the basics (one time)

1. **Python 3.11** — https://www.python.org/downloads/ (tick "Add to PATH" on Windows).
2. **VS Code** — https://code.visualstudio.com/
3. **Git** — https://git-scm.com/downloads
4. **VS Code extensions** (open the Extensions panel, the squares icon, and search):
   - *Python* (Microsoft) — running/debugging, environments
   - *Jupyter* (Microsoft) — notebooks inside VS Code
   - *Ruff* (Astral) — inline linting, matches this project
   - *Docker* (Microsoft) — only needed later, for the monitoring stack

> Docker Desktop itself (https://www.docker.com/products/docker-desktop/) is a
> **separate** install and only needed for Phase 4. Skip it for now.

---

## 1. Open the project and make a virtual environment

A virtual environment keeps this project's packages separate from your system.

1. In VS Code: **File -> Open Folder** -> select the `jetpilotguard` folder.
2. Open a terminal: **Terminal -> New Terminal** (or `` Ctrl+` ``).
3. Create and activate the environment:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. VS Code will pop up "We noticed a new virtual environment" — click **Yes** to
   select it as the interpreter. (Or press `Ctrl+Shift+P` ->
   "Python: Select Interpreter" -> pick the `.venv` one.)

---

## 2. Install the project

```bash
pip install --upgrade pip
pip install -e ".[dev,app,mcp,explain,serve,mlops,notebook]"
```

This installs JetPilotGuard plus every optional feature. Takes a minute or two.

---

## 3. Train the models

```bash
python -m scripts.train_model       # per-frame advisory
python -m scripts.train_temporal    # temporal model + comparison
```

You'll see honest held-out metrics print. Model files land in `models/`.

---

## 4. Run things and see it work

**Scripted demo (terminal):**
```bash
python -m scripts.cli_demo
```

**Full evaluation report:**
```bash
python -m scripts.evaluate_system
```

**Interactive cockpit (opens in browser):**
```bash
streamlit run scripts/streamlit_app.py
```

**The REST API service:**
```bash
uvicorn jetpilotguard.io.service:app --reload
```
Then open http://127.0.0.1:8000/docs — an interactive page where you can send
telemetry and see the assessment. This is FastAPI's auto-generated UI.

---

## 5. Run the tests

Either from the terminal:
```bash
pytest -q
```
…or click the **beaker icon** in VS Code's sidebar (Testing panel) to see all
tests with clickable green checkmarks.

---

## 6. Explore the notebooks

Open `notebooks/01_data_and_temporal_model.ipynb` in VS Code. Click **Run All**
(or Shift+Enter cell by cell). If prompted to pick a kernel, choose the `.venv`
one. The second notebook shows SHAP explanations.

---

## 7. Experiment tracking with MLflow

```bash
python -m scripts.train_with_mlflow
mlflow ui
```
Open http://127.0.0.1:5000 to browse and compare training runs — parameters,
metrics, and saved models, all recorded automatically.

---

## 8. Data/model versioning with DVC

```bash
dvc init
dvc repro          # runs the pipeline stages defined in dvc.yaml
dvc metrics show   # shows tracked metrics
```
DVC records the exact commands that produce your models so anyone can reproduce
them.

---

## 9. The monitoring stack (needs Docker Desktop)

Install Docker Desktop first, make sure it's running, then:

```bash
docker compose up --build
```
This starts three services. Open:
- http://localhost:8000/docs — the API
- http://localhost:9090 — Prometheus
- http://localhost:3000 — Grafana (login **admin** / **admin**)

In a second terminal, generate live traffic so the dashboards move:
```bash
python -m scripts.load_generator --rate 5
```

In Grafana, open the pre-loaded **JetPilotGuard Live Monitoring** dashboard.

When done:
```bash
docker compose down        # stops everything, frees memory
```

See `docs/MONITORING.md` for what each panel means.

---

## Troubleshooting

- **"command not found: python"** — try `python3`.
- **VS Code uses the wrong Python** — `Ctrl+Shift+P` -> "Python: Select
  Interpreter" -> choose `.venv`.
- **A package won't import** — make sure the venv is activated (you should see
  `(.venv)` in the terminal prompt) and re-run the `pip install -e` line.
- **Docker command not found** — Docker Desktop must be installed *and running*.
