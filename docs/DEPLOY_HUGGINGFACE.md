# Free Live Demo on Hugging Face Spaces

Hugging Face Spaces hosts the Streamlit cockpit demo for free, giving you a
public URL to put on your résumé. No credit card, no server to manage.

## Steps

1. Create a free account at https://huggingface.co
2. Click **New Space** -> choose **Streamlit** as the SDK -> name it `jetpilotguard`.
3. In your Space's files, add:
   - `app.py` — a thin entry point (see below)
   - `requirements.txt` — the deps the Space installs
   - your `src/` folder (so the package is importable)
4. Commit. The Space builds and gives you a public URL like
   `https://huggingface.co/spaces/<you>/jetpilotguard`.

## `app.py` for the Space

Spaces expect an `app.py` at the root. Point it at the existing Streamlit app:

```python
import runpy
runpy.run_path("scripts/streamlit_app.py", run_name="__main__")
```

## `requirements.txt` for the Space

```
streamlit
numpy>=1.24,<2.4
pandas>=2.0,<3.0
scikit-learn>=1.3
joblib>=1.3
```

## Note on the model file

The Streamlit app loads `models/impairment.joblib`. Either commit that file to
the Space, or add a line to `app.py` that trains it on first run:

```python
from pathlib import Path
if not Path("models/impairment.joblib").exists():
    import subprocess; subprocess.run(["python", "-m", "scripts.train_model"])
```

Training takes a few seconds, so on-first-run training is fine for a demo.
