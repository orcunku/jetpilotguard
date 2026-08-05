"""Hugging Face Spaces entry point.

Spaces looks for an `app.py` at the repo root and runs it. This launches the
Gradio demo. The package under src/ must be importable -- see the Space setup
instructions in docs/DEPLOY_HUGGINGFACE.md.
"""

import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable when running from the repo root on Spaces.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Train the model on first boot if it's not present (fast, a few seconds).
if not (Path(__file__).resolve().parent / "models" / "impairment.joblib").exists():
    subprocess.run([sys.executable, "-m", "scripts.train_model"], check=True)

from scripts.gradio_app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
