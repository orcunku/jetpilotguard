"""Train the impairment classifier and persist it with its metrics.

Usage:
    python -m scripts.train_model

Writes:
    models/impairment.joblib   -- the calibrated, trained model
    models/metrics.json        -- held-out evaluation metrics
"""

from __future__ import annotations

from pathlib import Path

from jetpilotguard.ml.classifier import train_and_evaluate

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    print("Generating data and training impairment classifier...")
    model, report = train_and_evaluate()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "impairment.joblib"
    metrics_path = MODELS_DIR / "metrics.json"

    model.save(model_path)
    metrics_path.write_text(report.to_json())

    print("\n=== Held-out evaluation ===")
    print(report.summary())
    print(f"\nSaved model  -> {model_path}")
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
