"""Train both models with MLflow experiment tracking.

Usage:
    python -m scripts.train_with_mlflow
    mlflow ui          # then open http://127.0.0.1:5000 to browse runs

Teaching note
-------------
MLflow records each training run: its parameters (n_samples, seed...), its
metrics (ROC-AUC, F1...), and the model artifact itself. Instead of numbers
scrolling past in the terminal and vanishing, you get a searchable history you
can compare across runs -- the foundation of reproducible ML experimentation.

Everything is stored locally in an ./mlruns folder. No account, no cloud, free.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from jetpilotguard.ml.classifier import train_and_evaluate
from jetpilotguard.ml.temporal import compare_temporal_vs_frame

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _log_per_frame(n_samples: int, seed: int) -> None:
    with mlflow.start_run(run_name="per_frame_classifier"):
        mlflow.log_params({"model": "per_frame", "n_samples": n_samples, "seed": seed})
        model, report = train_and_evaluate(n_samples=n_samples, random_state=seed)
        mlflow.log_metrics({
            "roc_auc": report.roc_auc,
            "average_precision": report.average_precision,
            "f1": report.f1,
            "brier_score": report.brier_score,
            "cv_roc_auc_mean": report.cv_roc_auc_mean,
        })
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(MODELS_DIR / "impairment.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "impairment.joblib"))
        print(f"[per-frame]  ROC-AUC={report.roc_auc:.3f}  logged to MLflow")


def _log_temporal(n_windows: int, seed: int) -> None:
    with mlflow.start_run(run_name="temporal_vs_frame"):
        mlflow.log_params({"model": "temporal", "n_windows": n_windows, "seed": seed})
        model, report = compare_temporal_vs_frame(n_windows=n_windows, seed=seed)
        mlflow.log_metrics({
            "frame_roc_auc": report.frame_roc_auc,
            "temporal_roc_auc": report.temporal_roc_auc,
            "auc_uplift": report.auc_uplift,
            "temporal_f1": report.temporal_f1,
        })
        model.save(MODELS_DIR / "temporal_impairment.joblib")
        mlflow.log_artifact(str(MODELS_DIR / "temporal_impairment.joblib"))
        print(
            f"[temporal]   uplift={report.auc_uplift:+.3f} "
            f"(frame {report.frame_roc_auc:.3f} -> temporal "
            f"{report.temporal_roc_auc:.3f})  logged to MLflow"
        )


def main() -> None:
    mlflow.set_experiment("jetpilotguard-impairment")
    print("Training with MLflow tracking (stored in ./mlruns)...")
    _log_per_frame(n_samples=8000, seed=42)
    _log_temporal(n_windows=6000, seed=42)
    print("\nDone. Browse runs with:  mlflow ui")


if __name__ == "__main__":
    main()
