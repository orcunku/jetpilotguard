"""Train the temporal impairment model and report the head-to-head comparison
against the per-frame baseline.

Usage:
    python -m scripts.train_temporal

Writes:
    models/temporal_impairment.joblib
    models/temporal_comparison.json
"""

from __future__ import annotations

from pathlib import Path

from jetpilotguard.ml.temporal import compare_temporal_vs_frame

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    print("Training temporal model and comparing to per-frame baseline...")
    model, report = compare_temporal_vs_frame(n_windows=6000, window_len=15)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / "temporal_impairment.joblib")
    (MODELS_DIR / "temporal_comparison.json").write_text(report.to_json())

    print("\n=== Temporal vs per-frame (held-out) ===")
    print(report.summary())

    if report.auc_uplift > 0:
        print(
            f"\nTemporal modelling improves ROC-AUC by {report.auc_uplift:+.3f} "
            "on identical held-out windows."
        )
    else:  # pragma: no cover
        print(
            "\nNo temporal uplift on this run -- investigate before claiming one."
        )
    print(f"\nSaved -> {MODELS_DIR / 'temporal_impairment.joblib'}")


if __name__ == "__main__":
    main()
