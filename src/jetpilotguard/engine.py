"""JetPilotGuard engine: fuses signal filtering, ML advisory, and the
deterministic safety watchdog into a single real-time assessment.

Layering (most important design decision in the project):

    RAW TELEMETRY
        -> Kalman filtering (altitude, g-force, ppg)
        -> ML advisory  (soft, learned, can warn but NOT override)
        -> Watchdog     (hard, rule-based, the ONLY thing that overrides)
        -> Fused status

The ML layer can raise an ADVISORY. Only the deterministic watchdog can raise a
CRITICAL_OVERRIDE. This keeps the safety-critical trigger fully auditable while
still getting value from ML for early, softer warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from jetpilotguard.filters import KalmanFilter1D
from jetpilotguard.ml.classifier import ImpairmentModel
from jetpilotguard.safety.watchdog import CollisionWatchdog, WatchdogConfig
from jetpilotguard.telemetry import TelemetryPacket

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "impairment.joblib"


class StatusLevel(str, Enum):
    NOMINAL = "NOMINAL"
    ADVISORY = "ADVISORY"
    CRITICAL_OVERRIDE = "CRITICAL_OVERRIDE"


@dataclass(frozen=True, slots=True)
class Assessment:
    """Complete result of processing one telemetry packet."""

    status: StatusLevel
    auto_gcas_engaged: bool
    watchdog_override: bool
    ml_advisory: bool
    ml_impairment_probability: float
    time_to_impact_s: float
    reason: str
    filtered: TelemetryPacket

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "auto_gcas_engaged": self.auto_gcas_engaged,
            "watchdog_override": self.watchdog_override,
            "ml_advisory": self.ml_advisory,
            "ml_impairment_probability": self.ml_impairment_probability,
            "time_to_impact_s": self.time_to_impact_s,
            "reason": self.reason,
            "filtered": {
                "altitude": self.filtered.altitude,
                "airspeed": self.filtered.airspeed,
                "pitch": self.filtered.pitch,
                "roll": self.filtered.roll,
                "g_force": self.filtered.g_force,
                "ppg_amplitude": self.filtered.ppg_amplitude,
                "gaze_offset_deg": self.filtered.gaze_offset_deg,
            },
        }


class JetPilotGuardEngine:
    """Real-time fusion engine.

    Args:
        model: A pre-trained ImpairmentModel. If None, one is loaded from
            ``model_path``; if that file is missing a clear error is raised
            telling the user to run the training script (we never silently
            retrain on startup -- that hides model drift and is slow).
        model_path: Where to load the persisted model from.
        advisory_threshold: P(impaired) at or above which an ADVISORY is raised.
        watchdog_config: Optional override of watchdog thresholds.
        stateful_filtering: If True, Kalman filters carry state across packets
            (correct for a real stream). Set False for independent one-shot
            scoring (e.g. batch evaluation of unrelated scenarios).
    """

    def __init__(
        self,
        model: ImpairmentModel | None = None,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        advisory_threshold: float | None = None,
        watchdog_config: WatchdogConfig | None = None,
        stateful_filtering: bool = True,
    ) -> None:
        if model is None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"No trained model at {model_path}. "
                    "Run: python -m scripts.train_model"
                )
            model = ImpairmentModel.load(model_path)
        self._model = model
        # Fall back to the model's own F1-tuned threshold unless overridden.
        self._advisory_threshold = (
            advisory_threshold
            if advisory_threshold is not None
            else model.decision_threshold
        )
        self._watchdog = CollisionWatchdog(watchdog_config)
        self._stateful = stateful_filtering
        self._kf_alt = KalmanFilter1D()
        self._kf_g = KalmanFilter1D()
        self._kf_ppg = KalmanFilter1D()

    def _filter(self, packet: TelemetryPacket) -> TelemetryPacket:
        if not self._stateful:
            self._kf_alt.reset()
            self._kf_g.reset()
            self._kf_ppg.reset()
        return TelemetryPacket(
            altitude=self._kf_alt.update(packet.altitude),
            airspeed=packet.airspeed,
            pitch=packet.pitch,
            roll=packet.roll,
            g_force=self._kf_g.update(packet.g_force),
            ppg_amplitude=self._kf_ppg.update(packet.ppg_amplitude),
            gaze_offset_deg=packet.gaze_offset_deg,
        )

    def process(self, packet: TelemetryPacket) -> Assessment:
        """Assess one telemetry packet through all layers."""
        clean = self._filter(packet)

        # ML advisory (soft). Uses filtered physiological signals.
        p_impaired = self._model.predict_proba_one(
            {
                "g_force": clean.g_force,
                "ppg_amplitude": clean.ppg_amplitude,
                "gaze_offset_deg": clean.gaze_offset_deg,
                "pitch": clean.pitch,
                "roll": clean.roll,
                "altitude": clean.altitude,
            }
        )
        ml_advisory = p_impaired >= self._advisory_threshold

        # Deterministic watchdog (hard). Runs on *raw* kinematics -- we do not
        # want the filter's lag to delay a genuine collision trigger.
        wd = self._watchdog.evaluate(packet)

        if wd.override:
            status = StatusLevel.CRITICAL_OVERRIDE
        elif ml_advisory:
            status = StatusLevel.ADVISORY
        else:
            status = StatusLevel.NOMINAL

        return Assessment(
            status=status,
            auto_gcas_engaged=wd.override,
            watchdog_override=wd.override,
            ml_advisory=ml_advisory,
            ml_impairment_probability=p_impaired,
            time_to_impact_s=wd.time_to_impact_s,
            reason=wd.reason,
            filtered=clean,
        )
