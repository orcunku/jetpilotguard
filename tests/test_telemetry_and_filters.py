
import pytest

from jetpilotguard.filters import KalmanFilter1D
from jetpilotguard.telemetry import TelemetryPacket


def test_valid_packet_constructs():
    p = TelemetryPacket(3000, 350, 0, 0, 1.0, 0.35, 2.0)
    assert p.altitude == 3000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"altitude": -10},
        {"airspeed": 2000},
        {"pitch": 120},
        {"g_force": 50},
        {"ppg_amplitude": 1.5},
        {"gaze_offset_deg": -3},
    ],
)
def test_out_of_range_rejected(kwargs):
    base = dict(
        altitude=3000, airspeed=350, pitch=0, roll=0,
        g_force=1.0, ppg_amplitude=0.35, gaze_offset_deg=2.0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        TelemetryPacket(**base)


def test_packet_is_immutable():
    from dataclasses import FrozenInstanceError

    p = TelemetryPacket(3000, 350, 0, 0, 1.0, 0.35, 2.0)
    with pytest.raises(FrozenInstanceError):
        p.altitude = 10  # frozen dataclass


def test_kalman_first_measurement_passes_through():
    kf = KalmanFilter1D()
    assert kf.update(42.0) == 42.0


def test_kalman_smooths_toward_signal():
    kf = KalmanFilter1D(process_variance=1e-3, measurement_variance=1e-1)
    kf.update(0.0)
    outputs = [kf.update(100.0) for _ in range(50)]
    # Should climb monotonically toward 100 and get close.
    assert outputs == sorted(outputs)
    assert 90.0 < outputs[-1] <= 100.0


def test_kalman_reset():
    kf = KalmanFilter1D()
    kf.update(5.0)
    kf.reset()
    assert kf.update(99.0) == 99.0


def test_kalman_rejects_bad_variance():
    with pytest.raises(ValueError):
        KalmanFilter1D(process_variance=0)
