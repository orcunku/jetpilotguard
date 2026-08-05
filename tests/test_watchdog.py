import math

from jetpilotguard.safety.watchdog import CollisionWatchdog, WatchdogConfig
from jetpilotguard.telemetry import TelemetryPacket


def make(alt, spd, pitch):
    return TelemetryPacket(alt, spd, pitch, 0.0, 1.0, 0.35, 2.0)


def test_level_flight_no_override():
    wd = CollisionWatchdog()
    res = wd.evaluate(make(5000, 350, 0))
    assert not res.override
    assert math.isinf(res.time_to_impact_s)


def test_climb_never_triggers():
    wd = CollisionWatchdog()
    assert not wd.evaluate(make(300, 400, 20)).override


def test_low_altitude_steep_dive_triggers_hard_floor():
    wd = CollisionWatchdog()
    res = wd.evaluate(make(400, 450, -40))
    assert res.override
    assert "hard floor" in res.reason or "OVERRIDE" in res.reason


def test_high_altitude_shallow_descent_safe():
    wd = CollisionWatchdog()
    # Descending, but plenty of altitude and shallow angle.
    assert not wd.evaluate(make(6000, 300, -5)).override


def test_imminent_tti_triggers_at_altitude():
    wd = CollisionWatchdog()
    # Steep, fast dive; low enough TTI to trip the TTI rule even above floor.
    res = wd.evaluate(make(600, 600, -60))
    assert res.override


def test_vertical_speed_zero_when_climbing():
    wd = CollisionWatchdog()
    assert wd.vertical_speed_fps(make(3000, 400, 10)) == 0.0


def test_config_is_respected():
    strict = CollisionWatchdog(WatchdogConfig(hard_floor_altitude_ft=2000))
    # A dive at 1500ft would be safe under defaults (500ft floor) but not here.
    assert strict.evaluate(make(1500, 400, -30)).override
