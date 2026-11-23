import numpy as np
import pytest
from disturbances import friction_torque, wind_torque, dead_zone, ou_wind_series, hold_offset_rad

RHO, AREA, CM, LC = 1.10, 2.008 * 1.002, 0.10, 1.002


def test_wind_torque_zero_at_zero_speed():
    assert wind_torque(0.0, RHO, AREA, CM, LC) == 0.0


def test_wind_torque_scales_with_v_squared():
    assert wind_torque(10.0, RHO, AREA, CM, LC) / wind_torque(5.0, RHO, AREA, CM, LC) == pytest.approx(4.0)


def test_wind_torque_value():
    assert wind_torque(15.0, 1.10, 2.012, 0.10, 1.002) == pytest.approx(24.95, abs=0.05)


def test_friction_sign_and_saturation():
    tc, b, ws = 0.01, 1e-5, 0.5
    assert friction_torque(-3.0, tc, b, ws) == pytest.approx(-friction_torque(3.0, tc, b, ws))
    assert friction_torque(100.0, tc, b, ws) == pytest.approx(tc + b * 100.0, rel=1e-9)


def test_dead_zone():
    assert dead_zone(0.5, 1.0) == 0.0
    assert dead_zone(1.5, 1.0) == pytest.approx(0.5)
    assert dead_zone(-1.5, 1.0) == pytest.approx(-0.5)


def test_ou_statistics():
    t, v = ou_wind_series(8.0, 0.2, 50.0, 3600.0, 0.1, np.random.default_rng(1))
    assert len(t) == len(v)
    assert v.mean() == pytest.approx(8.0, abs=0.3)
    assert v.std() == pytest.approx(1.6, abs=0.3)


def test_ou_zero_wind_no_division_error():
    t, v = ou_wind_series(0.0, 0.2, 50.0, 60.0, 0.1, np.random.default_rng(1))
    assert np.all(v == 0.0)


def test_ou_reproducible_with_same_seed():
    _, v1 = ou_wind_series(6.0, 0.2, 50.0, 100.0, 0.1, np.random.default_rng(7))
    _, v2 = ou_wind_series(6.0, 0.2, 50.0, 100.0, 0.1, np.random.default_rng(7))
    assert np.array_equal(v1, v2)


def test_hold_offset():
    bl = np.radians(0.3)
    assert hold_offset_rad(0.0, bl, 4000.0, RHO, AREA, CM, LC, +1) == pytest.approx(bl)
    assert hold_offset_rad(0.0, bl, 4000.0, RHO, AREA, CM, LC, -1) == pytest.approx(-bl)
