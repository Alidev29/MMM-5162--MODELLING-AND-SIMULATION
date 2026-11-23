import numpy as np
import pandas as pd
import pytest
from solar_position import sun_position_spa, sun_position_simple, pointing_error_deg

LAT, LON, ALT = 24.4672, 39.6111, 620.0
TZ = 'Asia/Riyadh'


def test_spa_matches_nrel_reference():
    # NREL SPA report example: 2003-10-17 12:30:30 UTC-7, Golden CO
    t = pd.DatetimeIndex(['2003-10-17 12:30:30'], tz='Etc/GMT+7')
    sp = sun_position_spa(t, 39.742476, -105.1786, 1830.14,
                          pressure_pa=82000, temperature_c=11.0, delta_t=67.0)
    assert sp['azimuth_deg'].iloc[0] == pytest.approx(194.34024, abs=0.001)
    assert 90 - sp['elevation_deg'].iloc[0] == pytest.approx(50.11162, abs=0.01)


def _equinox_day():
    times = pd.date_range('2026-03-20 10:00', '2026-03-20 14:00', freq='10s', tz=TZ)
    return sun_position_spa(times, LAT, LON, ALT)


def test_sun_due_south_at_solar_noon_madinah():
    sp = _equinox_day()
    noon = sp['elevation_deg'].idxmax()
    assert sp.loc[noon, 'azimuth_deg'] == pytest.approx(180.0, abs=0.5)


def test_equinox_noon_elevation():
    sp = _equinox_day()
    assert sp['elevation_deg'].max() == pytest.approx(90 - LAT, abs=0.3)


def test_simple_close_to_spa():
    times = pd.date_range('2026-06-21 00:00', '2026-06-21 23:00', freq='1h', tz=TZ)
    spa = sun_position_spa(times, LAT, LON, ALT)
    simple = sun_position_simple(times, LAT, LON)
    up = spa['elevation_deg'].to_numpy() > 0
    err = pointing_error_deg(spa['azimuth_deg'].to_numpy()[up], spa['elevation_deg'].to_numpy()[up],
                             simple['azimuth_deg'].to_numpy()[up], simple['elevation_deg'].to_numpy()[up])
    assert np.all(err < 1.0)


def test_pointing_error_zero_for_identical():
    assert pointing_error_deg(123.0, 45.0, 123.0, 45.0) == pytest.approx(0.0, abs=1e-9)


def test_pointing_error_pure_elevation():
    assert pointing_error_deg(180.0, 30.0, 180.0, 31.0) == pytest.approx(1.0, abs=1e-9)


def test_pointing_error_small_near_zenith():
    assert pointing_error_deg(0.0, 89.9, 180.0, 89.9) == pytest.approx(0.2, abs=1e-9)
