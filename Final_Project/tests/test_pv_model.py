import os
import numpy as np
import pandas as pd
import pytest
from params import load_parameters
from pv_model import (fit_single_diode, clear_sky_irradiance, plane_of_array_irradiance,
                      cell_temperature, pv_power, iv_curve)
from conftest import DATA_DIR

PV = load_parameters(os.path.join(DATA_DIR, 'pv_parameters.csv'))
SD = fit_single_diode(PV)


def _p(g, t):
    return float(np.atleast_1d(pv_power(g, t, PV, SD))[0])


def test_stc_power_matches_datasheet():
    assert _p(1000.0, 25.0) == pytest.approx(400.2, rel=0.01)


def test_stc_voc_isc():
    v, i = iv_curve(1000.0, 25.0, PV, SD)
    assert v.max() == pytest.approx(49.3, rel=0.01)
    assert i.max() == pytest.approx(10.33, rel=0.01)


def test_power_increases_with_irradiance():
    assert _p(800, 25) > _p(400, 25) > _p(200, 25) > 0


def test_power_decreases_with_temperature():
    assert _p(1000, 60) < _p(1000, 25)


def test_temperature_coefficient():
    coeff = (_p(1000, 50) / _p(1000, 25) - 1.0) / 25.0 * 100.0
    assert coeff == pytest.approx(-0.37, abs=0.08)


def test_zero_irradiance_gives_zero_power():
    assert _p(0.0, 25.0) == 0.0


def test_poa_normal_incidence_is_max():
    args = dict(dni=800.0, dhi=100.0, ghi=700.0, sun_az_deg=120.0, sun_el_deg=40.0, albedo=0.25)
    on_sun = plane_of_array_irradiance(panel_az_deg=120.0, panel_el_deg=40.0, **args)
    for daz, del_ in [(20, 0), (-20, 0), (0, 20), (0, -20)]:
        off = plane_of_array_irradiance(panel_az_deg=120.0 + daz, panel_el_deg=40.0 + del_, **args)
        assert float(on_sun) > float(off)


def test_poa_beam_zero_when_sun_behind():
    # panel normal pointing north-ish horizon, sun in the south: only diffuse + reflected remain
    poa = plane_of_array_irradiance(800.0, 100.0, 700.0, 180.0, 30.0, 0.0, 0.0, 0.25)
    beta = np.radians(90.0)
    expected = 100.0 * (1 + np.cos(beta)) / 2 + 700.0 * 0.25 * (1 - np.cos(beta)) / 2
    assert float(poa) == pytest.approx(expected, abs=1e-9)


def test_cell_temperature_noct():
    assert float(cell_temperature(800.0, 20.0, 45.0)) == pytest.approx(45.0)


def test_fit_rejects_inconsistent_panel():
    bad = dict(PV)
    bad['V_mp'] = bad['V_oc'] + 5.0
    with pytest.raises(ValueError, match='pv_parameters.csv'):
        fit_single_diode(bad)


def test_clear_sky_noon_reasonable():
    times = pd.date_range('2026-06-21 12:00', periods=1, freq='1min', tz='Asia/Riyadh')
    cs = clear_sky_irradiance(times, 24.4672, 39.6111, 620.0)
    assert 800 < cs['ghi'].iloc[0] < 1150
