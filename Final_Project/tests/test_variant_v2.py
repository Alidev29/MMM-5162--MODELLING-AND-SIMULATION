# File: test_variant_v2.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Tests of the project variant switch (PROJECT_VARIANT) and of the v2 module, the Canadian Solar
#          CS1U-400MS, against its own datasheet (STC values used by the fit; NMOT values not used by the fit).
# Author: Ali Khuzam (481014706)
# Date: 2026-11-22
import filecmp
import os
import numpy as np
import pytest
import params
from params import load_parameters
from pv_model import fit_single_diode, pv_power, iv_curve
from conftest import DATA_DIR

DATA_V2 = os.path.join(DATA_DIR, '..', 'data_v2')
PV2 = load_parameters(os.path.join(DATA_V2, 'pv_parameters.csv'))
NMOT = load_parameters(os.path.join(DATA_V2, 'pv_nmot_reference.csv'))
SD2 = fit_single_diode(PV2)


def _p(g, t):
    return float(np.atleast_1d(pv_power(g, t, PV2, SD2))[0])


def test_default_variant_is_v1(monkeypatch, tmp_path):
    monkeypatch.delenv('PROJECT_VARIANT', raising=False)
    paths = params.project_paths(os.path.join(str(tmp_path), 'code', 'x.py'))
    assert os.path.basename(paths['data']) == 'data'
    assert paths['tables'].endswith(os.path.join('results', 'tables'))


def test_v2_variant_paths(monkeypatch, tmp_path):
    monkeypatch.setenv('PROJECT_VARIANT', 'v2')
    paths = params.project_paths(os.path.join(str(tmp_path), 'code', 'x.py'))
    assert os.path.basename(paths['data']) == 'data_v2'
    assert paths['tables'].endswith(os.path.join('results_v2', 'tables'))
    assert paths['figures'].endswith(os.path.join('results_v2', 'figures'))
    assert paths['raw'].endswith(os.path.join('data', 'raw'))     # weather data shared


def test_unknown_variant_rejected(monkeypatch):
    monkeypatch.setenv('PROJECT_VARIANT', 'v3')
    with pytest.raises(ValueError, match='PROJECT_VARIANT'):
        params.project_variant()


def test_v2_changes_only_the_pv_module():
    for name in ('site_parameters.csv', 'drive_parameters.csv', 'controller_parameters.csv',
                 'disturbance_ranges.csv'):
        assert filecmp.cmp(os.path.join(DATA_DIR, name), os.path.join(DATA_V2, name), shallow=False), name
    pv1 = load_parameters(os.path.join(DATA_DIR, 'pv_parameters.csv'))
    assert (PV2['length_m'], PV2['width_m']) == (pv1['length_m'], pv1['width_m'])   # same wind load


def test_v2_stc_matches_datasheet():
    v, i = iv_curve(1000.0, 25.0, PV2, SD2)
    assert _p(1000.0, 25.0) == pytest.approx(PV2['P_max'], rel=0.01)
    assert v.max() == pytest.approx(PV2['V_oc'], rel=0.01)
    assert i.max() == pytest.approx(PV2['I_sc'], rel=0.01)


def test_v2_temperature_coefficient():
    coeff = (_p(1000, 50) / _p(1000, 25) - 1.0) / 25.0 * 100.0
    assert coeff == pytest.approx(PV2['gamma_pmp'], abs=0.08)


def test_v2_nmot_independent_validation():
    """At NMOT (800 W/m^2, cell at the NMOT temperature) the model must match the datasheet NMOT column,
    which the fit does not use: P_mp within 3 %, the other four quantities within 2 %."""
    g, t = NMOT['irradiance_Wm2'], PV2['noct_c']
    v, i = iv_curve(g, t, PV2, SD2, n_points=20001)
    k = int(np.argmax(v * i))
    assert _p(g, t) == pytest.approx(NMOT['P_max'], rel=0.03)
    for name, value in (('V_mp', v[k]), ('I_mp', i[k]), ('V_oc', v[-1]), ('I_sc', i[0])):
        assert value == pytest.approx(NMOT[name], rel=0.02), name


def test_v2_fit_insensitive_to_cell_count():
    """The cell count is not on the datasheet; 72, 78 and 84 cells give the same power."""
    ref = _p(800.0, 43.0)
    for n in (78, 84):
        pv = dict(PV2, cells_in_series=n)
        sd = fit_single_diode(pv)
        assert float(pv_power(800.0, 43.0, pv, sd)[0]) == pytest.approx(ref, rel=1e-6)
