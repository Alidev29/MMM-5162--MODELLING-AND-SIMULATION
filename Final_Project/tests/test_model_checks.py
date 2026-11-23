import os
import pytest
from params import load_parameters, load_disturbance_ranges
from model_checks import check_parameters, check_levels, check_pv_output
from conftest import DATA_DIR


def _load():
    site = load_parameters(os.path.join(DATA_DIR, 'site_parameters.csv'))
    pv = load_parameters(os.path.join(DATA_DIR, 'pv_parameters.csv'))
    drive = load_parameters(os.path.join(DATA_DIR, 'drive_parameters.csv'))
    ctrl = load_parameters(os.path.join(DATA_DIR, 'controller_parameters.csv'))
    return site, pv, drive, ctrl


def _ranges():
    return load_disturbance_ranges(os.path.join(DATA_DIR, 'disturbance_ranges.csv'))


def _nominal(ranges):
    return {name: float(ranges.loc[name, 'Nominal']) for name in ranges.index}


def test_rejects_negative_resistance():
    site, pv, drive, ctrl = _load()
    drive['R'] = -1
    with pytest.raises(ValueError):
        check_parameters(site, pv, drive, ctrl)


def test_rejects_zero_inertia():
    site, pv, drive, ctrl = _load()
    drive['Jp_el'] = 0
    with pytest.raises(ValueError):
        check_parameters(site, pv, drive, ctrl)


def test_rejects_latitude_out_of_range():
    site, pv, drive, ctrl = _load()
    site['lat_deg'] = 95
    with pytest.raises(ValueError):
        check_parameters(site, pv, drive, ctrl)


def test_rejects_nonpositive_update():
    site, pv, drive, ctrl = _load()
    ctrl['update_s'] = 0
    with pytest.raises(ValueError):
        check_parameters(site, pv, drive, ctrl)


def test_rejects_level_outside_range():
    ranges = _ranges()
    levels = _nominal(ranges)
    levels['backlash_deg'] = 2.0
    with pytest.raises(ValueError):
        check_levels(levels, ranges)


def test_rejects_power_above_incident():
    with pytest.raises(ValueError):
        check_pv_output(900.0, 400.0, 2.0)


def test_rejects_negative_power():
    with pytest.raises(ValueError):
        check_pv_output(-1.0, 400.0, 2.0)


def test_accepts_nominal():
    site, pv, drive, ctrl = _load()
    check_parameters(site, pv, drive, ctrl)
    ranges = _ranges()
    check_levels(_nominal(ranges), ranges)
    check_pv_output([0.0, 300.0], [0.0, 1000.0], 2.012)
