import os
import pytest
import params
from params import load_parameters, load_disturbance_ranges, axis_params
from conftest import DATA_DIR

SITE_CSV = os.path.join(DATA_DIR, 'site_parameters.csv')
DRIVE_CSV = os.path.join(DATA_DIR, 'drive_parameters.csv')
RANGES_CSV = os.path.join(DATA_DIR, 'disturbance_ranges.csv')


def test_seed():
    assert params.SEED == 5162


def test_load_site():
    assert load_parameters(SITE_CSV)['lat_deg'] == pytest.approx(24.4672)


def test_axis_params():
    drive = load_parameters(DRIVE_CSV)
    ax = axis_params(drive, 'az')
    assert ax['Jp'] == 15.0 and 'Jp_el' not in ax
    assert axis_params(drive, 'el')['Jp'] == 12.0


def test_ranges_index():
    assert load_disturbance_ranges(RANGES_CSV).loc['backlash_deg', 'Max'] == 1.0


def test_project_paths_created(tmp_path):
    fake_script = os.path.join(str(tmp_path), 'code', 'x.py')
    paths = params.project_paths(fake_script)
    for key in ('data', 'raw', 'tables', 'figures'):
        assert os.path.isdir(paths[key])
