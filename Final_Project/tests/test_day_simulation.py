import os
import time
import numpy as np
import pytest
from params import load_parameters, load_disturbance_ranges, SEED
from pv_model import fit_single_diode
from day_simulation import (ideal_levels, nominal_levels, simulate_day, fixed_tilt_energy_Wh,
                            load_vibration_curve)
from conftest import DATA_DIR

SITE = load_parameters(os.path.join(DATA_DIR, 'site_parameters.csv'))
PV = load_parameters(os.path.join(DATA_DIR, 'pv_parameters.csv'))
DRIVE = load_parameters(os.path.join(DATA_DIR, 'drive_parameters.csv'))
CTRL = load_parameters(os.path.join(DATA_DIR, 'controller_parameters.csv'))
RANGES = load_disturbance_ranges(os.path.join(DATA_DIR, 'disturbance_ranges.csv'))
SD = fit_single_diode(PV)
T_AIR = 25.0


def _run(date='2026-03-20', seed=SEED, ctrl=CTRL, **changes):
    levels = ideal_levels()
    levels.update(changes)
    return simulate_day(date, SITE, PV, SD, DRIVE, ctrl, levels, T_AIR, rng=np.random.default_rng(seed))


def test_ideal_day_small_error():
    res = _run()
    assert res['rms_error_deg'] < 0.25
    assert res['energy_loss_pct'] < 0.01


def test_ideal_error_bounded_by_drift():
    res = _run()
    assert res['max_error_deg'] < CTRL['tol_deg'] + 0.25 * CTRL['update_s'] / 120 + 0.05


def test_tracking_beats_fixed_tilt():
    res = _run('2026-06-21')
    fixed = fixed_tilt_energy_Wh('2026-06-21', SITE, PV, SD, T_AIR, SITE['lat_deg'])
    assert 1.15 < res['pv_energy_Wh'] / fixed < 1.60


def test_zenith_day_does_not_blow_up():
    res = _run('2026-06-21')
    assert res['rms_error_deg'] < 0.5
    assert res['n_unsettled'] == 0


def test_backlash_increases_error():
    assert _run(backlash_deg=1.0)['rms_error_deg'] > _run()['rms_error_deg']


def test_latency_increases_error():
    assert _run(tau_d_s=120.0)['rms_error_deg'] > _run()['rms_error_deg']


def test_strong_wind_loads_the_drive():
    # User decision 2026-11-18: under wind load the PID integrator carries the torque, so moves get
    # longer and cost more motor energy; the RMS error may fall slightly (less overshoot past the
    # reference: +0.013 vs +0.024 deg), so it is not asserted to rise.
    calm, windy = _run(), _run(v_mean_mps=12.0)
    assert windy['motor_energy_Wh'] > calm['motor_energy_Wh']
    assert windy['n_unsettled'] == 0


def test_same_seed_same_result():
    levels = nominal_levels(RANGES)
    a = simulate_day('2026-03-20', SITE, PV, SD, DRIVE, CTRL, levels, T_AIR, rng=np.random.default_rng(3))
    b = simulate_day('2026-03-20', SITE, PV, SD, DRIVE, CTRL, levels, T_AIR, rng=np.random.default_rng(3))
    assert a['rms_error_deg'] == b['rms_error_deg']


def test_energy_ordering():
    res = simulate_day('2026-03-20', SITE, PV, SD, DRIVE, CTRL, nominal_levels(RANGES), T_AIR,
                       rng=np.random.default_rng(SEED))
    # 'ideal' = sun-pointing energy (spec). With the isotropic sky and albedo 0.25 the ground-reflected
    # term exceeds the diffuse term, so a slightly steeper panel can collect marginally more than
    # sun-pointing; the loss may therefore be ~ -0.001 %. Approved by the user (option 1, 2026-11-18).
    assert res['pv_energy_Wh'] >= res['net_energy_Wh'] > 0
    assert abs(res['energy_loss_pct']) < 0.01


def test_stalled_drive_completes_day():
    res = _run(Tc_Nm=5.0)
    assert isinstance(res, dict)
    assert res['n_unsettled'] > 0


def test_keep_series_columns():
    res = simulate_day('2026-03-20', SITE, PV, SD, DRIVE, CTRL, ideal_levels(), T_AIR,
                       rng=np.random.default_rng(SEED), keep_series=True)
    for col in ('time', 'sun_az', 'sun_el', 'panel_az', 'panel_el', 'error_deg', 'poa', 'power_w'):
        assert col in res['series'].columns


def test_vibration_curve_interpolates_and_clamps(tmp_path):
    path = os.path.join(str(tmp_path), 'vib.csv')
    with open(path, 'w') as fh:
        fh.write('v_mean_mps,rms_vibration_deg\n2,0.01\n4,0.03\n')
    curve = load_vibration_curve(path)
    assert curve(3.0) == pytest.approx(0.02)
    assert curve(0.0) == pytest.approx(0.01)
    assert curve(20.0) == pytest.approx(0.03)


def test_day_runtime_budget():
    levels = nominal_levels(RANGES)
    t0 = time.perf_counter()
    simulate_day('2026-03-20', SITE, PV, SD, DRIVE, CTRL, levels, T_AIR, rng=np.random.default_rng(SEED))
    elapsed = time.perf_counter() - t0
    print(f'nominal day wall time: {elapsed:.2f} s')
    assert elapsed < 30.0   # budget raised from 15 s by user decision 2026-11-18


def test_zero_wind_backlash_does_not_offset_hold():
    # User decision 2026-11-18: with zero wind torque nothing presses the panel onto a gear flank,
    # so the held panel stays where the move left it (within tol of the reference).
    ideal = _run()['rms_error_deg']
    assert _run(backlash_deg=1.0)['rms_error_deg'] < ideal + 0.02
