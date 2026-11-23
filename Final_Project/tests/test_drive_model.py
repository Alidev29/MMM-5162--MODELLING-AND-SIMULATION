import os
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from params import load_parameters, axis_params
from drive_model import (simulate_move, locked_panel_derivatives, natural_frequency_hz,
                         motor_time_constant_s, linear_axis_model, closed_loop_matrix)
from conftest import DATA_DIR

DRIVE = load_parameters(os.path.join(DATA_DIR, 'drive_parameters.csv'))
CTRL = load_parameters(os.path.join(DATA_DIR, 'controller_parameters.csv'))
EL = axis_params(DRIVE, 'el')
TOL = np.radians(CTRL['tol_deg'])


def _move(step_deg, Tc=0.0, **kw):
    return simulate_move(0.0, np.radians(step_deg), EL, CTRL, Tc, 0.0, 0.0, **kw)


def test_move_settles_without_disturbance():
    res = _move(0.5)
    assert res['settled']
    assert abs(res['theta_end_rad'] - np.radians(0.5)) < TOL
    assert res['duration_s'] < 10.0
    # no overshoot beyond tol
    theta_p = res['x'][3]
    assert np.max(theta_p) - np.radians(0.5) < TOL


def test_move_energy_positive_and_small():
    res = _move(0.5)
    assert 0.0 < res['energy_J'] < 500.0


def test_larger_step_takes_longer():
    assert _move(5.0)['duration_s'] > _move(0.5)['duration_s']


def test_friction_increases_energy():
    assert _move(0.5, Tc=0.02)['energy_J'] > _move(0.5, Tc=0.0)['energy_J']


def test_stalled_move_returns_unsettled():
    res = _move(0.5, Tc=5.0)
    assert res['settled'] is False
    assert res['duration_s'] == pytest.approx(CTRL['timeout_s'], rel=1e-6)


def test_natural_frequency():
    assert natural_frequency_hz(EL) == pytest.approx(np.sqrt(4000 / 12) / (2 * np.pi), rel=1e-12)
    assert natural_frequency_hz(EL) == pytest.approx(2.906, abs=0.001)


def test_motor_time_constant():
    assert motor_time_constant_s(EL) == pytest.approx(2e-5 * 2.0 / (0.05 * 0.05))


def test_locked_panel_rings_at_natural_frequency():
    dt = 0.005
    t_eval = np.arange(0.0, 40.0, dt)
    sol = solve_ivp(locked_panel_derivatives, (0.0, t_eval[-1]), [0.01, 0.0], t_eval=t_eval,
                    args=(EL, lambda t: 0.0), rtol=1e-8, atol=1e-11, max_step=0.01)
    phi = sol.y[0] - sol.y[0].mean()
    n_fft = 2 ** 18
    spec = np.abs(np.fft.rfft(phi, n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    f_peak = freqs[np.argmax(spec[1:]) + 1]
    assert f_peak == pytest.approx(natural_frequency_hz(EL), rel=0.03)


def test_locked_panel_static_deflection():
    sol = solve_ivp(locked_panel_derivatives, (0.0, 60.0), [0.0, 0.0],
                    args=(EL, lambda t: 20.0), rtol=1e-8, atol=1e-11, max_step=0.05)
    assert sol.y[0, -1] == pytest.approx(-20.0 / 4000.0, rel=1e-3)


def test_tolerance_refinement_stable():
    a = _move(0.5, rtol=1e-6, atol=1e-9)['theta_end_rad']
    b = _move(0.5, rtol=1e-9, atol=1e-12)['theta_end_rad']
    assert abs(a - b) < 1e-5


def test_move_starts_from_true_panel_angle():
    # Held panel already at the reference although theta_m/N sits a backlash away: no move needed.
    b = np.radians(0.3)
    res = simulate_move(-b, np.radians(0.0), EL, CTRL, 0.0, 0.0, b, theta_p0_rad=0.0)
    assert res['duration_s'] == 0.0
    assert res['theta_end_rad'] == pytest.approx(-b)
    assert res['x'][3, 0] == pytest.approx(0.0)


def test_move_from_flank_brings_panel_within_tol():
    # Panel resting on a flank (theta_p = theta_m/N + b + T/ks), constant wind pushing it there.
    b = np.radians(0.3)
    torque = 1.0
    theta_p0 = b + torque / EL['ks']
    ref = np.radians(0.5)
    res = simulate_move(0.0, ref, EL, CTRL, 0.01, -torque, b, theta_p0_rad=theta_p0)
    assert res['settled']
    assert abs(res['x'][3, -1] - ref) < TOL
    assert abs(res['theta_end_rad'] + theta_p0 - ref) < TOL   # hold = theta_end + offset ~ ref


def test_linear_model_matches_nonlinear_move():
    # without friction, backlash and wind and with a step small enough to stay below V_max, the linear
    # closed loop (PID on the measured angle) reproduces the nonlinear move
    from scipy.linalg import expm
    A, B, C = linear_axis_model(EL)
    acl = closed_loop_matrix(A, B, C, CTRL['Kp'], CTRL['Ki'], CTRL['Kd'], rate_row=4)
    assert np.all(np.linalg.eigvals(acl).real < 0)
    step = np.radians(0.05)
    ctrl = dict(CTRL, tol_deg=1e-6, timeout_s=3.0)
    res = simulate_move(0.0, step, EL, ctrl, 0.0, 0.0, 0.0)
    t_end = res['t'][-1]
    x0 = np.zeros(6)
    x0[1], x0[3] = -step * EL['N'], -step   # error coordinates: motor and panel start one step below the reference
    x_lin = expm(acl * t_end) @ x0
    assert x_lin[3] + step == pytest.approx(res['x'][3, -1], abs=1e-3 * step)
