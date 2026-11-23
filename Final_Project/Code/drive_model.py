# File: drive_model.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Two-mass axis model (DC motor + worm gearbox with backlash + torsionally flexible panel)
#          under PID position control; one move simulation; locked-panel vibration model.
# Inputs: axis parameter dict (params.axis_params), controller dict, disturbance values
# Outputs: state trajectories, end angle, move duration and motor energy
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17
#
# State x = [i, theta_m, omega_m, theta_p, omega_p, e_int] (A, rad, rad/s, rad, rad/s, rad*s);
# theta_m is the motor-side angle (before the gearbox), theta_p the panel angle.

import math
import numpy as np
from scipy.integrate import LSODA
from scipy.optimize import brentq

SETTLE_OMEGA_RAD_S = 1e-4
EPS = np.finfo(float).eps


def control_voltage(x, theta_ref, ctrl):
    """PID with derivative on measurement, saturated at +/-V_max. Returns (V, V_unsat, e)."""
    e = theta_ref - x[3]
    v_unsat = ctrl['Kp'] * e + ctrl['Ki'] * x[5] - ctrl['Kd'] * x[4]
    v_max = ctrl['V_max']
    volt = v_max if v_unsat > v_max else (-v_max if v_unsat < -v_max else v_unsat)
    return volt, v_unsat, e


def axis_derivatives(t, x, theta_ref, ax, ctrl, Tc, T_wind, backlash_rad):
    """Right-hand side of the axis equations (spec section 5.3), scalar math for speed."""
    i, theta_m, omega_m, theta_p, omega_p, e_int = x
    e = theta_ref - theta_p
    v_unsat = ctrl['Kp'] * e + ctrl['Ki'] * e_int - ctrl['Kd'] * omega_p
    v_max = ctrl['V_max']
    volt = v_max if v_unsat > v_max else (-v_max if v_unsat < -v_max else v_unsat)
    n_gear = ax['N']
    twist = theta_m / n_gear - theta_p
    if twist > backlash_rad:
        twist_dz = twist - backlash_rad
    elif twist < -backlash_rad:
        twist_dz = twist + backlash_rad
    else:
        twist_dz = 0.0
    shaft = ax['ks'] * twist_dz + ax['cs'] * (omega_m / n_gear - omega_p)
    friction = Tc * math.tanh(omega_m / ax['omega_s']) + ax['b'] * omega_m
    di = (volt - ax['R'] * i - ax['Ke'] * omega_m) / ax['L']
    domega_m = (ax['Kt'] * i - friction - shaft / (n_gear * ax['eta'])) / ax['Jm']
    domega_p = (shaft - T_wind) / ax['Jp']
    # Back-calculation anti-windup (Astrom & Hagglund): de_int/dt = e + (V - V_unsat)/(Ki*Tt).
    # It equals the spec's de_int/dt = e whenever V is not saturated, and unlike conditional
    # integration it is continuous, so a stalled move does not chatter on the saturation boundary.
    de_int = e
    if volt != v_unsat and ctrl['Ki'] > 0.0:
        de_int += (volt - v_unsat) / (ctrl['Ki'] * tracking_time_s(ctrl))
    return np.array([di, omega_m, domega_m, omega_p, domega_p, de_int])


def tracking_time_s(ctrl):
    """Anti-windup tracking time Tt = sqrt(Ti*Td) (Ti = Kp/Ki, Td = Kd/Kp); Ti when Kd = 0."""
    t_i = ctrl['Kp'] / ctrl['Ki']
    t_d = ctrl['Kd'] / ctrl['Kp']
    return math.sqrt(t_i * t_d) if t_d > 0.0 else t_i


def simulate_move(theta0_rad, theta_ref_rad, ax, ctrl, Tc, T_wind, backlash_rad,
                  rtol=1e-6, atol=1e-9, theta_p0_rad=None):
    """Integrate one move until |e| < tol and |omega_m/N| < 1e-4 rad/s, or until timeout_s.
    theta0_rad is the worm-side output angle theta_m/N; theta_p0_rad the true held panel angle
    (resting on a backlash flank, default theta0_rad). The PID measures the panel angle, so the
    move decision and the controller start from theta_p0_rad."""
    tol = np.radians(ctrl['tol_deg'])
    timeout = ctrl['timeout_s']
    if theta_p0_rad is None:
        theta_p0_rad = theta0_rad
    x0 = np.array([0.0, theta0_rad * ax['N'], 0.0, theta_p0_rad, 0.0, 0.0])
    if abs(theta_ref_rad - theta_p0_rad) < tol:
        return {'theta_end_rad': float(theta0_rad), 'duration_s': 0.0, 'energy_J': 0.0,
                'settled': True, 't': np.array([0.0]), 'x': x0.reshape(-1, 1)}

    # Stop when the panel error is inside tol and the worm (motor) side output speed is ~0.
    # The motor side is what the self-locking worm holds; the lightly damped panel rings about it
    # at its torsional frequency, so omega_p only crosses zero momentarily (ill-posed stop test).
    def settled_event(x):
        return max(abs(theta_ref_rad - x[3]) / tol,
                   abs(x[2] / ax['N']) / SETTLE_OMEGA_RAD_S) - 1.0

    # LSODA stepped directly (same solver as solve_ivp(method='LSODA')); the terminal event is
    # detected as a +/- sign change between accepted steps and refined with brentq on the dense
    # output, exactly as solve_ivp does, without its per-step event overhead.
    solver = LSODA(lambda t, x: axis_derivatives(t, x, theta_ref_rad, ax, ctrl, Tc, T_wind, backlash_rad),
                   0.0, x0, timeout, rtol=rtol, atol=atol)
    t_points = [0.0]
    x_points = [x0]
    g_prev = settled_event(x0)
    settled = False
    while solver.status == 'running':
        solver.step()
        if solver.status == 'failed':
            break
        g_now = settled_event(solver.y)
        if g_prev > 0.0 and g_now <= 0.0:
            dense = solver.dense_output()
            t_hit = brentq(lambda t: settled_event(dense(t)), solver.t_old, solver.t, xtol=4 * EPS)
            t_points.append(t_hit)
            x_points.append(dense(t_hit))
            settled = True
            break
        t_points.append(solver.t)
        x_points.append(solver.y.copy())
        g_prev = g_now
    t_arr = np.array(t_points)
    x_arr = np.array(x_points).T
    volts = np.clip(ctrl['Kp'] * (theta_ref_rad - x_arr[3]) + ctrl['Ki'] * x_arr[5] - ctrl['Kd'] * x_arr[4],
                    -ctrl['V_max'], ctrl['V_max'])
    energy = float(np.trapezoid(np.abs(volts * x_arr[0]), t_arr))
    # After shut-off the worm locks the motor side; the panel rests at theta_m/N (the static
    # backlash and wind offsets are added by the caller through hold_offset_rad).
    return {'theta_end_rad': float(x_arr[1, -1] / ax['N']), 'duration_s': float(t_arr[-1]), 'energy_J': energy,
            'settled': settled, 't': t_arr, 'x': x_arr}


def locked_panel_derivatives(t, y, ax, T_wind_func):
    """Locked drive, flexible panel: Jp*phi'' = -ks*phi - cs*phi' - T_wind(t); y = [phi, omega]."""
    phi, omega = y
    return np.array([omega, (-ax['ks'] * phi - ax['cs'] * omega - T_wind_func(t)) / ax['Jp']])


def natural_frequency_hz(ax):
    """Torsional natural frequency of the locked panel: sqrt(ks/Jp)/(2*pi)."""
    return float(np.sqrt(ax['ks'] / ax['Jp']) / (2.0 * np.pi))


def motor_time_constant_s(ax):
    """No-load mechanical time constant of the motor: Jm*R/(Kt*Ke)."""
    return float(ax['Jm'] * ax['R'] / (ax['Kt'] * ax['Ke']))


def linear_axis_model(ax):
    """Axis equations linearised at zero disturbances (no Coulomb friction, no backlash, no wind, no
    saturation): state [i, theta_m, omega_m, theta_p, omega_p], input motor voltage V, output panel angle
    theta_p. Returns (A, B, C) of dx/dt = A x + B V, theta_p = C x."""
    n_gear, eta = ax['N'], ax['eta']
    ks, cs = ax['ks'], ax['cs']
    A = np.zeros((5, 5))
    A[0, 0], A[0, 2] = -ax['R'] / ax['L'], -ax['Ke'] / ax['L']
    A[1, 2] = 1.0
    # shaft torque = ks (theta_m/N - theta_p) + cs (omega_m/N - omega_p), reflected to the motor as shaft/(N eta)
    k_ref = 1.0 / (n_gear * eta * ax['Jm'])
    A[2, 0] = ax['Kt'] / ax['Jm']
    A[2, 1] = -k_ref * ks / n_gear
    A[2, 2] = -ax['b'] / ax['Jm'] - k_ref * cs / n_gear
    A[2, 3] = k_ref * ks
    A[2, 4] = k_ref * cs
    A[3, 4] = 1.0
    A[4, 1], A[4, 2] = ks / (n_gear * ax['Jp']), cs / (n_gear * ax['Jp'])
    A[4, 3], A[4, 4] = -ks / ax['Jp'], -cs / ax['Jp']
    B = np.zeros(5)
    B[0] = 1.0 / ax['L']
    C = np.zeros(5)
    C[3] = 1.0
    return A, B, C


def closed_loop_matrix(A, B, C, kp, ki, kd, rate_row):
    """Closed loop of the linear axis with PID on the measured angle (reference 0): V = -kp y + ki z - kd
    dy/dt, dz/dt = -y, where dy/dt is state row `rate_row` (omega_p). Returns the 6x6 system matrix."""
    n = A.shape[0]
    rate = np.zeros(n)
    rate[rate_row] = 1.0
    gain = -kp * C - kd * rate
    acl = np.zeros((n + 1, n + 1))
    acl[:n, :n] = A + np.outer(B, gain)
    acl[:n, n] = B * ki
    acl[n, :n] = -C
    return acl
