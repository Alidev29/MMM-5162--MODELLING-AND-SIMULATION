# File: 12_pid_gain_check.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Check of the representative PID gains of the elevation axis without changing them: loop margins
#          of the axis linearised at zero disturbances, the 0.5 deg step response at the nominal
#          disturbances (as study 10) and the nominal day 2026-03-20 (as study 04), for all three gains
#          scaled together by 0.5, 1 and 1.25, and the largest stable gain scale of the linearised loop
# Inputs: data/site_parameters.csv, pv_parameters.csv, drive_parameters.csv, controller_parameters.csv,
#         disturbance_ranges.csv, data/raw/nasa_power_madinah_2021_2023.csv, results/tables/vibration_curve.csv
# Outputs: tables 12_pid_gain_check.csv and 12_pid_stability_limit.csv
# Author: Ali Khuzam (481014706)
# Date: 2026-11-20

import os
import numpy as np
import pandas as pd

from params import SEED, load_parameters, load_disturbance_ranges, project_paths, axis_params
from model_checks import check_parameters
from pv_model import fit_single_diode
from weather_data import load_power_hourly, monthly_daytime_temperature
from day_simulation import simulate_day, load_vibration_curve, nominal_levels
from disturbances import wind_torque, hold_offset_rad
from drive_model import simulate_move, linear_axis_model, closed_loop_matrix
from analysis_tools import step_metrics, frequency_response, loop_margins

paths = project_paths(__file__)
tables_dir = paths['tables']

# ---------------------------------------------------------------- parameters
GAIN_SCALES = (0.5, 1.0, 1.25)       # Kp, Ki and Kd scaled together (Ti and Td, and so the anti-windup, unchanged)
DATE = '2026-03-20'               # nominal day of study 04
STEP_DEG = 0.5                    # step of study 10, wind pressing the panel against the move
WINDOW_S = 60.0                   # free-response window of study 10 (stop test disabled)
SETTLING_BAND = 0.02
W_GRID = np.logspace(-3, 4, 40001)   # rad/s, for the loop margins
LIMIT_TOL = 1e-4                     # tolerance of the bisection for the largest stable gain scale


def step_case(ax, ctrl, site, pv, drive, ranges):
    """Study 10 case 'wind against the move' with the given controller."""
    tc = float(ranges.loc['Tc_Nm', 'Nominal'])
    v = float(ranges.loc['v_mean_mps', 'Nominal'])
    backlash = np.radians(float(ranges.loc['backlash_deg', 'Nominal']))
    area = pv['length_m'] * pv['width_m']
    torque = wind_torque(v, site['air_density'], area, drive['Cm'], drive['Lc'])
    side = -1.0
    panel0 = hold_offset_rad(v, backlash, ax['ks'], site['air_density'], area, drive['Cm'], drive['Lc'], side)
    ref = np.radians(STEP_DEG)
    free = simulate_move(0.0, ref, ax, dict(ctrl, tol_deg=1e-6, timeout_s=WINDOW_S), tc, -side * torque, backlash,
                         theta_p0_rad=panel0)
    normal = simulate_move(0.0, ref, ax, ctrl, tc, -side * torque, backlash, theta_p0_rad=panel0)
    x = free['x']
    volts_unsat = ctrl['Kp'] * (ref - x[3]) + ctrl['Ki'] * x[5] - ctrl['Kd'] * x[4]
    actual = STEP_DEG - np.degrees(panel0)
    m = step_metrics(free['t'], np.degrees(x[3]), np.degrees(panel0), STEP_DEG,
                     band=SETTLING_BAND * STEP_DEG / abs(actual))
    return {'step_actual_deg': actual, 'step_rise_time_s': m['rise_time_s'], 'step_overshoot_pct': m['overshoot_pct'],
            'step_overshoot_deg': m['overshoot_pct'] / 100.0 * actual, 'step_settling_time_s': m['settling_time_s'],
            'step_peak_voltage_V': float(np.max(np.abs(volts_unsat))),
            'step_saturated': bool(np.max(np.abs(volts_unsat)) > ctrl['V_max']),
            'step_move_end_s': normal['duration_s']}


def stable(A, B, C, ctrl, g):
    poles = np.linalg.eigvals(closed_loop_matrix(A, B, C, ctrl['Kp'] * g, ctrl['Ki'] * g, ctrl['Kd'] * g, rate_row=4))
    return bool(np.all(poles.real < 0))


def stability_limit(A, B, C, ctrl, lo=1.0, hi=4.0):
    """Largest common gain scale with a stable linear closed loop, by bisection between lo (stable) and hi."""
    assert stable(A, B, C, ctrl, lo) and not stable(A, B, C, ctrl, hi)
    while hi - lo > LIMIT_TOL:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if stable(A, B, C, ctrl, mid) else (lo, mid)
    return lo


def main():
    site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
    pv = load_parameters(os.path.join(paths['data'], 'pv_parameters.csv'))
    drive = load_parameters(os.path.join(paths['data'], 'drive_parameters.csv'))
    ctrl = load_parameters(os.path.join(paths['data'], 'controller_parameters.csv'))
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    check_parameters(site, pv, drive, ctrl)
    weather = load_power_hourly(os.path.join(paths['raw'], 'nasa_power_madinah_2021_2023.csv'))
    t_air = monthly_daytime_temperature(weather)[int(DATE[5:7])]
    sd = fit_single_diode(pv)
    vib = load_vibration_curve(os.path.join(tables_dir, 'vibration_curve.csv'))
    levels = nominal_levels(ranges)
    ax = axis_params(drive, 'el')
    A, B, C = linear_axis_model(ax)
    plant = frequency_response(A, B, C, W_GRID)
    print('=== Parameters ===')
    print(f"  PID Kp {ctrl['Kp']:g}, Ki {ctrl['Ki']:g}, Kd {ctrl['Kd']:g} (elevation axis), scales {GAIN_SCALES}")
    print(f'  margins: axis linearised at zero disturbances, loop broken at the motor voltage (the same loop for '
          f'PID on error and derivative on measurement); day {DATE} at nominal disturbances, SEED {SEED}')

    rows = []
    for g in GAIN_SCALES:
        c = dict(ctrl, Kp=ctrl['Kp'] * g, Ki=ctrl['Ki'] * g, Kd=ctrl['Kd'] * g)
        s = 1j * W_GRID
        loop = (c['Kp'] + c['Ki'] / s + c['Kd'] * s) * plant
        mg = loop_margins(loop, W_GRID)
        poles = np.linalg.eigvals(closed_loop_matrix(A, B, C, c['Kp'], c['Ki'], c['Kd'], rate_row=4))
        day = simulate_day(DATE, site, pv, sd, drive, c, levels, t_air, vibration_curve=vib,
                           rng=np.random.default_rng(SEED))
        row = {'gain_scale': g, 'Kp': c['Kp'], 'Ki': c['Ki'], 'Kd': c['Kd'],
               'crossover_rad_s': mg['w_c'], 'phase_margin_deg': mg['phase_margin_deg'],
               'gain_margin_db': mg['gain_margin_db'], 'phase_crossover_rad_s': mg['w_180'],
               'closed_loop_stable': bool(np.all(poles.real < 0))}
        row.update(step_case(ax, c, site, pv, drive, ranges))
        row.update({'day_rms_error_deg': day['rms_error_deg'], 'day_max_error_deg': day['max_error_deg'],
                    'day_motor_energy_Wh': day['motor_energy_Wh'], 'day_net_energy_Wh': day['net_energy_Wh'],
                    'day_n_unsettled': day['n_unsettled']})
        rows.append(row)
        print(f'  scale {g}: done')
    df = pd.DataFrame(rows)
    ref = df.set_index('gain_scale').loc[1.0]
    df['day_rms_change_pct'] = 100.0 * (df['day_rms_error_deg'] / ref['day_rms_error_deg'] - 1.0)
    df.round(6).to_csv(os.path.join(tables_dir, '12_pid_gain_check.csv'), index=False)
    g_max = stability_limit(A, B, C, ctrl)
    w_panel = float(np.sqrt(ax['ks'] / ax['Jp']))
    lim = pd.DataFrame([{'max_stable_gain_scale': g_max, 'gm_at_x1_db': ref['gain_margin_db'],
                         'gm_limit_from_margin': 10 ** (ref['gain_margin_db'] / 20.0),
                         'phase_crossover_x1_rad_s': ref['phase_crossover_rad_s'],
                         'panel_mode_rad_s': w_panel, 'panel_mode_hz': w_panel / (2 * np.pi)}])
    lim.round(6).to_csv(os.path.join(tables_dir, '12_pid_stability_limit.csv'), index=False)

    oat = pd.read_csv(os.path.join(tables_dir, '04_oat_sweep.csv'))
    nominal_rms = oat[(oat.disturbance == 'Tc_Nm') & (oat.level == levels['Tc_Nm'])].rms_error_deg.iloc[0]
    print('\n=== Checks ===')
    same = abs(ref['day_rms_error_deg'] - nominal_rms) < 1e-6
    print(f"  {'PASS' if same else 'FAIL'}: gain x1 reproduces the nominal-day RMS of study 04 "
          f"({ref['day_rms_error_deg']:.6f} vs {nominal_rms:.6f} deg)")
    print(f"  {'PASS' if df.closed_loop_stable.all() else 'FAIL'}: linear closed loop stable at every gain scale")
    agree = abs(g_max - lim.gm_limit_from_margin.iloc[0]) < 0.01
    print(f"  {'PASS' if agree else 'FAIL'}: largest stable scale from the poles ({g_max:.3f}) matches the gain "
          f"margin at x1 ({lim.gm_limit_from_margin.iloc[0]:.3f})")
    print('\n=== Results ===')
    print(df.round(4).T.to_string())
    print(lim.round(4).T.to_string())
    print('\n=== Engineering conclusion ===')
    print(f"  At the representative gains the phase margin is {ref.phase_margin_deg:.1f} deg and the gain margin "
          f"{ref.gain_margin_db:.1f} dB; the 0.5 deg step overshoots by {ref.step_overshoot_pct:.1f} % of the move "
          f"with a peak voltage of {ref.step_peak_voltage_V:.1f} V (no saturation). Scaling all gains by "
          f"{min(GAIN_SCALES)}-{max(GAIN_SCALES)} changes the daily RMS error by {df.day_rms_change_pct.min():+.2f} "
          f"to {df.day_rms_change_pct.max():+.2f} %. The loop becomes unstable above x{g_max:.2f}: the phase "
          f"crosses -180 deg at {ref.phase_crossover_rad_s:.1f} rad/s, just above the panel torsional mode "
          f"({w_panel:.1f} rad/s, {w_panel / (2 * np.pi):.1f} Hz).")
    print('\n=== Files saved ===')
    print('  table : results/tables/12_pid_gain_check.csv')
    print('  table : results/tables/12_pid_stability_limit.csv')


if __name__ == '__main__':
    main()
