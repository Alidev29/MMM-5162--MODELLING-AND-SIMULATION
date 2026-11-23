# File: 10_step_response.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Closed-loop step response of the elevation axis (one 0.5 deg correction move) at the nominal
#          disturbances: panel angle, motor current and voltage against time, with rise time, overshoot
#          and settling time, for the wind pressing the panel against and along the move direction
# Inputs: data/site_parameters.csv, pv_parameters.csv, drive_parameters.csv, controller_parameters.csv,
#         disturbance_ranges.csv (nominal Tc, wind and backlash)
# Outputs: table 10_step_response.csv; figure 10_step_response.png; series/10_step_response_series.csv
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params import load_parameters, load_disturbance_ranges, project_paths, axis_params
from model_checks import check_parameters
from disturbances import wind_torque, hold_offset_rad
from drive_model import simulate_move
from analysis_tools import step_metrics

paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# ---------------------------------------------------------------- parameters
STEP_DEG = 0.5
WINDOW_S = 60.0                 # observation window of the free response (longer than a normal move)
FREE_TOL_DEG = 1e-6             # stop test disabled (the PID law itself does not use tol)
SETTLING_BAND = 0.02            # settling band +/-2 % of the 0.5 deg step, i.e. +/-0.01 deg for both cases
SIDES = {'wind against the move': -1.0, 'wind along the move': 1.0}


def main():
    site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
    pv = load_parameters(os.path.join(paths['data'], 'pv_parameters.csv'))
    drive = load_parameters(os.path.join(paths['data'], 'drive_parameters.csv'))
    ctrl = load_parameters(os.path.join(paths['data'], 'controller_parameters.csv'))
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    check_parameters(site, pv, drive, ctrl)
    ax = axis_params(drive, 'el')
    tc = float(ranges.loc['Tc_Nm', 'Nominal'])
    v = float(ranges.loc['v_mean_mps', 'Nominal'])
    backlash = np.radians(float(ranges.loc['backlash_deg', 'Nominal']))
    area = pv['length_m'] * pv['width_m']
    torque = wind_torque(v, site['air_density'], area, drive['Cm'], drive['Lc'])
    ref = np.radians(STEP_DEG)
    free_ctrl = dict(ctrl, tol_deg=FREE_TOL_DEG, timeout_s=WINDOW_S)
    print('=== Parameters ===')
    print(f"  elevation axis, step {STEP_DEG} deg, Tc {tc} N*m, wind {v} m/s (static torque {torque:.3f} N*m), "
          f"backlash {np.degrees(backlash):.2f} deg")
    print(f"  PID Kp {ctrl['Kp']:g}, Ki {ctrl['Ki']:g}, Kd {ctrl['Kd']:g}, V_max {ctrl['V_max']:g} V; window {WINDOW_S} s, "
          f"settling band +/-{100 * SETTLING_BAND:g} %")

    rows, curves = [], {}
    for label, side in SIDES.items():
        # The worm side starts at 0; the held panel rests on the flank pushed by the wind (as in the day model).
        panel0 = hold_offset_rad(v, backlash, ax['ks'], site['air_density'], area, drive['Cm'], drive['Lc'], side)
        free = simulate_move(0.0, ref, ax, free_ctrl, tc, -side * torque, backlash, theta_p0_rad=panel0)
        normal = simulate_move(0.0, ref, ax, ctrl, tc, -side * torque, backlash, theta_p0_rad=panel0)
        t, x = free['t'], free['x']
        theta_p = np.degrees(x[3])
        volts = np.clip(ctrl['Kp'] * (ref - x[3]) + ctrl['Ki'] * x[5] - ctrl['Kd'] * x[4], -ctrl['V_max'], ctrl['V_max'])
        actual_step = STEP_DEG - np.degrees(panel0)          # the panel starts on a backlash flank
        m = step_metrics(t, theta_p, np.degrees(panel0), STEP_DEG,
                         band=SETTLING_BAND * STEP_DEG / abs(actual_step))
        curves[label] = (t, theta_p, x[0], volts, normal['duration_s'])
        rows.append({'case': label, 'panel_start_deg': np.degrees(panel0), 'target_deg': STEP_DEG,
                     'actual_step_deg': actual_step, 'rise_time_s': m['rise_time_s'],
                     'overshoot_pct': m['overshoot_pct'], 'overshoot_deg': m['overshoot_pct'] / 100.0 * actual_step,
                     'settling_time_s': m['settling_time_s'], 'peak_current_A': float(np.max(np.abs(x[0]))),
                     'peak_voltage_V': float(np.max(np.abs(volts))),
                     'final_error_deg': float(STEP_DEG - theta_p[-1]),
                     'move_end_with_tol_s': normal['duration_s'], 'move_energy_J': normal['energy_J']})
    df = pd.DataFrame(rows)
    df.round(6).to_csv(os.path.join(tables_dir, '10_step_response.csv'), index=False)
    series_dir = os.path.join(tables_dir, 'series')
    os.makedirs(series_dir, exist_ok=True)
    pd.concat([pd.DataFrame({'case': label, 't_s': c[0], 'panel_angle_deg': c[1], 'current_A': c[2], 'voltage_V': c[3]})
               for label, c in curves.items()]).round(8).to_csv(
        os.path.join(series_dir, '10_step_response_series.csv'), index=False)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for label, (t, theta_p, cur, volts, t_end) in curves.items():
        line, = axs[0].plot(t, theta_p, label=label)
        axs[0].axvline(t_end, color=line.get_color(), ls=':', lw=1)
        axs[1].plot(t, cur, color=line.get_color())
        axs[2].plot(t, volts, color=line.get_color())
    axs[0].axhline(STEP_DEG, color='k', ls='--', lw=1, label=f'reference {STEP_DEG} deg')
    axs[0].axhspan(STEP_DEG * (1 - SETTLING_BAND), STEP_DEG * (1 + SETTLING_BAND), color='0.85',
                   label=f'settling band +/-{SETTLING_BAND * STEP_DEG:g} deg')
    axs[0].set_ylabel('Panel angle (deg)')
    axs[0].set_title('Elevation step response at nominal disturbances (dotted: move end with tol = 0.1 deg)')
    axs[0].legend(fontsize=8)
    axs[1].set_ylabel('Motor current (A)')
    axs[2].axhline(ctrl['V_max'], color='r', ls='--', lw=1, label='saturation +/-V_max')
    axs[2].axhline(-ctrl['V_max'], color='r', ls='--', lw=1)
    axs[2].set_ylabel('Motor voltage (V)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend(fontsize=8)
    for axp in axs:
        axp.grid(alpha=0.3)
    axs[2].set_xlim(0, WINDOW_S)
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, '10_step_response.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    reached = bool(np.all(np.abs(df['final_error_deg']) < STEP_DEG * SETTLING_BAND))
    print('\n=== Checks ===')
    print(f"  {'PASS' if reached else 'FAIL'}: panel inside the +/-{SETTLING_BAND * STEP_DEG:g} deg band at the end of the window")
    print(f"  {'PASS' if bool((df['move_end_with_tol_s'] < ctrl['timeout_s']).all()) else 'FAIL'}: "
          f"normal moves end before the {ctrl['timeout_s']:g} s timeout")
    print('\n=== Results ===')
    print(df.round(4).to_string(index=False))
    print('\n=== Engineering conclusion ===')
    for row in df.itertuples():
        print(f"  {row.case}: step {row.actual_step_deg:.3f} deg, rise {row.rise_time_s:.3f} s, overshoot "
              f"{row.overshoot_deg:.3f} deg ({row.overshoot_pct:.1f} %), settling "
              f"{row.settling_time_s:.3f} s; with tol 0.1 deg the move ends after {row.move_end_with_tol_s:.3f} s.")
    print('\n=== Files saved ===')
    print('  table : results/tables/10_step_response.csv')
    print('  figure: results/figures/10_step_response.png')
    print('  series: results/tables/series/10_step_response_series.csv')


if __name__ == '__main__':
    main()
