# File: 04_sensitivity_map.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Deterministic sensitivity of tracking accuracy and energy to the four disturbances:
#          one-at-a-time sweeps, normalised sensitivities at the nominal point, 7x7 map of the top two
# Inputs: data/site_parameters.csv, data/pv_parameters.csv, data/drive_parameters.csv,
#         data/controller_parameters.csv, data/disturbance_ranges.csv,
#         data/raw/nasa_power_madinah_2021_2023.csv, results/tables/vibration_curve.csv,
#         results/tables/01_baseline_days.csv
# Outputs: tables 04_oat_sweep.csv, 04_normalized_sensitivity.csv, 04_map_2d.csv, 04_limit_checks.csv;
#          figures 04_oat_rms_error.png, 04_oat_energy_loss.png, 04_sensitivity_ranking.png,
#          04_map_rms_error.png, 04_map_energy_loss.png
# Usage: python 04_sensitivity_map.py (full study) or python 04_sensitivity_map.py --plot-only (figures
#        redrawn from the saved tables, no simulation)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params import SEED, load_parameters, load_disturbance_ranges, project_paths
from model_checks import check_parameters, check_levels
from pv_model import fit_single_diode
from weather_data import load_power_hourly, monthly_daytime_temperature
from day_simulation import simulate_day, load_vibration_curve, nominal_levels, ideal_levels, LEVEL_KEYS

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# ---------------------------------------------------------------- parameters
DATE = '2026-03-20'
N_LEVELS = 7
REL_STEP = 0.10                 # central difference +/-10 % of Nominal
POINTING_LIMIT_DEG = 0.5
N_WORKERS = min(os.cpu_count(), 8)   # each worker holds ~300 MB (pandas/pvlib/scipy); 7.5 GB RAM machine
LABELS = {'Tc_Nm': 'Coulomb friction Tc (N*m)', 'v_mean_mps': 'Mean wind speed (m/s)',
          'backlash_deg': 'Backlash δ (deg)', 'tau_d_s': 'Timing delay τd (s)'}
METRICS = ('rms_error_deg', 'max_error_deg', 'energy_loss_pct', 'net_energy_Wh')


def load_context():
    """Every input the day simulation needs (loaded in each worker process)."""
    site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
    pv = load_parameters(os.path.join(paths['data'], 'pv_parameters.csv'))
    drive = load_parameters(os.path.join(paths['data'], 'drive_parameters.csv'))
    ctrl = load_parameters(os.path.join(paths['data'], 'controller_parameters.csv'))
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    check_parameters(site, pv, drive, ctrl)
    weather = load_power_hourly(os.path.join(paths['raw'], 'nasa_power_madinah_2021_2023.csv'))
    t_air = monthly_daytime_temperature(weather)[int(DATE[5:7])]
    return {'site': site, 'pv': pv, 'drive': drive, 'ctrl': ctrl, 'ranges': ranges, 'sd': fit_single_diode(pv),
            't_air': t_air, 'vib': load_vibration_curve(os.path.join(tables_dir, 'vibration_curve.csv'))}


_CONTEXT = None


def run_day(levels):
    """One simulated day at the given disturbance levels (common random numbers: fresh SEED rng)."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = load_context()
    c = _CONTEXT
    check_levels(levels, c['ranges'])
    res = simulate_day(DATE, c['site'], c['pv'], c['sd'], c['drive'], c['ctrl'], levels, c['t_air'],
                       vibration_curve=c['vib'], rng=np.random.default_rng(SEED))
    return {key: res[key] for key in METRICS + ('n_unsettled',)}


def run_many(level_list):
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        return list(pool.map(run_day, level_list))


def plot_figures(df_oat, df_sens, df_map, nominal, ranges):
    """All figures of study 04 from its result tables (called after the runs, or alone with --plot-only)."""
    plt.rcParams.update({'font.size': 13})
    top_two = list(df_sens['disturbance'].iloc[:2])
    grid_a = np.linspace(ranges.loc[top_two[0], 'Min'], ranges.loc[top_two[0], 'Max'], N_LEVELS)
    grid_b = np.linspace(ranges.loc[top_two[1], 'Min'], ranges.loc[top_two[1], 'Max'], N_LEVELS)
    names = []
    for metric, name, ylabel in (('rms_error_deg', '04_oat_rms_error.png', 'RMS pointing error (deg)'),
                                 ('energy_loss_pct', '04_oat_energy_loss.png', 'Energy loss vs sun-pointing (%)')):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        all_vals = df_oat[metric].to_numpy()
        y_lo = min(0.0, np.floor(all_vals.min() / 0.005) * 0.005)
        y_hi = np.ceil(all_vals.max() * 1.1 / 0.01) * 0.01
        for idx, key in enumerate(LEVEL_KEYS):
            axp = axs.flat[idx]
            sub = df_oat[df_oat['disturbance'] == key].sort_values('level')
            axp.plot(sub['level'], sub[metric], 'o-', label='one-at-a-time sweep')
            axp.axvline(nominal[key], color='k', ls=':', label='nominal')
            axp.set_xlim(ranges.loc[key, 'Min'], ranges.loc[key, 'Max'])
            axp.set_ylim(y_lo, y_hi)
            axp.set_xlabel(LABELS[key])
            axp.set_ylabel(ylabel)
            axp.set_title(f"{LABELS[key].split(' (')[0]} (others nominal)")
            axp.legend(fontsize=12)
            axp.grid(alpha=0.3)
        fig.suptitle(f'One-at-a-time disturbance sweeps, {DATE}')
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        names.append(name)

    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    x_pos = np.arange(len(LEVEL_KEYS))
    order = list(df_sens['disturbance'])
    ax_plot.bar(x_pos - 0.2, df_sens['S_rms_error_deg'], 0.4, label='RMS pointing error')
    ax_plot.bar(x_pos + 0.2, df_sens['S_energy_loss_pct'], 0.4, label='energy loss')
    ax_plot.set_xticks(x_pos, order)
    s_abs = np.nanmax(np.abs(df_sens[['S_rms_error_deg', 'S_energy_loss_pct']].to_numpy()))
    s_lim = np.ceil(s_abs * 1.1 * 10) / 10
    ax_plot.set_ylim(-s_lim, s_lim)
    ax_plot.axhline(0.0, color='k', lw=0.8)
    ax_plot.set_xlabel('Disturbance (ranked by |S| on RMS error)')
    ax_plot.set_ylabel('Normalised sensitivity S = (dY/Y)/(dX/X) (-)')
    ax_plot.set_title('Normalised sensitivity at the nominal point (+/-10 % central difference)')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3, axis='y')
    fig.savefig(os.path.join(figures_dir, '04_sensitivity_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    names.append('04_sensitivity_ranking.png')

    for metric, name, clabel in (('rms_error_deg', '04_map_rms_error.png', 'RMS pointing error (deg)'),
                                 ('energy_loss_pct', '04_map_energy_loss.png', 'Energy loss (%)')):
        z_grid = df_map.pivot(index=top_two[1], columns=top_two[0], values=metric).to_numpy()
        fig, ax_plot = plt.subplots(figsize=(7, 5))
        mesh = ax_plot.pcolormesh(grid_a, grid_b, z_grid, shading='nearest', cmap='viridis')
        fig.colorbar(mesh, ax=ax_plot, label=clabel)
        if metric == 'rms_error_deg' and z_grid.min() < POINTING_LIMIT_DEG < z_grid.max():
            cont = ax_plot.contour(grid_a, grid_b, z_grid, levels=[POINTING_LIMIT_DEG], colors='w', linewidths=2)
            ax_plot.clabel(cont, fmt=f'{POINTING_LIMIT_DEG:g} deg')
        ax_plot.set_xlim(grid_a[0] - 0.5 * (grid_a[1] - grid_a[0]), grid_a[-1] + 0.5 * (grid_a[1] - grid_a[0]))
        ax_plot.set_ylim(grid_b[0] - 0.5 * (grid_b[1] - grid_b[0]), grid_b[-1] + 0.5 * (grid_b[1] - grid_b[0]))
        ax_plot.set_xlabel(LABELS[top_two[0]])
        ax_plot.set_ylabel(LABELS[top_two[1]])
        ax_plot.set_title(f'{clabel} over the two most influential disturbances')
        fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        names.append(name)

    return names


def plot_only():
    """Redraw the figures from the saved tables in results/tables without re-simulating any day."""
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    df_oat = pd.read_csv(os.path.join(tables_dir, '04_oat_sweep.csv'))
    df_sens = pd.read_csv(os.path.join(tables_dir, '04_normalized_sensitivity.csv')).sort_values('rank')
    df_map = pd.read_csv(os.path.join(tables_dir, '04_map_2d.csv'))
    for name in plot_figures(df_oat, df_sens, df_map, nominal_levels(ranges), ranges):
        print(f'  figure: results/figures/{name}')


def main():
    t_start = time.perf_counter()
    context = load_context()
    ranges = context['ranges']
    nominal = nominal_levels(ranges)
    print('=== Parameters ===')
    print(f"  date {DATE}, air temperature {context['t_air']:.2f} degC (NASA POWER monthly daytime mean), SEED {SEED}")
    for key in LEVEL_KEYS:
        print(f"  {key:13s} nominal {ranges.loc[key, 'Nominal']:g}, range [{ranges.loc[key, 'Min']:g}, "
              f"{ranges.loc[key, 'Max']:g}] {ranges.loc[key, 'Unit']}")
    print(f'  {N_LEVELS} OAT levels, +/-{100 * REL_STEP:g} % central differences, {N_WORKERS} workers')
    saved_tables, saved_figures = [], []

    # ------------------------------------------------------------ batch 1: OAT, nominal, +/-10 %, zero
    jobs, tags = [], []
    for key in LEVEL_KEYS:
        for level in np.linspace(ranges.loc[key, 'Min'], ranges.loc[key, 'Max'], N_LEVELS):
            jobs.append(dict(nominal, **{key: float(level)}))
            tags.append(('oat', key, float(level)))
        for sign in (-1, 1):
            jobs.append(dict(nominal, **{key: nominal[key] * (1.0 + sign * REL_STEP)}))
            tags.append(('diff', key, sign))
    jobs.append(dict(nominal))
    tags.append(('nominal', None, None))
    jobs.append(ideal_levels())
    tags.append(('zero', None, None))
    results = run_many(jobs)
    print(f'  batch 1: {len(jobs)} days in {time.perf_counter() - t_start:.0f} s')

    oat_rows = [dict(disturbance=tag[1], level=tag[2], **res) for tag, res in zip(tags, results) if tag[0] == 'oat']
    df_oat = pd.DataFrame(oat_rows)
    df_oat[['disturbance', 'level'] + list(METRICS)].round(6).to_csv(
        os.path.join(tables_dir, '04_oat_sweep.csv'), index=False)
    saved_tables.append('04_oat_sweep.csv')
    base = next(res for tag, res in zip(tags, results) if tag[0] == 'nominal')
    zero = next(res for tag, res in zip(tags, results) if tag[0] == 'zero')

    sens_rows = []
    for key in LEVEL_KEYS:
        minus = next(res for tag, res in zip(tags, results) if tag[:3] == ('diff', key, -1))
        plus = next(res for tag, res in zip(tags, results) if tag[:3] == ('diff', key, 1))
        row = {'disturbance': key, 'nominal': nominal[key]}
        for metric in ('rms_error_deg', 'energy_loss_pct'):
            row[f'S_{metric}'] = ((plus[metric] - minus[metric]) / base[metric]) / (2.0 * REL_STEP)
        sens_rows.append(row)
    df_sens = pd.DataFrame(sens_rows)
    df_sens['rank'] = df_sens['S_rms_error_deg'].abs().rank(ascending=False, method='first').astype(int)
    df_sens = df_sens.sort_values('rank')
    df_sens.round(6).to_csv(os.path.join(tables_dir, '04_normalized_sensitivity.csv'), index=False)
    saved_tables.append('04_normalized_sensitivity.csv')
    top_two = list(df_sens['disturbance'].iloc[:2])

    # ------------------------------------------------------------ batch 2: 7x7 map of the top two
    grid_a = np.linspace(ranges.loc[top_two[0], 'Min'], ranges.loc[top_two[0], 'Max'], N_LEVELS)
    grid_b = np.linspace(ranges.loc[top_two[1], 'Min'], ranges.loc[top_two[1], 'Max'], N_LEVELS)
    map_jobs = [dict(nominal, **{top_two[0]: float(a), top_two[1]: float(b)}) for a in grid_a for b in grid_b]
    map_results = run_many(map_jobs)
    df_map = pd.DataFrame([{top_two[0]: job[top_two[0]], top_two[1]: job[top_two[1]], **res}
                           for job, res in zip(map_jobs, map_results)])
    df_map[top_two + list(METRICS)].round(6).to_csv(os.path.join(tables_dir, '04_map_2d.csv'), index=False)
    saved_tables.append('04_map_2d.csv')
    print(f'  batch 2: {len(map_jobs)} days, total {time.perf_counter() - t_start:.0f} s')

    # ------------------------------------------------------------ limit checks
    # Causes of non-monotonic trends (diagnosed 2026-11-18, kept as findings by user decision).
    notes = {
        'Tc_Nm': 'friction slows the final approach and reduces the mean +0.024 deg overshoot past the '
                 'mid-period reference',
        'v_mean_mps': 'wind load makes the integrator carry the torque, reducing the overshoot (dip near 5 m/s); '
                      'above that vibration and load raise the error',
        'backlash_deg': 'panel-angle feedback compensates backlash while the wind presses the panel on one flank '
                        '(flat curve; b = 0 slightly higher)',
        'tau_d_s': 'a short delay (about 20 s) cancels the overshoot lead; larger delays dominate',
    }
    checks = []
    for key in LEVEL_KEYS:
        rms = df_oat[df_oat['disturbance'] == key].sort_values('level')['rms_error_deg'].to_numpy()
        steps = np.diff(rms)
        checks.append({'check': f'rms error non-decreasing over the {key} range',
                       'value': float(steps.min()), 'criterion': 'min step >= 0',
                       'result': 'PASS' if np.all(steps >= 0.0) else 'FAIL',
                       'note': '' if np.all(steps >= 0.0) else notes[key]})
    df_01 = pd.read_csv(os.path.join(tables_dir, '01_baseline_days.csv')).set_index('date')
    for metric in ('rms_error_deg', 'max_error_deg'):
        ref = float(df_01.loc[DATE, metric])
        diff = zero[metric] - ref
        checks.append({'check': f'all levels zero: {metric} vs script 01 ({ref:.4f})', 'value': diff,
                       'criterion': '|diff| < 1e-4 (01 table rounding)',
                       'result': 'PASS' if abs(diff) < 1e-4 else 'FAIL', 'note': ''})
    df_checks = pd.DataFrame(checks)
    df_checks['value'] = df_checks['value'].round(6)
    df_checks.to_csv(os.path.join(tables_dir, '04_limit_checks.csv'), index=False)
    saved_tables.append('04_limit_checks.csv')

    # ------------------------------------------------------------ figures
    saved_figures += plot_figures(df_oat, df_sens, df_map, nominal, ranges)

    # ------------------------------------------------------------ summary
    print('\n=== Normalised sensitivity (nominal point) ===')
    for _, row in df_sens.iterrows():
        print(f"  #{row['rank']} {row['disturbance']:13s} S_rms = {row['S_rms_error_deg']:+.4f}, "
              f"S_loss = {row['S_energy_loss_pct']:+.4f}")
    print(f"  nominal day: rms {base['rms_error_deg']:.4f} deg, loss {base['energy_loss_pct']:.5f} %, "
          f"unsettled moves {base['n_unsettled']}")
    print(f"  OAT extremes: rms {df_oat['rms_error_deg'].min():.4f}-{df_oat['rms_error_deg'].max():.4f} deg, "
          f"loss {df_oat['energy_loss_pct'].min():.5f}-{df_oat['energy_loss_pct'].max():.5f} %")
    print(f"  map: rms {df_map['rms_error_deg'].min():.4f}-{df_map['rms_error_deg'].max():.4f} deg, "
          f"loss {df_map['energy_loss_pct'].min():.5f}-{df_map['energy_loss_pct'].max():.5f} %")
    print('\n=== Limit checks ===')
    for _, row in df_checks.iterrows():
        print(f"  {row['result']}: {row['check']} (value {row['value']:+.6f}, {row['criterion']})")
    print(f'  total run time {time.perf_counter() - t_start:.0f} s')
    print('\n=== Files saved ===')
    for name in saved_tables:
        print(f'  table : results/tables/{name}')
    for name in saved_figures:
        print(f'  figure: results/figures/{name}')


if __name__ == '__main__':
    plot_only() if '--plot-only' in sys.argv else main()
