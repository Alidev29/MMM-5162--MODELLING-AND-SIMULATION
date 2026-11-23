# File: 05_monte_carlo_risk.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Monte Carlo risk of exceeding the 0.5 deg RMS pointing limit under uncertain disturbances,
#          convergence with N, and ranking of which input uncertainty matters most
# Inputs: data/*.csv parameter files, data/disturbance_ranges.csv (Nominal, Min, Max, SD),
#         results/tables/02_selected_wind_distribution.csv, results/tables/vibration_curve.csv,
#         data/raw/nasa_power_madinah_2021_2023.csv; command-line flag --recompute
# Outputs: tables 05_mc_trials.csv, 05_risk_convergence.csv, 05_uncertainty_ranking.csv;
#          figures 05_rms_error_histogram.png, 05_energy_loss_histogram.png, 05_convergence.png,
#          05_uncertainty_ranking.png, 05_input_samples.png
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
from scipy import stats

from params import SEED, load_parameters, load_disturbance_ranges, project_paths
from model_checks import check_parameters, check_levels
from pv_model import fit_single_diode
from weather_data import load_power_hourly, monthly_daytime_temperature, load_selected_distribution
from day_simulation import simulate_day, load_vibration_curve, LEVEL_KEYS

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']
TRIALS_CSV = os.path.join(tables_dir, '05_mc_trials.csv')

# ---------------------------------------------------------------- parameters
DATE = '2026-03-20'
N_BASE = 1000
N_REDUCED = 500
N_CHECKPOINTS = (250, 500, 1000)
FAIL_LIMIT_DEG = 0.5
Z_95 = 1.96
SD_FACTOR = 0.5                 # each reduced run halves one input's spread
N_WORKERS = min(os.cpu_count(), 8)   # each worker holds ~300 MB (pandas/pvlib/scipy); 7.5 GB RAM machine
TRUNCNORM_KEYS = ('Tc_Nm', 'backlash_deg', 'tau_d_s')
SETS = ('baseline',) + tuple(f'reduced_{key}' for key in LEVEL_KEYS)
METRICS = ('rms_error_deg', 'max_error_deg', 'energy_loss_pct', 'net_energy_Wh', 'motor_energy_Wh')
UNITS = {'Tc_Nm': 'N*m', 'v_mean_mps': 'm/s', 'backlash_deg': 'deg', 'tau_d_s': 's'}


def load_context():
    site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
    pv = load_parameters(os.path.join(paths['data'], 'pv_parameters.csv'))
    drive = load_parameters(os.path.join(paths['data'], 'drive_parameters.csv'))
    ctrl = load_parameters(os.path.join(paths['data'], 'controller_parameters.csv'))
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    check_parameters(site, pv, drive, ctrl)
    weather = load_power_hourly(os.path.join(paths['raw'], 'nasa_power_madinah_2021_2023.csv'))
    return {'site': site, 'pv': pv, 'drive': drive, 'ctrl': ctrl, 'ranges': ranges, 'sd': fit_single_diode(pv),
            't_air': monthly_daytime_temperature(weather)[int(DATE[5:7])],
            'vib': load_vibration_curve(os.path.join(tables_dir, 'vibration_curve.csv')),
            'wind': load_selected_distribution(os.path.join(tables_dir, '02_selected_wind_distribution.csv'))}


def input_distributions(ranges, wind, reduced=None):
    """Frozen input distributions; `reduced` names the input whose spread is halved (means kept)."""
    dists = {}
    for key in TRUNCNORM_KEYS:
        nominal, sd = float(ranges.loc[key, 'Nominal']), float(ranges.loc[key, 'SD'])
        sd = sd * SD_FACTOR if key == reduced else sd
        lower, upper = float(ranges.loc[key, 'Min']), float(ranges.loc[key, 'Max'])
        dists[key] = stats.truncnorm((lower - nominal) / sd, (upper - nominal) / sd, loc=nominal, scale=sd)
    dists['v_mean_mps'] = wind
    return dists


def sample_levels(k, dists, ranges, wind_factor):
    """Trial k: rng_k = default_rng([SEED, k]); one uniform per input mapped through the inverse CDF
    (common random numbers across the baseline and the reduced runs); rng_k then drives the day."""
    rng = np.random.default_rng([SEED, k])
    u = rng.random(len(LEVEL_KEYS))
    levels = {key: float(dists[key].ppf(u_i)) for key, u_i in zip(LEVEL_KEYS, u)}
    wind_mean = float(dists['v_mean_mps'].mean())
    v = wind_mean + wind_factor * (levels['v_mean_mps'] - wind_mean)   # compression toward the mean
    levels['v_mean_mps'] = float(np.clip(v, ranges.loc['v_mean_mps', 'Min'], ranges.loc['v_mean_mps', 'Max']))
    return levels, rng


_CONTEXT = None


def run_trial(job):
    """job = (set_name, k): sample the inputs, simulate the day, return one table row."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = load_context()
    c = _CONTEXT
    set_name, k = job
    reduced = set_name.replace('reduced_', '') if set_name != 'baseline' else None
    dists = input_distributions(c['ranges'], c['wind'], reduced if reduced in TRUNCNORM_KEYS else None)
    levels, rng = sample_levels(k, dists, c['ranges'], SD_FACTOR if reduced == 'v_mean_mps' else 1.0)
    check_levels(levels, c['ranges'])
    res = simulate_day(DATE, c['site'], c['pv'], c['sd'], c['drive'], c['ctrl'], levels, c['t_air'],
                       vibration_curve=c['vib'], rng=rng)
    return {'set': set_name, 'trial': k, 'seed': SEED, **levels, **{m: res[m] for m in METRICS},
            'n_unsettled': res['n_unsettled']}


def failure_ci(n_fail, n):
    """95 % CI: normal approximation, Wilson score when p_hat < 0.05 or > 0.95."""
    p_hat = n_fail / n
    if 0.05 <= p_hat <= 0.95:
        half = Z_95 * np.sqrt(p_hat * (1.0 - p_hat) / n)
        return p_hat, max(0.0, p_hat - half), min(1.0, p_hat + half), 'normal'
    denom = 1.0 + Z_95 ** 2 / n
    centre = (p_hat + Z_95 ** 2 / (2 * n)) / denom
    half = Z_95 * np.sqrt(p_hat * (1.0 - p_hat) / n + Z_95 ** 2 / (4 * n ** 2)) / denom
    return p_hat, max(0.0, centre - half), min(1.0, centre + half), 'Wilson'


def set_size(set_name):
    return N_BASE if set_name == 'baseline' else N_REDUCED


def main():
    recompute = '--recompute' in sys.argv
    t_start = time.perf_counter()
    context = load_context()
    ranges = context['ranges']
    print('=== Parameters ===')
    print(f"  date {DATE}, air temperature {context['t_air']:.2f} degC, SEED {SEED}, failure: rms > {FAIL_LIMIT_DEG} deg")
    for key in TRUNCNORM_KEYS:
        print(f"  {key:13s} truncnorm(nominal {ranges.loc[key, 'Nominal']:g}, SD {ranges.loc[key, 'SD']:g}) on "
              f"[{ranges.loc[key, 'Min']:g}, {ranges.loc[key, 'Max']:g}] {UNITS[key]}")
    wind = context['wind']
    print(f"  v_mean_mps    {wind.dist.name}{tuple(round(a, 4) for a in wind.args)} scale {wind.kwds['scale']:.4f}, "
          f"clipped to [{ranges.loc['v_mean_mps', 'Min']:g}, {ranges.loc['v_mean_mps', 'Max']:g}] m/s")
    print(f'  N baseline {N_BASE}, N per reduced run {N_REDUCED}, {N_WORKERS} workers')

    # ------------------------------------------------------------ trials (reused set by set when complete)
    cached = pd.read_csv(TRIALS_CSV) if os.path.exists(TRIALS_CSV) and not recompute else pd.DataFrame()
    frames = []
    for set_name in SETS:
        n_set = set_size(set_name)
        if len(cached):
            old = cached[(cached['set'] == set_name) & (cached['seed'] == SEED)]
            if len(old) == n_set and set(old['trial']) == set(range(n_set)):
                frames.append(old)
                print(f'  {set_name}: reused {n_set} trials from 05_mc_trials.csv')
                continue
        t_set = time.perf_counter()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            rows = list(pool.map(run_trial, [(set_name, k) for k in range(n_set)], chunksize=2))
        frames.append(pd.DataFrame(rows))
        pd.concat(frames + ([cached[~cached['set'].isin([f['set'].iloc[0] for f in frames])]] if len(cached) else []),
                  ignore_index=True).round(6).to_csv(TRIALS_CSV, index=False)     # checkpoint after each set
        print(f'  {set_name}: {n_set} trials in {time.perf_counter() - t_set:.0f} s', flush=True)
    # Statistics are always computed from the saved (rounded) trial table, so a rerun that reuses the
    # cache reproduces every number exactly.
    df_trials = pd.concat(frames, ignore_index=True).round(6)
    df_trials.to_csv(TRIALS_CSV, index=False)
    base = df_trials[df_trials['set'] == 'baseline'].sort_values('trial')
    fails = (base['rms_error_deg'] > FAIL_LIMIT_DEG).to_numpy()

    # ------------------------------------------------------------ convergence
    conv_rows = []
    for n in N_CHECKPOINTS:
        sub = base.iloc[:n]
        p_hat, lo, hi, method = failure_ci(int(fails[:n].sum()), n)
        conv_rows.append({'N': n, 'p_fail': p_hat, 'ci_low': lo, 'ci_high': hi, 'ci_method': method,
                          'mean_loss_pct': sub['energy_loss_pct'].mean(),
                          'p95_loss_pct': sub['energy_loss_pct'].quantile(0.95),
                          'mean_rms_deg': sub['rms_error_deg'].mean()})
    df_conv = pd.DataFrame(conv_rows)
    df_conv.round(6).to_csv(os.path.join(tables_dir, '05_risk_convergence.csv'), index=False)

    # ------------------------------------------------------------ uncertainty ranking (paired, first N_REDUCED)
    base_red = base.iloc[:N_REDUCED]
    p_base = float((base_red['rms_error_deg'] > FAIL_LIMIT_DEG).mean())
    rank_rows = []
    for key in LEVEL_KEYS:
        red = df_trials[df_trials['set'] == f'reduced_{key}'].sort_values('trial')
        p_red = float((red['rms_error_deg'] > FAIL_LIMIT_DEG).mean())
        rank_rows.append({'input': key, 'p_fail_baseline': p_base, 'p_fail_reduced': p_red,
                          'reduction': p_base - p_red,
                          'mean_rms_baseline_deg': base_red['rms_error_deg'].mean(),
                          'mean_rms_reduced_deg': red['rms_error_deg'].mean(),
                          'rms_reduction_deg': base_red['rms_error_deg'].mean() - red['rms_error_deg'].mean()})
    df_rank = pd.DataFrame(rank_rows)
    # Rank by the failure-probability reduction; ties (e.g. no failures) broken by the mean-RMS reduction.
    df_rank = df_rank.sort_values(['reduction', 'rms_reduction_deg'], ascending=False).reset_index(drop=True)
    df_rank['rank'] = np.arange(1, len(df_rank) + 1)
    df_rank.round(6).to_csv(os.path.join(tables_dir, '05_uncertainty_ranking.csv'), index=False)

    # ------------------------------------------------------------ credibility checks
    widths = (df_conv['ci_high'] - df_conv['ci_low']).to_numpy()
    width_ok = bool(np.all(np.diff(widths) < 0.0))
    mean_rows = []
    for key in LEVEL_KEYS:
        nominal = float(ranges.loc[key, 'Nominal'])
        sample_mean = float(base[key].mean())
        mean_rows.append((key, sample_mean, nominal, 100.0 * (sample_mean - nominal) / nominal))

    # ------------------------------------------------------------ figures
    saved_figures = []

    def save(fig, name):
        fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_figures.append(name)

    rms_hi = max(np.ceil(base['rms_error_deg'].max() * 10) / 10, FAIL_LIMIT_DEG + 0.1)
    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    ax_plot.hist(base['rms_error_deg'], bins=np.linspace(0, rms_hi, 41), color='C0', edgecolor='w',
                 label=f'N = {N_BASE} simulated days')
    ax_plot.axvline(FAIL_LIMIT_DEG, color='r', ls='--', label=f'failure limit {FAIL_LIMIT_DEG} deg')
    ax_plot.set_xlim(0, rms_hi)
    counts = np.histogram(base['rms_error_deg'], bins=np.linspace(0, rms_hi, 41))[0]
    ax_plot.set_ylim(0, np.ceil(counts.max() * 1.1 / 10) * 10)
    ax_plot.set_xlabel('Daily RMS pointing error (deg)')
    ax_plot.set_ylabel('Number of trials (-)')
    ax_plot.set_title(f'Monte Carlo distribution of the daily RMS pointing error, {DATE}')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3)
    save(fig, '05_rms_error_histogram.png')

    loss = base['energy_loss_pct']
    loss_lo, loss_hi = np.floor(loss.min() * 1e4) / 1e4, np.ceil(loss.max() * 1e4) / 1e4
    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    ax_plot.hist(loss, bins=np.linspace(loss_lo, loss_hi, 41), color='C1', edgecolor='w',
                 label=f'N = {N_BASE} simulated days')
    ax_plot.axvline(float(loss.quantile(0.95)), color='k', ls=':', label='95th percentile')
    ax_plot.set_xlim(loss_lo, loss_hi)
    counts = np.histogram(loss, bins=np.linspace(loss_lo, loss_hi, 41))[0]
    ax_plot.set_ylim(0, np.ceil(counts.max() * 1.1 / 10) * 10)
    ax_plot.set_xlabel('Daily energy loss vs sun-pointing (%)')
    ax_plot.set_ylabel('Number of trials (-)')
    ax_plot.set_title(f'Monte Carlo distribution of the daily energy loss, {DATE}')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3)
    save(fig, '05_energy_loss_histogram.png')

    n_axis = np.arange(1, N_BASE + 1)
    cum = np.array([failure_ci(int(fails[:n].sum()), n) for n in n_axis], dtype=object)
    p_cum = cum[:, 0].astype(float)
    lo_cum = cum[:, 1].astype(float)
    hi_cum = cum[:, 2].astype(float)
    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    ax_plot.fill_between(n_axis, lo_cum, hi_cum, color='C0', alpha=0.25, label='95 % confidence interval')
    ax_plot.plot(n_axis, p_cum, color='C0', label='cumulative failure probability')
    ax_plot.set_xlim(1, N_BASE)
    ax_plot.set_ylim(0, np.ceil(max(hi_cum[9:].max(), 0.01) * 1.1 * 100) / 100)
    ax_plot.set_xlabel('Number of trials N (-)')
    ax_plot.set_ylabel(f'P(RMS error > {FAIL_LIMIT_DEG} deg) (-)')
    ax_plot.set_title('Convergence of the failure probability (first 10 trials outside the y-range)')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3)
    save(fig, '05_convergence.png')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    order = list(df_rank['input'])
    axs[0].bar(order, df_rank['reduction'], color='C2')
    axs[0].set_ylabel('Reduction of P(fail) (-)')
    axs[0].set_title('Failure-probability reduction when one input SD is halved')
    red_abs = max(abs(df_rank['reduction']).max(), 0.001)
    axs[0].set_ylim(-red_abs * 1.2, red_abs * 1.2)
    axs[1].bar(order, df_rank['rms_reduction_deg'], color='C4')
    axs[1].set_ylabel('Reduction of mean daily RMS error (deg)')
    axs[1].set_title('Mean-RMS reduction when one input SD is halved')
    rms_abs = max(abs(df_rank['rms_reduction_deg']).max(), 0.001)
    axs[1].set_ylim(-rms_abs * 1.2, rms_abs * 1.2)
    for axp in axs:
        axp.axhline(0.0, color='k', lw=0.8)
        axp.set_xlabel('Input (ranked)')
        axp.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    save(fig, '05_uncertainty_ranking.png')

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    for idx, key in enumerate(LEVEL_KEYS):
        axp = axs.flat[idx]
        lower, upper = float(ranges.loc[key, 'Min']), float(ranges.loc[key, 'Max'])
        edges = np.linspace(lower, upper, 31)
        axp.hist(base[key], bins=edges, density=True, color='0.75', edgecolor='w', label='sampled')
        x_grid = np.linspace(lower, upper, 300)
        pdf = input_distributions(ranges, wind)[key].pdf(x_grid)
        axp.plot(x_grid, pdf, color='C3', label='input distribution')
        axp.axvline(float(ranges.loc[key, 'Nominal']), color='k', ls=':', label='nominal')
        axp.set_xlim(lower, upper)
        dens_max = max(np.histogram(base[key], bins=edges, density=True)[0].max(), np.max(pdf))
        axp.set_ylim(0, dens_max * 1.15)
        axp.set_xlabel(f'{key} ({UNITS[key]})')
        axp.set_ylabel(f'Probability density (1/{UNITS[key]})')
        axp.set_title(f'{key}: N = {N_BASE} samples')
        axp.legend(fontsize=8)
        axp.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, '05_input_samples.png')

    # ------------------------------------------------------------ summary
    print('\n=== Risk convergence ===')
    for _, row in df_conv.iterrows():
        print(f"  N={row['N']:4d}: p_fail={row['p_fail']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] "
              f"({row['ci_method']}), mean loss {row['mean_loss_pct']:.5f} %, p95 loss {row['p95_loss_pct']:.5f} %, "
              f"mean rms {row['mean_rms_deg']:.4f} deg")
    print(f"  rms range {base['rms_error_deg'].min():.4f}-{base['rms_error_deg'].max():.4f} deg, "
          f"unsettled moves total {int(base['n_unsettled'].sum())}")
    print('\n=== Uncertainty ranking (paired, first 500 trials) ===')
    for _, row in df_rank.iterrows():
        print(f"  #{row['rank']} {row['input']:13s} p_fail {row['p_fail_baseline']:.4f} -> {row['p_fail_reduced']:.4f}, "
              f"mean rms {row['mean_rms_baseline_deg']:.4f} -> {row['mean_rms_reduced_deg']:.4f} deg")
    print('\n=== Credibility checks ===')
    print(f"  {'PASS' if width_ok else 'FAIL'}: CI width decreases with N: " +
          ', '.join(f'{w:.4f}' for w in widths))
    for key, sample_mean, nominal, dev in mean_rows:
        print(f"  {'PASS' if abs(dev) <= 5.0 else 'FAIL'}: sample mean of {key} = {sample_mean:.4f} vs nominal "
              f"{nominal:g} ({dev:+.2f} %, limit 5 %)")
    print(f'  total run time {time.perf_counter() - t_start:.0f} s')
    print('\n=== Files saved ===')
    for name in ('05_mc_trials.csv', '05_risk_convergence.csv', '05_uncertainty_ranking.csv'):
        print(f'  table : results/tables/{name}')
    for name in saved_figures:
        print(f'  figure: results/figures/{name}')


if __name__ == '__main__':
    main()
