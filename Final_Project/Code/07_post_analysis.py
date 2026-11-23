# File: 07_post_analysis.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Post-analysis of the saved Monte Carlo and optimisation results without new simulation:
#          input-sample checks, output analysis and required N, reliability, threshold sensitivity,
#          failure region, global sensitivity (Spearman, PRCC) against one-at-a-time, Pareto front,
#          and the classification of each input uncertainty
# Inputs: results/tables/05_mc_trials.csv, 05_risk_convergence.csv, 04_normalized_sensitivity.csv,
#         06_grid_results.csv, 02_selected_wind_distribution.csv; data/disturbance_ranges.csv
# Outputs: tables 07_input_checks.csv, 07_output_analysis.csv, 07_threshold_curve.csv,
#          07_failure_region.csv, 07_global_sensitivity.csv, 07_pareto_front.csv, 07_uncertainty_types.csv;
#          figures 07_threshold_curve.png, 07_failure_region.png, 07_global_vs_oat.png, 07_pareto_front.png
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from params import SEED, load_disturbance_ranges, project_paths
from weather_data import load_selected_distribution
from analysis_tools import t_mean_ci, wilson_ci, proportion_se, required_n, prcc, pareto_mask

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

_spec = importlib.util.spec_from_file_location('mc05', os.path.join(script_dir, '05_monte_carlo_risk.py'))
mc05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc05)

# ---------------------------------------------------------------- parameters
FAIL_LIMIT_DEG = mc05.FAIL_LIMIT_DEG
THRESHOLDS_DEG = (0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00)
CPV_BAND_DEG = (0.3, 0.5)                    # tighter accuracy needed by concentrator PV (Section 5.4)
HALF_WIDTHS = (0.005, 0.01)                  # target 95 % half-widths on p_fail
REGION_V_MPS = 2.2                           # weak wind ...
REGION_B_DEG = (0.195, 0.2)                  # ... and large backlash (two boundary definitions)
INPUTS = ('tau_d_s', 'v_mean_mps', 'backlash_deg', 'Tc_Nm')
LABELS = {'tau_d_s': 'Timing delay tau_d', 'v_mean_mps': 'Mean wind v', 'backlash_deg': 'Backlash b',
          'Tc_Nm': 'Coulomb friction Tc'}
UNCERTAINTY_TYPES = [   # written as data: type and reason for each input (Topic 7)
    ('v_mean_mps', 'aleatory', 'Day-to-day weather variability; it cannot be reduced by better knowledge, '
                               'only described by the fitted Weibull distribution.'),
    ('backlash_deg', 'epistemic', 'A fixed property of the installed gearbox that is unknown here; a '
                                  'datasheet or a measurement would remove most of it.'),
    ('Tc_Nm', 'epistemic', 'A fixed property of the motor and gearbox that is unknown here; it can be '
                           'measured on the real drive.'),
    ('tau_d_s', 'mixed', 'The clock offset after synchronisation is a design choice (epistemic), while the '
                         'drift between synchronisations varies from day to day (aleatory).'),
]


def save_fig(fig, name, saved):
    fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved.append(name)


def main():
    trials = pd.read_csv(os.path.join(tables_dir, '05_mc_trials.csv'))
    base = trials[trials['set'] == 'baseline'].reset_index(drop=True)
    n = len(base)
    rms = base['rms_error_deg'].to_numpy()
    ranges = load_disturbance_ranges(os.path.join(paths['data'], 'disturbance_ranges.csv'))
    wind = load_selected_distribution(os.path.join(tables_dir, '02_selected_wind_distribution.csv'))
    dists = mc05.input_distributions(ranges, wind)
    saved_tables, saved_figures = [], []
    print('=== Parameters ===')
    print(f'  {n} baseline Monte Carlo days (study 05, SEED {SEED}), failure: rms > {FAIL_LIMIT_DEG} deg')

    def save_table(df, name):
        df.round(6).to_csv(os.path.join(tables_dir, name), index=False)
        saved_tables.append(name)

    # ------------------------------------------------------------ 1. input-sample checks (Topic 4)
    rows = []
    for key in INPUTS:
        x = base[key].to_numpy()
        target_mean, target_sd = float(dists[key].mean()), float(dists[key].std())
        rows.append({'quantity': key, 'kind': 'input', 'target_mean': target_mean, 'target_sd': target_sd,
                     'sample_mean': x.mean(), 'sample_sd': x.std(ddof=1), 'sample_cv': x.std(ddof=1) / x.mean(),
                     'mean_error_pct': 100.0 * (x.mean() - target_mean) / target_mean,
                     'sd_error_pct': 100.0 * (x.std(ddof=1) - target_sd) / target_sd})
    for key in ('rms_error_deg', 'energy_loss_pct'):
        x = base[key].to_numpy()
        rows.append({'quantity': key, 'kind': 'output', 'target_mean': np.nan, 'target_sd': np.nan,
                     'sample_mean': x.mean(), 'sample_sd': x.std(ddof=1), 'sample_cv': x.std(ddof=1) / x.mean(),
                     'mean_error_pct': np.nan, 'sd_error_pct': np.nan})
    df_inputs = pd.DataFrame(rows)
    save_table(df_inputs, '07_input_checks.csv')

    # ------------------------------------------------------------ 2-3. output analysis and reliability
    n_fail = int(np.sum(rms > FAIL_LIMIT_DEG))
    p_hat, p_lo, p_hi = wilson_ci(n_fail, n)
    out_rows = []
    for key, label in (('rms_error_deg', 'mean daily RMS error (deg)'), ('energy_loss_pct', 'mean energy loss (%)')):
        mean, lo, hi, se = t_mean_ci(base[key].to_numpy())
        out_rows.append({'quantity': label, 'value': mean, 'ci_low': lo, 'ci_high': hi, 'se': se,
                         'method': f't interval, {n - 1} dof'})
    out_rows.append({'quantity': 'failure probability p_fail', 'value': p_hat, 'ci_low': p_lo, 'ci_high': p_hi,
                     'se': proportion_se(p_hat, n), 'method': 'Wilson score'})
    out_rows.append({'quantity': 'reliability R = 1 - p_fail', 'value': 1.0 - p_hat, 'ci_low': 1.0 - p_hi,
                     'ci_high': 1.0 - p_lo, 'se': proportion_se(p_hat, n), 'method': 'Wilson score, transformed'})
    for h in HALF_WIDTHS:
        out_rows.append({'quantity': f'N for +/-{100 * h:g} % half-width on p_fail', 'value': required_n(p_hat, h),
                         'ci_low': np.nan, 'ci_high': np.nan, 'se': np.nan,
                         'method': f'normal approximation at p = {p_hat:g}'})
    df_out = pd.DataFrame(out_rows)
    save_table(df_out, '07_output_analysis.csv')

    # ------------------------------------------------------------ 4. threshold sensitivity
    thr_rows = []
    for limit in THRESHOLDS_DEG:
        k = int(np.sum(rms > limit))
        p, lo, hi = wilson_ci(k, n)
        thr_rows.append({'limit_deg': limit, 'n_fail': k, 'p_fail': p, 'ci_low': lo, 'ci_high': hi})
    df_thr = pd.DataFrame(thr_rows)
    save_table(df_thr, '07_threshold_curve.csv')

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(*CPV_BAND_DEG, color='tab:orange', alpha=0.15, label='accuracy band of CPV (0.3-0.5 deg)')
    ax.fill_between(df_thr['limit_deg'], 100 * df_thr['ci_low'], 100 * df_thr['ci_high'], color='tab:blue',
                    alpha=0.25, label='Wilson 95 % interval')
    ax.plot(df_thr['limit_deg'], 100 * df_thr['p_fail'], 'o-', color='tab:blue', label='p_fail')
    ax.axvline(FAIL_LIMIT_DEG, color='r', ls='--', label=f'design limit {FAIL_LIMIT_DEG} deg')
    ax.set_xlabel('Accuracy limit on the daily RMS error (deg)')
    ax.set_ylabel('Failure probability (%)')
    ax.set_title(f'Failure probability against the accuracy limit ({n} days)')
    ax.set_xlim(THRESHOLDS_DEG[0] - 0.02, THRESHOLDS_DEG[-1] + 0.02)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    save_fig(fig, '07_threshold_curve.png', saved_figures)

    # ------------------------------------------------------------ 5. failure region
    v, b = base['v_mean_mps'].to_numpy(), base['backlash_deg'].to_numpy()
    failed = rms > FAIL_LIMIT_DEG
    reg_rows = []
    for b_min, strict in ((REGION_B_DEG[0], False), (REGION_B_DEG[1], True)):
        inside = (v < REGION_V_MPS) & ((b > b_min) if strict else (b >= b_min))
        reg_rows.append({'region': f"v < {REGION_V_MPS} m/s and b {'>' if strict else '>='} {b_min} deg",
                         'days_inside': int(inside.sum()), 'failed_inside': int((inside & failed).sum()),
                         'days_outside': int((~inside).sum()), 'failed_outside': int((~inside & failed).sum()),
                         'p_fail_inside': float((inside & failed).sum() / inside.sum())})
    df_reg = pd.DataFrame(reg_rows)
    save_table(df_reg, '07_failure_region.csv')

    main_region = reg_rows[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(v[~failed], b[~failed], s=8, color='0.6', label=f'pass ({int((~failed).sum())} days)')
    ax.scatter(v[failed], b[failed], s=28, color='r', marker='x', label=f'fail, RMS > {FAIL_LIMIT_DEG} deg ({n_fail} days)')
    ax.add_patch(plt.Rectangle((0.0, REGION_B_DEG[0]), REGION_V_MPS, b.max() - REGION_B_DEG[0] + 0.02,
                               fill=False, ec='k', ls='--', lw=1.5,
                               label=f'v < {REGION_V_MPS} m/s, δ ≥ {REGION_B_DEG[0]} deg'))
    ax.annotate(f"{main_region['failed_inside']} of {main_region['days_inside']} days failed\n"
                f"(all {n_fail} failures inside)", xy=(REGION_V_MPS, b.max()), xytext=(REGION_V_MPS + 0.6, b.max() - 0.05),
                arrowprops={'arrowstyle': '->'}, fontsize=12)
    ax.set_xlabel('Mean wind speed v (m/s)')
    ax.set_ylabel('Backlash δ (deg)')
    ax.set_title('Failure region of the Monte Carlo days')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    save_fig(fig, '07_failure_region.png', saved_figures)

    # ------------------------------------------------------------ 6. global sensitivity vs one-at-a-time
    X = base[list(INPUTS)].to_numpy()
    prcc_vals = prcc(X, rms)
    oat = pd.read_csv(os.path.join(tables_dir, '04_normalized_sensitivity.csv')).set_index('disturbance')
    gs_rows = []
    for i, key in enumerate(INPUTS):
        gs_rows.append({'input': key, 'spearman': stats.spearmanr(X[:, i], rms)[0], 'prcc': prcc_vals[i],
                        'oat_S_rms': float(oat.loc[key, 'S_rms_error_deg']), 'oat_rank': int(oat.loc[key, 'rank'])})
    df_gs = pd.DataFrame(gs_rows)
    df_gs['prcc_rank'] = df_gs['prcc'].abs().rank(ascending=False).astype(int)
    save_table(df_gs, '07_global_sensitivity.csv')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    plt.rcParams.update({'font.size': 13})
    y_pos = np.arange(len(INPUTS))
    names = [LABELS[k] for k in INPUTS]
    axs[0].barh(y_pos - 0.2, df_gs['prcc'], 0.4, label='PRCC')
    axs[0].barh(y_pos + 0.2, df_gs['spearman'], 0.4, label='Spearman')
    axs[0].set_title(f'(a) Global: rank correlation\nwith RMS error ({n} days)', fontsize=13)
    axs[0].set_xlabel('Correlation coefficient (-)', fontsize=13)
    axs[1].barh(y_pos, df_gs['oat_S_rms'], 0.5, color='tab:green')
    axs[1].set_title('(b) Local: one-at-a-time\nnormalised sensitivity S', fontsize=13)
    axs[1].set_xlabel('S = (dy/dx)(x0/y0) (-)', fontsize=13)
    for axp in axs:
        axp.set_yticks(y_pos, names, fontsize=12)
        axp.tick_params(axis='x', labelsize=12)
        axp.axvline(0, color='k', lw=0.8)
        axp.invert_yaxis()
        axp.grid(alpha=0.3, axis='x')
    axs[0].legend(fontsize=12)
    fig.tight_layout()
    save_fig(fig, '07_global_vs_oat.png', saved_figures)
    plt.rcParams.update({'font.size': 10})

    # ------------------------------------------------------------ 7. Pareto front of the 06 grid
    grid = pd.read_csv(os.path.join(tables_dir, '06_grid_results.csv'))
    grid['pareto'] = pareto_mask(grid['net_energy_Wh'], grid['rms_error_deg'])
    front = grid[grid['pareto']].sort_values('rms_error_deg').reset_index(drop=True)
    save_table(front[['update_s', 'tol_deg', 'net_energy_Wh', 'rms_error_deg', 'feasible']], '07_pareto_front.csv')

    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    dom = grid[~grid['pareto']]
    for axp in axs:
        axp.scatter(dom['rms_error_deg'], dom['net_energy_Wh'], s=18, color='0.6', label='dominated grid points')
        axp.plot(front['rms_error_deg'], front['net_energy_Wh'], 'o-', color='tab:blue', label='Pareto front')
        axp.ticklabel_format(axis='y', useOffset=False)
        axp.set_xlabel('Mean RMS error (deg), minimise')
        axp.set_ylabel('Net energy (Wh/day), maximise')
        axp.grid(alpha=0.3)
    axs[0].axvline(FAIL_LIMIT_DEG, color='r', ls='--', label=f'accuracy limit {FAIL_LIMIT_DEG} deg')
    axs[0].set_title('(a) All 48 grid settings')
    axs[0].legend(fontsize=11, loc='lower left')
    zoom = grid[grid['rms_error_deg'] <= 0.3]               # the front and its nearest neighbours
    axs[1].set_xlim(zoom['rms_error_deg'].min() - 0.01, zoom['rms_error_deg'].max() + 0.01)
    span = zoom['net_energy_Wh'].max() - zoom['net_energy_Wh'].min()
    axs[1].set_ylim(zoom['net_energy_Wh'].min() - 0.1 * span, zoom['net_energy_Wh'].max() + 0.15 * span)
    offsets = [(10, -14), (-20, 12), (8, -18), (-105, 6), (-40, 10)]
    for (_, row), off in zip(front.iterrows(), offsets):
        axs[1].annotate(f"{row['update_s']:.0f} s / {row['tol_deg']:.2f} deg",
                        (row['rms_error_deg'], row['net_energy_Wh']), textcoords='offset points', xytext=off,
                        fontsize=11)
    axs[1].set_title('(b) Zoom on the Pareto front')
    fig.tight_layout()
    save_fig(fig, '07_pareto_front.png', saved_figures)
    plt.rcParams.update({'font.size': 10})

    # ------------------------------------------------------------ 8. uncertainty classification
    df_types = pd.DataFrame(UNCERTAINTY_TYPES, columns=['input', 'type', 'reason'])
    save_table(df_types, '07_uncertainty_types.csv')

    # ------------------------------------------------------------ checks and conclusion
    conv = pd.read_csv(os.path.join(tables_dir, '05_risk_convergence.csv')).set_index('N')
    same_p = abs(conv.loc[n, 'p_fail'] - p_hat) < 1e-12 and abs(conv.loc[n, 'ci_low'] - p_lo) < 1e-6
    within = df_inputs[df_inputs['kind'] == 'input']['mean_error_pct'].abs().max()
    print('\n=== Checks ===')
    print(f"  {'PASS' if same_p else 'FAIL'}: p_fail and Wilson CI equal to table 05 ({p_hat:.3f} [{p_lo:.4f}, {p_hi:.4f}])")
    print(f"  {'PASS' if within < 5.0 else 'FAIL'}: sample means of the inputs within 5 % of the targets "
          f"(largest {within:.2f} %)")
    print(f"  {'PASS' if main_region['failed_outside'] == 0 else 'FAIL'}: every failure inside the region "
          f"v < {REGION_V_MPS} m/s, b >= {REGION_B_DEG[0]} deg")
    print('\n=== Results ===')
    print(df_inputs.round(4).to_string(index=False))
    print(df_out.round(5).to_string(index=False))
    print(df_thr.round(4).to_string(index=False))
    print(df_reg.to_string(index=False))
    print(df_gs.round(3).to_string(index=False))
    print(front.round(4).to_string(index=False))
    base_row = grid[(grid['update_s'] == 120) & (grid['tol_deg'] == 0.10)].iloc[0]
    best_acc = front.iloc[0]
    print('\n=== Engineering conclusion ===')
    print(f"  Reliability R = {1 - p_hat:.3f} [{1 - p_hi:.3f}, {1 - p_lo:.3f}]; a +/-0.5 % half-width on p_fail "
          f"would need N = {required_n(p_hat, 0.005)} days.")
    print(f"  Most accurate Pareto setting {best_acc['update_s']:.0f} s / {best_acc['tol_deg']:.2f} deg lowers the RMS "
          f"error from {base_row['rms_error_deg']:.3f} to {best_acc['rms_error_deg']:.3f} deg for "
          f"{base_row['net_energy_Wh'] - best_acc['net_energy_Wh']:.3f} Wh/day.")
    print('\n=== Files saved ===')
    for name in saved_tables:
        print(f'  table : results/tables/{name}')
    for name in saved_figures:
        print(f'  figure: results/figures/{name}')


if __name__ == '__main__':
    main()
