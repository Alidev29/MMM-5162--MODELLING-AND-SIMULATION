# File: 06_optimization.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Simulation-based optimisation of the controller update interval and tolerance:
#          grid search and particle swarm optimisation, then a Monte Carlo robustness check
# Inputs: data/*.csv parameter files (update_s and tol_deg are the decision variables),
#         input distributions and sampling of script 05, results/tables/vibration_curve.csv
# Outputs: tables 06_grid_results.csv, 06_pso_history.csv, 06_method_comparison.csv, 06_robustness.csv;
#          figures 06_grid_net_energy.png, 06_grid_rms_error.png, 06_pso_convergence.png,
#          06_baseline_vs_optimum.png
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import time
import importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params import SEED, project_paths
from model_checks import check_parameters, check_levels
from day_simulation import simulate_day

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# Sampling and context are shared with script 05 (same input distributions and trial seeds).
_spec = importlib.util.spec_from_file_location('mc05', os.path.join(script_dir, '05_monte_carlo_risk.py'))
mc05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc05)

# ---------------------------------------------------------------- parameters
DATE = mc05.DATE
N_SCENARIOS = 5                     # fixed trials k = 0..4 (common random numbers)
N_ROBUST = 200
RMS_LIMIT_DEG = 0.5
PENALTY_WH_PER_DEG = 1000.0
BOUNDS = {'update_s': (60.0, 1200.0), 'tol_deg': (0.05, 0.5)}
GRID_UPDATE_S = (60, 120, 180, 300, 450, 600, 900, 1200)
GRID_TOL_DEG = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
PSO_SETTINGS = {'n_particles': 10, 'n_iter': 10, 'w': 0.7, 'c1': 1.5, 'c2': 1.5}
BASELINE = (120.0, 0.10)
N_WORKERS = min(os.cpu_count(), 8)   # each worker holds ~300 MB (pandas/pvlib/scipy); 7.5 GB RAM machine

_CONTEXT = None


def _context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = mc05.load_context()
    return _CONTEXT


def run_scenario(job):
    """job = (update_s, tol_deg, k): trial k of the Monte Carlo baseline sampling with the given settings."""
    update_s, tol_deg, k = job
    c = _context()
    ctrl = dict(c['ctrl'], update_s=float(update_s), tol_deg=float(tol_deg))
    check_parameters(c['site'], c['pv'], c['drive'], ctrl)
    dists = mc05.input_distributions(c['ranges'], c['wind'])
    levels, rng = mc05.sample_levels(k, dists, c['ranges'], 1.0)
    check_levels(levels, c['ranges'])
    res = simulate_day(DATE, c['site'], c['pv'], c['sd'], c['drive'], ctrl, levels, c['t_air'],
                       vibration_curve=c['vib'], rng=rng)
    return res['net_energy_Wh'], res['rms_error_deg']


def objective_J(net_wh, rms_deg):
    return net_wh - PENALTY_WH_PER_DEG * max(0.0, rms_deg - RMS_LIMIT_DEG)


class Evaluator:
    """evaluate(update_s, tol_deg) -> (mean_net_energy_Wh, mean_rms_error_deg) over the fixed scenarios,
    with a cache and batched parallel evaluation of many points."""

    def __init__(self, pool):
        self.pool = pool
        self.cache = {}
        self.n_simulations = 0

    @staticmethod
    def key(update_s, tol_deg):
        return (round(float(update_s), 6), round(float(tol_deg), 6))

    def evaluate_many(self, points):
        new = sorted({self.key(*p) for p in points} - set(self.cache))
        jobs = [(u, t, k) for (u, t) in new for k in range(N_SCENARIOS)]
        outs = list(self.pool.map(run_scenario, jobs, chunksize=1))
        self.n_simulations += len(jobs)
        for idx, point in enumerate(new):
            block = np.array(outs[idx * N_SCENARIOS:(idx + 1) * N_SCENARIOS])
            self.cache[point] = (float(block[:, 0].mean()), float(block[:, 1].mean()))
        return [self.cache[self.key(*p)] for p in points]

    def evaluate(self, update_s, tol_deg):
        return self.evaluate_many([(update_s, tol_deg)])[0]


def pso(objective, bounds, n_particles=10, n_iter=10, w=0.7, c1=1.5, c2=1.5, rng=None):
    """Particle swarm maximisation; objective(points) -> list of J (batched). Positions clipped to bounds.
    Returns best position, best J, and per-iteration history of the global best."""
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    pos = lower + rng.random((n_particles, len(bounds))) * (upper - lower)
    vel = (rng.random((n_particles, len(bounds))) - 0.5) * (upper - lower) * 0.2
    values = np.array(objective(pos))
    best_pos, best_val = pos.copy(), values.copy()
    g_idx = int(np.argmax(best_val))
    history = [(0, float(best_val[g_idx]), *best_pos[g_idx])]
    for iteration in range(1, n_iter):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = w * vel + c1 * r1 * (best_pos - pos) + c2 * r2 * (best_pos[g_idx] - pos)
        pos = np.clip(pos + vel, lower, upper)
        values = np.array(objective(pos))
        improved = values > best_val
        best_pos[improved], best_val[improved] = pos[improved], values[improved]
        g_idx = int(np.argmax(best_val))
        history.append((iteration, float(best_val[g_idx]), *best_pos[g_idx]))
    return best_pos[g_idx], float(best_val[g_idx]), history


def mean_ci(values):
    half = mc05.Z_95 * np.std(values, ddof=1) / np.sqrt(len(values))
    return float(np.mean(values)), float(np.mean(values) - half), float(np.mean(values) + half)


def main():
    t_start = time.perf_counter()
    c = _context()
    print('=== Parameters ===')
    print(f'  date {DATE}, {N_SCENARIOS} fixed scenarios (trials k = 0..{N_SCENARIOS - 1} of script 05), SEED {SEED}')
    print(f"  bounds update_s {BOUNDS['update_s']} s, tol_deg {BOUNDS['tol_deg']} deg; "
          f'J = net energy - {PENALTY_WH_PER_DEG:g}*max(0, rms - {RMS_LIMIT_DEG})')
    print(f"  baseline update_s {c['ctrl']['update_s']:g} s, tol_deg {c['ctrl']['tol_deg']:g} deg; PSO {PSO_SETTINGS}")
    saved_tables, saved_figures = [], []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ev = Evaluator(pool)

        # -------------------------------------------------------- grid search
        grid_points = [(u, t) for u in GRID_UPDATE_S for t in GRID_TOL_DEG]
        grid_vals = ev.evaluate_many(grid_points)
        df_grid = pd.DataFrame([{'update_s': u, 'tol_deg': t, 'net_energy_Wh': net, 'rms_error_deg': rms,
                                 'J': objective_J(net, rms), 'feasible': rms <= RMS_LIMIT_DEG}
                                for (u, t), (net, rms) in zip(grid_points, grid_vals)])
        df_grid.round(6).to_csv(os.path.join(tables_dir, '06_grid_results.csv'), index=False)
        saved_tables.append('06_grid_results.csv')
        grid_sims = ev.n_simulations
        print(f'  grid: {len(grid_points)} evaluations in {time.perf_counter() - t_start:.0f} s', flush=True)

        # -------------------------------------------------------- PSO (own cache, so its count is its own)
        ev_pso = Evaluator(pool)
        objective = lambda pts: [objective_J(*v) for v in ev_pso.evaluate_many([tuple(p) for p in pts])]
        best_x, best_J, history = pso(objective, [BOUNDS['update_s'], BOUNDS['tol_deg']],
                                      rng=np.random.default_rng(SEED), **PSO_SETTINGS)
        df_hist = pd.DataFrame(history, columns=['iteration', 'best_J', 'best_update_s', 'best_tol_deg'])
        df_hist.round(6).to_csv(os.path.join(tables_dir, '06_pso_history.csv'), index=False)
        saved_tables.append('06_pso_history.csv')
        print(f'  PSO: {len(ev_pso.cache)} distinct evaluations, total {time.perf_counter() - t_start:.0f} s', flush=True)

        # -------------------------------------------------------- comparison
        best_grid = df_grid.sort_values('J', ascending=False).iloc[0]
        pso_net, pso_rms = ev_pso.evaluate(*best_x)
        base_net, base_rms = ev.evaluate(*BASELINE)
        df_cmp = pd.DataFrame([
            {'method': 'baseline', 'update_s': BASELINE[0], 'tol_deg': BASELINE[1], 'net_energy_Wh': base_net,
             'rms_error_deg': base_rms, 'feasible': base_rms <= RMS_LIMIT_DEG, 'n_evaluations': 1},
            {'method': 'grid search', 'update_s': best_grid['update_s'], 'tol_deg': best_grid['tol_deg'],
             'net_energy_Wh': best_grid['net_energy_Wh'], 'rms_error_deg': best_grid['rms_error_deg'],
             'feasible': bool(best_grid['feasible']), 'n_evaluations': len(grid_points)},
            {'method': 'PSO', 'update_s': best_x[0], 'tol_deg': best_x[1], 'net_energy_Wh': pso_net,
             'rms_error_deg': pso_rms, 'feasible': pso_rms <= RMS_LIMIT_DEG,
             'n_evaluations': PSO_SETTINGS['n_particles'] * PSO_SETTINGS['n_iter']},
        ])
        df_cmp['J'] = [objective_J(n, r) for n, r in zip(df_cmp['net_energy_Wh'], df_cmp['rms_error_deg'])]
        df_cmp.round(6).to_csv(os.path.join(tables_dir, '06_method_comparison.csv'), index=False)
        saved_tables.append('06_method_comparison.csv')
        optimum = df_cmp.iloc[1:].sort_values('J', ascending=False).iloc[0]
        opt_setting = (float(optimum['update_s']), float(optimum['tol_deg']))

        # -------------------------------------------------------- robustness (200 Monte Carlo trials)
        robust_rows = []
        robust_data = {}
        for label, (u, t) in (('baseline', BASELINE), ('optimum', opt_setting)):
            outs = np.array(list(pool.map(run_scenario, [(u, t, k) for k in range(N_ROBUST)], chunksize=2)))
            robust_data[label] = outs
            n_fail = int(np.sum(outs[:, 1] > RMS_LIMIT_DEG))
            p_hat, p_lo, p_hi, method = mc05.failure_ci(n_fail, N_ROBUST)
            net_m, net_lo, net_hi = mean_ci(outs[:, 0])
            rms_m, rms_lo, rms_hi = mean_ci(outs[:, 1])
            robust_rows.append({'setting': label, 'update_s': u, 'tol_deg': t, 'N': N_ROBUST, 'p_fail': p_hat,
                                'p_fail_ci_low': p_lo, 'p_fail_ci_high': p_hi, 'ci_method': method,
                                'mean_net_energy_Wh': net_m, 'net_ci_low': net_lo, 'net_ci_high': net_hi,
                                'mean_rms_error_deg': rms_m, 'rms_ci_low': rms_lo, 'rms_ci_high': rms_hi})
        df_rob = pd.DataFrame(robust_rows)
        df_rob.round(6).to_csv(os.path.join(tables_dir, '06_robustness.csv'), index=False)
        saved_tables.append('06_robustness.csv')

    # ------------------------------------------------------------ figures
    def save(fig, name):
        fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_figures.append(name)

    u_idx = np.arange(len(GRID_UPDATE_S))
    t_idx = np.arange(len(GRID_TOL_DEG))
    for metric, name, clabel in (('net_energy_Wh', '06_grid_net_energy.png', 'Mean net energy (Wh/day)'),
                                 ('rms_error_deg', '06_grid_rms_error.png', 'Mean RMS pointing error (deg)')):
        z_grid = df_grid.pivot(index='tol_deg', columns='update_s', values=metric).to_numpy()
        rms_grid = df_grid.pivot(index='tol_deg', columns='update_s', values='rms_error_deg').to_numpy()
        fig, ax_plot = plt.subplots(figsize=(7.5, 5))
        mesh = ax_plot.pcolormesh(u_idx, t_idx, z_grid, shading='nearest', cmap='viridis')
        fig.colorbar(mesh, ax=ax_plot, label=clabel)
        if rms_grid.min() < RMS_LIMIT_DEG < rms_grid.max():
            cont = ax_plot.contour(u_idx, t_idx, rms_grid, levels=[RMS_LIMIT_DEG], colors='r', linewidths=2)
            ax_plot.clabel(cont, fmt=f'RMS = {RMS_LIMIT_DEG:g} deg (feasibility limit)')
        infeasible = rms_grid > RMS_LIMIT_DEG
        ti, ui = np.nonzero(infeasible)
        ax_plot.plot(ui, ti, 'rx', ms=8, label='infeasible (RMS > 0.5 deg)')
        bi = GRID_UPDATE_S.index(int(best_grid['update_s']))
        bj = GRID_TOL_DEG.index(float(best_grid['tol_deg']))
        ax_plot.plot(bi, bj, 'w*', ms=16, mec='k', label='grid optimum')
        ax_plot.set_xticks(u_idx, [str(u) for u in GRID_UPDATE_S])
        ax_plot.set_yticks(t_idx, [f'{t:.2f}' for t in GRID_TOL_DEG])
        ax_plot.set_xlim(-0.5, len(GRID_UPDATE_S) - 0.5)
        ax_plot.set_ylim(-0.5, len(GRID_TOL_DEG) - 0.5)
        ax_plot.set_xlabel('Update interval update_s (s)')
        ax_plot.set_ylabel('Controller tolerance tol_deg (deg)')
        ax_plot.set_title(f'Grid search: {clabel.lower()}')
        ax_plot.legend(loc='upper left', fontsize=8, framealpha=0.9)
        save(fig, name)

    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    ax_plot.plot(df_hist['iteration'], df_hist['best_J'], 'o-', label='PSO global best J')
    ax_plot.axhline(float(best_grid['J']), color='k', ls='--', label='best grid point')
    ax_plot.set_xlim(0, PSO_SETTINGS['n_iter'] - 1)
    j_all = np.concatenate([df_hist['best_J'].to_numpy(), [best_grid['J']]])
    pad = max(1e-3, 0.25 * (j_all.max() - j_all.min()))   # J varies by only ~mWh: range from the data
    ax_plot.set_ylim(j_all.min() - pad, j_all.max() + pad)
    ax_plot.ticklabel_format(axis='y', useOffset=False)
    ax_plot.set_xlabel('Iteration (-)')
    ax_plot.set_ylabel('Objective J (Wh/day)')
    ax_plot.set_title('Particle swarm convergence')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3)
    save(fig, '06_pso_convergence.png')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [f"baseline\n{BASELINE[0]:g} s, {BASELINE[1]:g} deg",
              f"optimum\n{opt_setting[0]:.0f} s, {opt_setting[1]:.2f} deg"]
    for axp, col, ylabel, lim in ((axs[0], 0, 'Daily net energy (Wh)', None),
                                  (axs[1], 1, 'Daily RMS pointing error (deg)', RMS_LIMIT_DEG)):
        data = [robust_data['baseline'][:, col], robust_data['optimum'][:, col]]
        axp.boxplot(data, tick_labels=labels)
        if lim is not None:
            axp.axhline(lim, color='r', ls='--', label=f'limit {lim:g} deg')
            axp.legend()
        lo = min(d.min() for d in data)
        hi = max(d.max() for d in data)
        if lim is not None:
            hi = max(hi, lim)
        span = hi - lo
        axp.set_ylim(lo - 0.1 * span, hi + 0.1 * span)
        axp.set_ylabel(ylabel)
        axp.grid(alpha=0.3, axis='y')
    axs[0].set_title(f'Net energy over {N_ROBUST} Monte Carlo days')
    axs[1].set_title(f'RMS error over {N_ROBUST} Monte Carlo days')
    fig.tight_layout()
    save(fig, '06_baseline_vs_optimum.png')

    # ------------------------------------------------------------ checks and conclusion
    pso_row = df_cmp[df_cmp['method'] == 'PSO'].iloc[0]
    j_ok = pso_row['J'] >= best_grid['J'] - 0.001 * abs(best_grid['J'])
    mono_ok = bool(np.all(np.diff(df_hist['best_J'].to_numpy()) >= 0.0))
    print('\n=== Method comparison ===')
    for _, row in df_cmp.iterrows():
        print(f"  {row['method']:12s} update {row['update_s']:7.1f} s, tol {row['tol_deg']:.3f} deg: net "
              f"{row['net_energy_Wh']:.3f} Wh, rms {row['rms_error_deg']:.4f} deg, J {row['J']:.3f}, "
              f"feasible {row['feasible']}, evaluations {row['n_evaluations']}")
    print('\n=== Checks ===')
    print(f"  {'PASS' if bool(pso_row['feasible']) else 'FAIL'}: PSO solution feasible (rms {pso_row['rms_error_deg']:.4f} deg)")
    print(f"  {'PASS' if j_ok else 'FAIL'}: J_PSO {pso_row['J']:.3f} >= J_grid {best_grid['J']:.3f} - 0.1 %")
    print(f"  {'PASS' if mono_ok else 'FAIL'}: PSO best-J history non-decreasing")
    print('\n=== Robustness (200 Monte Carlo days) ===')
    for _, row in df_rob.iterrows():
        print(f"  {row['setting']:8s} ({row['update_s']:.0f} s, {row['tol_deg']:.3f} deg): p_fail {row['p_fail']:.4f} "
              f"[{row['p_fail_ci_low']:.4f}, {row['p_fail_ci_high']:.4f}], net {row['mean_net_energy_Wh']:.3f} "
              f"[{row['net_ci_low']:.3f}, {row['net_ci_high']:.3f}] Wh, rms {row['mean_rms_error_deg']:.4f} "
              f"[{row['rms_ci_low']:.4f}, {row['rms_ci_high']:.4f}] deg")
    rob_base, rob_opt = df_rob.iloc[0], df_rob.iloc[1]
    gain = rob_opt['mean_net_energy_Wh'] - rob_base['mean_net_energy_Wh']
    active = abs(float(optimum['rms_error_deg']) - RMS_LIMIT_DEG) < 0.02
    print('\n=== Engineering conclusion ===')
    print(f"  Selected setting: update every {opt_setting[0]:.0f} s with tolerance {opt_setting[1]:.2f} deg "
          f"({optimum['method']}).")
    print(f"  Net energy change vs baseline: {gain:+.3f} Wh/day ({100 * gain / rob_base['mean_net_energy_Wh']:+.4f} %); "
          f"mean RMS error {rob_base['mean_rms_error_deg']:.3f} -> {rob_opt['mean_rms_error_deg']:.3f} deg.")
    print(f"  Constraint {'active' if active else 'not active'}: optimum RMS {optimum['rms_error_deg']:.4f} deg vs "
          f"limit {RMS_LIMIT_DEG} deg.")
    print(f'  total run time {time.perf_counter() - t_start:.0f} s')
    print('\n=== Files saved ===')
    for name in saved_tables:
        print(f'  table : results/tables/{name}')
    for name in saved_figures:
        print(f'  figure: results/figures/{name}')


if __name__ == '__main__':
    main()
