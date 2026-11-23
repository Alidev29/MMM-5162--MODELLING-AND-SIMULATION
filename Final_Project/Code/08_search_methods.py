# File: 08_search_methods.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Two further simulation-based search methods for the controller settings - random search
#          and local (hill-climbing) search from the grid best and from the baseline - compared with the grid search and PSO of study 06 on the
#          same objective and the same 5 common-random-number scenarios
# Inputs: data/*.csv parameter files; the evaluator, objective and bounds of 06_optimization.py;
#         results/tables/06_grid_results.csv, 06_method_comparison.csv
# Outputs: tables 08_random_search.csv, 08_local_search_path.csv, 08_method_comparison_all.csv;
#          figure 08_search_paths.png (python 08_search_methods.py --plot-only redraws it from the tables)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import sys
import time
import importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from params import SEED, project_paths
from analysis_tools import random_points, local_search

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# Evaluator, objective, bounds and scenarios are those of study 06 (imported, not copied).
_spec = importlib.util.spec_from_file_location('opt06', os.path.join(script_dir, '06_optimization.py'))
opt06 = importlib.util.module_from_spec(_spec)
sys.modules['opt06'] = opt06          # so worker processes can unpickle opt06.run_scenario
_spec.loader.exec_module(opt06)

# ---------------------------------------------------------------- parameters
N_RANDOM = 48                                   # same budget as the 48-point grid
LOCAL_STEPS = (15.0, 0.025)                     # update_s (s), tol_deg (deg)
BOUNDS = [opt06.BOUNDS['update_s'], opt06.BOUNDS['tol_deg']]
RANDOM_SEED_OFFSET = 8                          # generator default_rng([SEED, 8]): distinct from the PSO stream


def plot_figure(grid, df_rand, df_path, df_all, starts):
    """Decision-space figure from the result tables (called after the runs, or alone with --plot-only)."""
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(9, 6))
    feas = grid['feasible'].astype(bool)
    ax.scatter(grid.loc[feas, 'update_s'], grid.loc[feas, 'tol_deg'], s=30, marker='s', color='0.7',
               label='grid points (feasible)')
    ax.scatter(grid.loc[~feas, 'update_s'], grid.loc[~feas, 'tol_deg'], s=30, marker='x', color='0.5',
               label='grid points (infeasible)')
    rf = df_rand['feasible'].astype(bool)
    ax.scatter(df_rand.loc[rf, 'update_s'], df_rand.loc[rf, 'tol_deg'], s=22, color='tab:blue',
               label='random samples (feasible)')
    ax.scatter(df_rand.loc[~rf, 'update_s'], df_rand.loc[~rf, 'tol_deg'], s=22, facecolors='none',
               edgecolors='tab:blue', label='random samples (infeasible)')
    for start_name, colour in zip(starts, ('tab:green', 'tab:purple')):
        sub = df_path[df_path['start'] == start_name]
        ax.plot(sub['update_s'], sub['tol_deg'], 'o-', color=colour, ms=5, lw=2, zorder=5,
                label=f'local search path from {start_name} ({len(sub) - 1} moves)')
    for row in df_all.itertuples():
        marker = 'P' if row.method.startswith('local') else {'grid search': 's', 'PSO': '*', 'random search': 'D'}[row.method]
        fill = {'PSO': 'gold', 'local search (grid best start)': 'tab:green',
                'local search (baseline start)': 'tab:purple'}.get(row.method, 'w')
        ax.plot(row.update_s, row.tol_deg, marker, ms=13, mec='k', mfc=fill, zorder=3,
                label=f'best of {row.method}')
    ax.set_xlabel('Update interval update_s (s)')
    ax.set_ylabel('Controller tolerance tol_deg (deg)')
    ax.set_title('Search methods in the decision space' + chr(10) + '(same objective and scenarios)')
    ax.set_xlim(BOUNDS[0][0] - 30, BOUNDS[0][1] + 30)
    ax.set_ylim(BOUNDS[1][0] - 0.02, BOUNDS[1][1] + 0.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.savefig(os.path.join(figures_dir, '08_search_paths.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_only():
    """Redraw 08_search_paths.png from the saved tables in results/tables without new simulations."""
    grid = pd.read_csv(os.path.join(tables_dir, '06_grid_results.csv'))
    df_rand = pd.read_csv(os.path.join(tables_dir, '08_random_search.csv'))
    df_path = pd.read_csv(os.path.join(tables_dir, '08_local_search_path.csv'))
    df_all = pd.read_csv(os.path.join(tables_dir, '08_method_comparison_all.csv'))
    plot_figure(grid, df_rand, df_path, df_all, list(dict.fromkeys(df_path['start'])))
    print('  figure: results/figures/08_search_paths.png')


def main():
    t_start = time.perf_counter()
    print('=== Parameters ===')
    print(f'  objective and scenarios of study 06: {opt06.N_SCENARIOS} fixed trials, date {opt06.DATE}, '
          f'J = net energy - {opt06.PENALTY_WH_PER_DEG:g}*max(0, rms - {opt06.RMS_LIMIT_DEG})')
    print(f'  bounds update_s {BOUNDS[0]} s, tol_deg {BOUNDS[1]} deg; random search {N_RANDOM} points '
          f'(rng [{SEED}, {RANDOM_SEED_OFFSET}]); local search steps {LOCAL_STEPS}')
    grid = pd.read_csv(os.path.join(tables_dir, '06_grid_results.csv'))
    cmp06 = pd.read_csv(os.path.join(tables_dir, '06_method_comparison.csv')).set_index('method')
    saved_tables, saved_figures = [], []

    with ProcessPoolExecutor(max_workers=opt06.N_WORKERS) as pool:
        # -------------------------------------------------------- random search
        ev_rand = opt06.Evaluator(pool)
        pts = random_points(BOUNDS, N_RANDOM, np.random.default_rng([SEED, RANDOM_SEED_OFFSET]))
        vals = ev_rand.evaluate_many([tuple(p) for p in pts])
        df_rand = pd.DataFrame([{'update_s': u, 'tol_deg': t, 'net_energy_Wh': net, 'rms_error_deg': rms,
                                 'J': opt06.objective_J(net, rms), 'feasible': rms <= opt06.RMS_LIMIT_DEG}
                                for (u, t), (net, rms) in zip(pts, vals)])
        df_rand.round(6).to_csv(os.path.join(tables_dir, '08_random_search.csv'), index=False)
        saved_tables.append('08_random_search.csv')
        print(f'  random search: {N_RANDOM} evaluations in {time.perf_counter() - t_start:.0f} s', flush=True)

        # -------------------------------------------------------- local search from two starts
        ev_loc = opt06.Evaluator(pool)          # shared cache: a point is simulated once

        def evaluate(point):
            net, rms = ev_loc.evaluate(*point)
            return opt06.objective_J(net, rms), rms <= opt06.RMS_LIMIT_DEG

        starts = {'grid best': (float(cmp06.loc['grid search', 'update_s']), float(cmp06.loc['grid search', 'tol_deg'])),
                  'baseline': tuple(float(v) for v in opt06.BASELINE)}
        path_rows, local_evals = [], {}
        for start_name, start in starts.items():
            n_before = len(ev_loc.cache)
            path, n_calls = local_search(evaluate, start, LOCAL_STEPS, BOUNDS)
            new_points = len(ev_loc.cache) - n_before
            local_evals[start_name] = n_calls          # objective evaluations requested by this run
            for i, (pt, j) in enumerate(path):
                net, rms = ev_loc.evaluate(*pt)
                path_rows.append({'start': start_name, 'step': i, 'update_s': pt[0], 'tol_deg': pt[1], 'J': j,
                                  'net_energy_Wh': net, 'rms_error_deg': rms})
            print(f'  local search from {start_name} {start}: {len(path) - 1} moves, {new_points} new evaluations, '
                  f'total {time.perf_counter() - t_start:.0f} s', flush=True)
        df_path = pd.DataFrame(path_rows)
        df_path.round(6).to_csv(os.path.join(tables_dir, '08_local_search_path.csv'), index=False)
        saved_tables.append('08_local_search_path.csv')

    # ------------------------------------------------------------ comparison of the four methods
    best_rand = df_rand[df_rand['feasible']].sort_values('J', ascending=False).iloc[0]
    rows = []
    for method in ('grid search', 'PSO'):
        r = cmp06.loc[method]
        rows.append({'method': method, 'update_s': r['update_s'], 'tol_deg': r['tol_deg'],
                     'net_energy_Wh': r['net_energy_Wh'], 'rms_error_deg': r['rms_error_deg'], 'J': r['J'],
                     'n_evaluations': int(r['n_evaluations']), 'source': 'study 06'})
    rows.append({'method': 'random search', 'update_s': best_rand['update_s'], 'tol_deg': best_rand['tol_deg'],
                 'net_energy_Wh': best_rand['net_energy_Wh'], 'rms_error_deg': best_rand['rms_error_deg'],
                 'J': best_rand['J'], 'n_evaluations': N_RANDOM, 'source': 'study 08'})
    for start_name in starts:
        sub = df_path[df_path['start'] == start_name]
        end = sub.iloc[-1]
        rows.append({'method': f'local search ({start_name} start)', 'update_s': end['update_s'],
                     'tol_deg': end['tol_deg'], 'net_energy_Wh': end['net_energy_Wh'],
                     'rms_error_deg': end['rms_error_deg'], 'J': end['J'],
                     'n_evaluations': local_evals[start_name], 'source': f'study 08, {len(sub) - 1} moves'})
    df_all = pd.DataFrame(rows)
    df_all['J_minus_grid_mWh'] = 1000.0 * (df_all['J'] - float(cmp06.loc['grid search', 'J']))
    df_all.round(6).to_csv(os.path.join(tables_dir, '08_method_comparison_all.csv'), index=False)
    saved_tables.append('08_method_comparison_all.csv')

    # ------------------------------------------------------------ figure: decision space
    plot_figure(grid, df_rand, df_path, df_all, list(starts))
    saved_figures.append('08_search_paths.png')

    # ------------------------------------------------------------ checks and conclusion
    in_bounds = bool(df_rand['update_s'].between(*BOUNDS[0]).all() and df_rand['tol_deg'].between(*BOUNDS[1]).all())
    climbs = all(bool(np.all(np.diff(df_path[df_path['start'] == name]['J'].to_numpy()) > 0.0)) for name in starts)
    grid_start = abs(df_path[df_path['start'] == 'grid best'].iloc[0]['J'] - float(cmp06.loc['grid search', 'J'])) < 1e-6
    print('\n=== Checks ===')
    print(f"  {'PASS' if in_bounds else 'FAIL'}: all random samples inside the bounds")
    print(f"  {'PASS' if grid_start else 'FAIL'}: local search starts at the grid best with the same J as study 06 "
          f"(common random numbers reproduce it)")
    print(f"  {'PASS' if climbs else 'FAIL'}: J increases at every local-search move")
    print('\n=== Method comparison ===')
    print(df_all.round(4).to_string(index=False))
    best = df_all.sort_values('J', ascending=False).iloc[0]
    print('\n=== Engineering conclusion ===')
    print(f"  Best of the methods: {best['method']} at {best['update_s']:.0f} s / {best['tol_deg']:.3f} deg; "
          f"all lie within {df_all['J'].max() - df_all['J'].min():.4f} Wh/day of each other, so the choice of "
          f"search method does not change the recommendation of study 06.")
    print(f'  total run time {time.perf_counter() - t_start:.0f} s')
    print('\n=== Files saved ===')
    for name in saved_tables:
        print(f'  table : results/tables/{name}')
    for name in saved_figures:
        print(f'  figure: results/figures/{name}')


if __name__ == '__main__':
    plot_only() if '--plot-only' in sys.argv else main()
