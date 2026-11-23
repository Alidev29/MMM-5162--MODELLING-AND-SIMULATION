# File: compare_variants.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Compares the results of variant v1 (results/tables) with variant v2 (results_v2/tables, Canadian Solar
#          CS1U-400MS module): the key energy numbers side by side, and every numeric column of every common table
#          checked for identity, with each column classed as energy (depends on the PV module) or non-energy
# Inputs: results/tables/*.csv, results_v2/tables/*.csv
# Outputs: results_v2/v1_vs_v2_comparison.csv, results_v2/v1_vs_v2_identity_check.csv
# Usage: python code/compare_variants.py   (after both variants have been run)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-23

import os
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(script_dir, '..'))
T1 = os.path.join(root, 'results', 'tables')
T2 = os.path.join(root, 'results_v2', 'tables')
OUT = os.path.join(root, 'results_v2')
TOL = 1e-9                     # absolute difference below which two values count as identical

# columns whose values come from the PV module (energy, power, gain, the objective J built from net energy)
ENERGY_COL = re.compile(r'energy|_Wh|^net_|gain_pct|loss|^J$|J_minus|best_J|^Model$|^Error_pct$|^Datasheet', re.I)
# tables that are wholly about the PV module
ENERGY_TABLES = {'01_pv_validation.csv'}
# tables whose rows are chosen by the energy objective (optimum, Pareto front, search paths): a changed choice also
# changes the non-energy columns of those rows, which is then a consequence of the energy change
DECISION_TABLES = {'06_grid_results.csv', '06_method_comparison.csv', '06_pso_history.csv', '06_robustness.csv',
                   '07_pareto_front.csv', '08_local_search_path.csv', '08_method_comparison_all.csv',
                   '08_random_search.csv'}


def tab(folder, name):
    return pd.read_csv(os.path.join(folder, name))


def identity_check():
    rows = []
    for name in sorted(set(os.listdir(T1)) & set(os.listdir(T2))):
        if not name.endswith('.csv'):
            continue
        a, b = tab(T1, name), tab(T2, name)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            rows.append({'table': name, 'column': '(shape)', 'category': 'n/a', 'identical': False,
                         'max_abs_diff': float('nan'), 'max_rel_diff_pct': float('nan'),
                         'note': f'shape {a.shape} vs {b.shape}'})
            continue
        # long-format tables name the quantity in a text column (e.g. 'pv_energy_Wh', 'mean energy loss (%)'):
        # a row whose text names an energy quantity holds energy values in every numeric column
        text_cols = [c for c in a.columns if not pd.api.types.is_numeric_dtype(a[c])]
        energy_row = a[text_cols].astype(str).apply(lambda r: bool(ENERGY_COL.search(' '.join(map(str, r)))), axis=1) \
            if text_cols else pd.Series(False, index=a.index)
        for col in a.columns:
            if not (pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col])):
                same = a[col].astype(str).equals(b[col].astype(str))
                if not same:
                    rows.append({'table': name, 'column': col, 'category': 'text', 'identical': False,
                                 'max_abs_diff': float('nan'), 'max_rel_diff_pct': float('nan'),
                                 'note': 'text differs'})
                continue
            diff = (b[col].astype(float) - a[col].astype(float)).abs()
            rel = diff / a[col].astype(float).abs().where(a[col] != 0) * 100.0
            energy = name in ENERGY_TABLES or bool(ENERGY_COL.search(col))
            plain = 'non-energy (decision table)' if name in DECISION_TABLES else 'non-energy'
            if energy:
                parts = [('energy', pd.Series(True, index=a.index), '')]
            elif energy_row.any():
                parts = [(plain, ~energy_row, 'rows not naming an energy quantity'),
                         ('energy', energy_row, 'rows naming an energy quantity')]
            else:
                parts = [(plain, pd.Series(True, index=a.index), '')]
            for category, mask, note in parts:
                d, r = diff[mask], rel[mask]
                rows.append({'table': name, 'column': col, 'category': category,
                             'identical': bool(d.max() <= TOL if len(d.dropna()) else True),
                             'max_abs_diff': float(d.max()), 'max_rel_diff_pct': float(r.max()), 'note': note})
    return pd.DataFrame(rows)


def key_numbers():
    out = []

    def add(group, quantity, unit, v1, v2):
        out.append({'group': group, 'quantity': quantity, 'unit': unit, 'v1': v1, 'v2': v2,
                    'change': v2 - v1, 'change_pct': 100.0 * (v2 / v1 - 1.0) if v1 else float('nan')})

    pv1, pv2 = (tab(t, '01_pv_validation.csv').set_index('Quantity') for t in (T1, T2))
    for q in ('P_max', 'V_mp', 'I_mp', 'V_oc', 'I_sc'):
        add('PV module at STC (datasheet)', q, pv1.loc[q, 'Unit'], pv1.loc[q, 'Datasheet'], pv2.loc[q, 'Datasheet'])
        add('PV module at STC (model)', q, pv1.loc[q, 'Unit'], pv1.loc[q, 'Model'], pv2.loc[q, 'Model'])
    d1, d2 = (tab(t, '01_baseline_days.csv').set_index('date') for t in (T1, T2))
    for date in d1.index:
        for col, unit in (('pv_energy_Wh', 'Wh'), ('fixed_tilt_energy_Wh', 'Wh'), ('gain_pct', '%'),
                          ('motor_energy_Wh', 'Wh'), ('net_energy_Wh', 'Wh'), ('rms_error_deg', 'deg'),
                          ('max_error_deg', 'deg')):
            add(f'Reference day {date}', col, unit, d1.loc[date, col], d2.loc[date, col])
    o1, o2 = tab(T1, '04_oat_sweep.csv'), tab(T2, '04_oat_sweep.csv')
    add('Sensitivity (04)', 'max daily energy loss, one at a time', '%', o1.energy_loss_pct.max(),
        o2.energy_loss_pct.max())
    m1, m2 = tab(T1, '04_map_2d.csv'), tab(T2, '04_map_2d.csv')
    add('Sensitivity (04)', 'max daily energy loss, 7x7 map', '%', m1.energy_loss_pct.max(), m2.energy_loss_pct.max())
    s1, s2 = (tab(t, '04_normalized_sensitivity.csv').set_index('disturbance') for t in (T1, T2))
    for dist in s1.index:
        add('Sensitivity (04)', f'S_energy_loss {dist}', '-', s1.loc[dist, 'S_energy_loss_pct'],
            s2.loc[dist, 'S_energy_loss_pct'])
        add('Sensitivity (04)', f'S_rms_error {dist}', '-', s1.loc[dist, 'S_rms_error_deg'],
            s2.loc[dist, 'S_rms_error_deg'])
    c1, c2 = (tab(t, '05_risk_convergence.csv').set_index('N') for t in (T1, T2))
    for col, unit in (('p_fail', '-'), ('ci_low', '-'), ('ci_high', '-'), ('mean_loss_pct', '%'),
                      ('p95_loss_pct', '%'), ('mean_rms_deg', 'deg')):
        add('Monte Carlo, 1000 days (05)', col, unit, c1.loc[1000, col], c2.loc[1000, col])
    t1, t2 = tab(T1, '05_mc_trials.csv'), tab(T2, '05_mc_trials.csv')
    b1, b2 = t1[t1['set'] == 'baseline'], t2[t2['set'] == 'baseline']
    add('Monte Carlo, 1000 days (05)', 'max daily energy loss', '%', b1.energy_loss_pct.max(), b2.energy_loss_pct.max())
    add('Monte Carlo, 1000 days (05)', 'mean net energy', 'Wh', b1.net_energy_Wh.mean(), b2.net_energy_Wh.mean())
    k1, k2 = (tab(t, '06_method_comparison.csv').set_index('method') for t in (T1, T2))
    for meth in k1.index:
        for col, unit in (('update_s', 's'), ('tol_deg', 'deg'), ('net_energy_Wh', 'Wh'), ('rms_error_deg', 'deg')):
            add(f'Optimisation (06), {meth}', col, unit, k1.loc[meth, col], k2.loc[meth, col])
        if meth != 'baseline':
            add(f'Optimisation (06), {meth}', 'J - J_baseline', 'mWh',
                1000 * (k1.loc[meth, 'J'] - k1.loc['baseline', 'J']), 1000 * (k2.loc[meth, 'J'] - k2.loc['baseline', 'J']))
    r1, r2 = (tab(t, '06_robustness.csv').set_index('setting') for t in (T1, T2))
    for st in r1.index:
        for col, unit in (('update_s', 's'), ('tol_deg', 'deg'), ('p_fail', '-'), ('mean_net_energy_Wh', 'Wh'),
                          ('mean_rms_error_deg', 'deg')):
            add(f'Robustness, 200 days (06), {st}', col, unit, r1.loc[st, col], r2.loc[st, col])
    add('Robustness, 200 days (06)', 'optimum - baseline net energy', 'mWh',
        1000 * (r1.loc['optimum', 'mean_net_energy_Wh'] - r1.loc['baseline', 'mean_net_energy_Wh']),
        1000 * (r2.loc['optimum', 'mean_net_energy_Wh'] - r2.loc['baseline', 'mean_net_energy_Wh']))
    f1, f2 = tab(T1, '07_failure_region.csv'), tab(T2, '07_failure_region.csv')
    for col in ('days_inside', 'failed_inside', 'days_outside', 'failed_outside'):
        add('Failure region (07)', col, 'days', float(f1[col].iloc[0]), float(f2[col].iloc[0]))
    g1, g2 = (tab(t, '07_global_sensitivity.csv').set_index('input') for t in (T1, T2))
    for inp in g1.index:
        add('Global sensitivity (07)', f'PRCC {inp}', '-', g1.loc[inp, 'prcc'], g2.loc[inp, 'prcc'])
    a1, a2 = (tab(t, '08_method_comparison_all.csv').set_index('method') for t in (T1, T2))
    for meth in a1.index:
        add('Search methods (08)', f'J - J_grid {meth}', 'mWh', a1.loc[meth, 'J_minus_grid_mWh'],
            a2.loc[meth, 'J_minus_grid_mWh'])
    p1, p2 = (tab(t, '12_pid_gain_check.csv').set_index('gain_scale') for t in (T1, T2))
    for g in p1.index:
        for col, unit in (('day_rms_error_deg', 'deg'), ('day_net_energy_Wh', 'Wh'), ('gain_margin_db', 'dB')):
            add(f'PID check (12), x{g:g}', col, unit, p1.loc[g, col], p2.loc[g, col])
    return pd.DataFrame(out)


def main():
    idc = identity_check()
    idc.round(9).to_csv(os.path.join(OUT, 'v1_vs_v2_identity_check.csv'), index=False)
    key = key_numbers()
    key.round(6).to_csv(os.path.join(OUT, 'v1_vs_v2_comparison.csv'), index=False)

    print('=== Identity check of every numeric column ===')
    for cat, grp in idc.groupby('category'):
        print(f'  {cat:30s} columns {len(grp):4d}, identical {int(grp.identical.sum()):4d}')
    plain = idc[(idc.category == 'non-energy') & ~idc.identical]
    print(f"  {'PASS' if plain.empty else 'FAIL'}: every non-energy column outside the decision tables is identical")
    if not plain.empty:
        print(plain.to_string(index=False))
    dec = idc[(idc.category == 'non-energy (decision table)') & ~idc.identical]
    if not dec.empty:
        print('  non-energy columns that changed in decision tables (the energy objective chose other points):')
        print(dec[['table', 'column', 'max_abs_diff']].to_string(index=False))
    print('\n=== Key numbers, v1 vs v2 ===')
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(key.round(4).to_string(index=False))
    print('\n=== Files saved ===')
    print('  results_v2/v1_vs_v2_comparison.csv')
    print('  results_v2/v1_vs_v2_identity_check.csv')


if __name__ == '__main__':
    main()
