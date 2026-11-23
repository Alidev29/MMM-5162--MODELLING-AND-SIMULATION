# File: 11_export_series.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Save, as data files, the series that the figures of studies 01, 03, 05 and 06 plot but their
#          tables do not hold, so every figure can be checked and redrawn: the equinox day at 10 s, sun paths,
#          PV curves, the simplified-vs-SPA series, the event-vs-continuous hour, the gust and free-decay
#          series, the 200 robustness days of study 06 and the full-day series of 58 Monte Carlo days.
#          Studies 01 and 03 are executed unchanged (runpy) and their plotted arrays are read from their
#          namespace; 05 and 06 days are re-simulated with their own functions and seeds [SEED, k].
# Inputs: code/01_validate_baseline.py, code/03_gust_vibration_study.py, 05_monte_carlo_risk.py,
#         06_optimization.py and everything they read; results/tables/05_mc_trials.csv, 06_robustness.csv
# Outputs: results/tables/series/*.csv and 05_day_series.npz with 05_day_series_index.csv
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import sys
import time
import runpy
import importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from params import SEED, project_paths
from day_simulation import simulate_day
from model_checks import check_levels

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
series_dir = os.path.join(tables_dir, 'series')

_spec = importlib.util.spec_from_file_location('opt06', os.path.join(script_dir, '06_optimization.py'))
opt06 = importlib.util.module_from_spec(_spec)
sys.modules['opt06'] = opt06            # worker processes unpickle opt06.run_scenario
_spec.loader.exec_module(opt06)
mc05 = opt06.mc05

# ---------------------------------------------------------------- parameters
N_DAYS_PER_CLASS = 29                    # all 29 failed baseline days and as many passing days
SELECT_STREAM = 11                       # passing days drawn with default_rng([SEED, 11])
FFT_MAX_HZ = 10.0                        # spectrum range shown in the 03 figure
FREE_DECAY_SHOWN_S = 10.0                # free-decay range shown in the 03 figure
REPRO_TOL = 1e-6


def save_csv(df, name, saved, digits=8):
    df.round(digits).to_csv(os.path.join(series_dir, name), index=False)
    saved.append(name)


def mc_day_with_series(k):
    """Baseline trial k of study 05 re-simulated with keep_series=True (same seed [SEED, k])."""
    c = opt06._context()
    dists = mc05.input_distributions(c['ranges'], c['wind'])
    levels, rng = mc05.sample_levels(k, dists, c['ranges'], 1.0)
    check_levels(levels, c['ranges'])
    res = simulate_day(mc05.DATE, c['site'], c['pv'], c['sd'], c['drive'], c['ctrl'], levels, c['t_air'],
                       vibration_curve=c['vib'], rng=rng, keep_series=True)
    return k, levels, res


def export_study01(saved):
    ns = runpy.run_path(os.path.join(script_dir, '01_validate_baseline.py'), run_name='study01')
    series, hours = ns['series'], ns['hours']
    day = pd.DataFrame({'local_time_h': hours, 'sun_az_deg': series['sun_az'], 'sun_el_deg': series['sun_el'],
                        'panel_az_deg': series['panel_az'], 'panel_el_deg': series['panel_el'],
                        'pointing_error_deg': series['error_deg'], 'tracker_power_W': series['power_w'],
                        'fixed_power_W': ns['power_fixed']})
    save_csv(day, '01_equinox_day_10s.csv', saved)
    rows = []
    for date in ns['REFERENCE_DAYS']:
        times = pd.date_range(f'{date} 00:00', periods=24 * 60, freq='1min', tz=ns['TIMEZONE'])
        sp = ns['sun_position_spa'](times, ns['site']['lat_deg'], ns['site']['lon_deg'], ns['site']['altitude_m'])
        up = sp['elevation_deg'] > 0
        rows.append(pd.DataFrame({'date': date, 'local_time': times[up].strftime('%H:%M'),
                                  'sun_az_deg': sp['azimuth_deg'][up].to_numpy(),
                                  'sun_el_deg': sp['elevation_deg'][up].to_numpy()}))
    save_csv(pd.concat(rows), '01_sun_path_reference_days.csv', saved)
    curves = []
    for g in ns['IRRADIANCE_LEVELS']:
        v, i = ns['iv_curve'](g, ns['T_STC_C'], ns['pv'], ns['sd'], n_points=400)
        curves.append(pd.DataFrame({'irradiance_Wm2': g, 'voltage_V': v, 'current_A': i, 'power_W': v * i}))
    save_csv(pd.concat(curves), '01_pv_iv_pv_curves.csv', saved)
    year_times, daylight = ns['year_times'], ns['daylight']
    save_csv(pd.DataFrame({'time': year_times[daylight].strftime('%Y-%m-%d %H:%M'),
                           'angular_difference_deg': ns['simple_diff']}), '01_simple_vs_spa_hourly.csv', saved)
    save_csv(pd.DataFrame({'day_of_year': ns['daily_max'].index, 'daily_max_deg': ns['daily_max'].to_numpy(),
                           'daily_mean_deg': ns['daily_mean'].to_numpy()}), '01_simple_vs_spa_daily.csv', saved)
    save_csv(pd.DataFrame({'local_time_h': ns['hour_axis'], 'error_event_deg': ns['error_event'],
                           'error_continuous_deg': ns['error_cont']}), '01_event_vs_continuous_hour.csv', saved)


def export_study03(saved):
    ns = runpy.run_path(os.path.join(script_dir, '03_gust_vibration_study.py'), run_name='study03')
    t, v, phi_el = ns['examples']['el']
    _, _, phi_az = ns['examples']['az']
    save_csv(pd.DataFrame({'t_s': t, 'wind_mps': v, 'deflection_az_deg': np.degrees(phi_az),
                           'deflection_el_deg': np.degrees(phi_el)}), '03_gust_example_10mps.csv', saved)
    t_free, phi_free, freqs, spectrum, f_sim, f_formula = ns['decay_example']
    shown = t_free <= FREE_DECAY_SHOWN_S
    save_csv(pd.DataFrame({'t_s': t_free[shown], 'deflection_el_deg': np.degrees(phi_free[shown])}),
             '03_free_decay_el.csv', saved, digits=10)
    band = freqs <= FFT_MAX_HZ
    save_csv(pd.DataFrame({'frequency_Hz': freqs[band], 'normalised_magnitude': spectrum[band] / spectrum.max()}),
             '03_free_decay_spectrum_el.csv', saved, digits=10)


def main():
    t_start = time.perf_counter()
    os.makedirs(series_dir, exist_ok=True)
    saved = []
    print('=== Parameters ===')
    print(f'  SEED {SEED}; MC days: all failed baseline days + {N_DAYS_PER_CLASS} passing days '
          f'(rng [{SEED}, {SELECT_STREAM}]); robustness: {opt06.N_ROBUST} days x 2 settings')

    export_study01(saved)
    print(f'  study 01 series saved, {time.perf_counter() - t_start:.0f} s', flush=True)
    export_study03(saved)
    print(f'  study 03 series saved, {time.perf_counter() - t_start:.0f} s', flush=True)

    trials = pd.read_csv(os.path.join(tables_dir, '05_mc_trials.csv'))
    base = trials[trials['set'] == 'baseline'].set_index('trial')
    failed = base.index[base['rms_error_deg'] > mc05.FAIL_LIMIT_DEG].to_numpy()
    passing = base.index[base['rms_error_deg'] <= mc05.FAIL_LIMIT_DEG].to_numpy()
    chosen_pass = np.sort(np.random.default_rng([SEED, SELECT_STREAM]).choice(passing, N_DAYS_PER_CLASS,
                                                                             replace=False))
    days = np.concatenate([failed, chosen_pass])

    rob06 = pd.read_csv(os.path.join(tables_dir, '06_robustness.csv'))
    with ProcessPoolExecutor(max_workers=opt06.N_WORKERS) as pool:
        # ---------------------------------------------------- 06: per-day robustness values
        rob_rows = []
        for _, r in rob06.iterrows():
            outs = np.array(list(pool.map(opt06.run_scenario, [(r['update_s'], r['tol_deg'], k)
                                                              for k in range(opt06.N_ROBUST)], chunksize=2)))
            rob_rows.append(pd.DataFrame({'setting': r['setting'], 'update_s': r['update_s'], 'tol_deg': r['tol_deg'],
                                          'trial': np.arange(opt06.N_ROBUST), 'net_energy_Wh': outs[:, 0],
                                          'rms_error_deg': outs[:, 1]}))
        df_rob = pd.concat(rob_rows)
        save_csv(df_rob, '06_robustness_days.csv', saved)
        print(f'  study 06 robustness days saved, {time.perf_counter() - t_start:.0f} s', flush=True)

        # ---------------------------------------------------- 05: full-day series of 58 MC days
        results = list(pool.map(mc_day_with_series, days.tolist(), chunksize=1))
    arrays, index_rows = {}, []
    for k, levels, res in results:
        s = res['series']
        if not arrays:
            arrays['local_time_h'] = ((s['time'] - s['time'].iloc[0].normalize()).dt.total_seconds() / 3600.0).to_numpy()
        for col in ('sun_az', 'sun_el', 'panel_az', 'panel_el', 'error_deg', 'power_w'):
            arrays[f'k{k:04d}_{col}'] = s[col].to_numpy()
        saved_rms = float(base.loc[k, 'rms_error_deg'])
        index_rows.append({'trial': k, 'class': 'fail' if saved_rms > mc05.FAIL_LIMIT_DEG else 'pass', **levels,
                           'rms_error_deg_rerun': res['rms_error_deg'], 'rms_error_deg_05': saved_rms,
                           'abs_difference_deg': abs(res['rms_error_deg'] - saved_rms)})
    np.savez_compressed(os.path.join(series_dir, '05_day_series.npz'), **arrays)
    saved.append('05_day_series.npz')
    df_index = pd.DataFrame(index_rows)
    save_csv(df_index, '05_day_series_index.csv', saved, digits=10)

    # ------------------------------------------------------------ reproduction checks
    summary_ok = []
    for _, r in rob06.iterrows():
        sub = df_rob[df_rob['setting'] == r['setting']]
        n_fail = int(np.sum(sub['rms_error_deg'] > opt06.RMS_LIMIT_DEG))
        p_hat = n_fail / len(sub)
        net_m, rms_m = sub['net_energy_Wh'].mean(), sub['rms_error_deg'].mean()
        # 06 stored its settings rounded to 6 decimals; the baseline (120 s, 0.1 deg) is exact, the PSO optimum
        # (e.g. 149.977323 s) is re-simulated at the rounded value, so its means may move in the 6th decimal.
        tol = 1e-6 if r['setting'] == 'baseline' else 1e-4
        ok = (abs(p_hat - r['p_fail']) < 1e-9 and abs(net_m - r['mean_net_energy_Wh']) <= tol
              and abs(rms_m - r['mean_rms_error_deg']) <= tol)
        summary_ok.append((r['setting'], ok, tol))
        print(f"  06 {r['setting']}: p_fail {p_hat:.4f} vs {r['p_fail']:.4f}, net {net_m:.6f} vs "
              f"{r['mean_net_energy_Wh']:.6f}, rms {rms_m:.6f} vs {r['mean_rms_error_deg']:.6f}")
    max_diff = float(df_index['abs_difference_deg'].max())
    print('\n=== Checks ===')
    for setting, ok, tol in summary_ok:
        print(f"  {'PASS' if ok else 'FAIL'}: per-day robustness values reproduce 06_robustness.csv for the {setting} "
              f"(identical p_fail, means within {tol:g}{'; setting stored rounded to 6 decimals' if tol > 1e-6 else ''})")
    print(f"  {'PASS' if max_diff <= REPRO_TOL else 'FAIL'}: {len(df_index)} re-simulated MC days match "
          f"05_mc_trials.csv (largest RMS difference {max_diff:.2e} deg, limit {REPRO_TOL:g})")
    print(f"  {'PASS' if (df_index['class'] == 'fail').sum() == len(failed) else 'FAIL'}: all {len(failed)} failed "
          f"baseline days included")
    print(f'  total run time {time.perf_counter() - t_start:.0f} s')
    print('\n=== Files saved ===')
    for name in saved:
        size = os.path.getsize(os.path.join(series_dir, name))
        print(f'  results/tables/series/{name}  ({size / 1024:.0f} kB)')


if __name__ == '__main__':
    main()
