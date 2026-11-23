# File: 02_fit_wind_distribution.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Fit candidate distributions to Madinah daytime hourly wind speed (NASA POWER 2021-2023),
#          test goodness of fit and select the distribution used by the Monte Carlo study
# Inputs: data/site_parameters.csv, data/raw/nasa_power_madinah_2021_2023.csv (downloaded once),
#         data/disturbance_ranges.csv
# Outputs: tables 02_wind_fit_comparison.csv, 02_selected_wind_distribution.csv;
#          figures 02_wind_histogram_fits.png, 02_wind_cdf.png, 02_wind_qq.png;
#          updated Nominal of v_mean_mps in data/disturbance_ranges.csv
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from params import SEED, load_parameters, project_paths
from weather_data import (download_power_hourly, load_power_hourly, daytime_wind, save_selected_distribution,
                          DAY_START_HOUR, DAY_END_HOUR)

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# ---------------------------------------------------------------- parameters
site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
RAW_CSV = os.path.join(paths['raw'], 'nasa_power_madinah_2021_2023.csv')
RANGES_CSV = os.path.join(paths['data'], 'disturbance_ranges.csv')
START_YEAR, END_YEAR = 2021, 2023
CANDIDATES = ('weibull_min', 'lognorm', 'gamma')
LABELS = {'weibull_min': 'Weibull', 'lognorm': 'Lognormal', 'gamma': 'Gamma'}
N_SUBSAMPLE = 500
N_FREE_PARAMS = 2           # shape and scale (loc fixed at 0)
MIN_EXPECTED_PER_BIN = 5

print('=== Parameters ===')
print(f"  site lat/lon = {site['lat_deg']:g}, {site['lon_deg']:g} deg; years {START_YEAR}-{END_YEAR}")
print(f'  daytime window = {DAY_START_HOUR:02d}:00-{DAY_END_HOUR:02d}:00 LST; candidates = {CANDIDATES}')
print(f'  subsample n = {N_SUBSAMPLE}, SEED = {SEED}')

saved_tables = []
saved_figures = []


# ---------------------------------------------------------------- functions
def n_equiprobable_bins(n):
    """Number of equal-probability bins: 2*n^0.4 (Mann-Wald rule), capped so that n/k >= 5."""
    return int(min(np.floor(2.0 * n ** 0.4), np.floor(n / MIN_EXPECTED_PER_BIN)))


def chi_square_test(sample, frozen, n_params):
    """Chi-square goodness of fit with equal-probability bins under the fitted distribution."""
    k = n_equiprobable_bins(len(sample))
    edges = frozen.ppf(np.linspace(0.0, 1.0, k + 1))
    observed = np.histogram(sample, bins=edges)[0]
    expected = np.full(k, len(sample) / k)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    dof = k - 1 - n_params
    return chi2, float(stats.chi2.sf(chi2, dof)), k, dof


def fit_candidate(name, sample):
    family = getattr(stats, name)
    shape, loc, scale = family.fit(sample, floc=0.0)
    return family(shape, loc=loc, scale=scale), (shape, loc, scale)


# ---------------------------------------------------------------- data
download_power_hourly(site['lat_deg'], site['lon_deg'], START_YEAR, END_YEAR, RAW_CSV)
df_weather = load_power_hourly(RAW_CSV)
wind = daytime_wind(df_weather)
if len(wind) == 0 or np.any(wind <= 0.0):
    raise ValueError(f'daytime wind sample must be non-empty and strictly positive, n={len(wind)}')
rng = np.random.default_rng(SEED)
wind_sub = rng.choice(wind, size=N_SUBSAMPLE, replace=False)
print(f'  daytime hourly wind: n = {len(wind)}, mean = {wind.mean():.3f} m/s, sd = {wind.std(ddof=1):.3f} m/s')

# ---------------------------------------------------------------- fits and tests
fits = {}
rows = []
for name in CANDIDATES:
    frozen, (shape, loc, scale) = fit_candidate(name, wind)
    fits[name] = frozen
    row = {'distribution': name, 'shape': shape, 'loc': loc, 'scale': scale,
           'mean_data_mps': wind.mean(), 'mean_fit_mps': frozen.mean(),
           'sd_data_mps': wind.std(ddof=1), 'sd_fit_mps': frozen.std()}
    for label, sample in (('full', wind), ('n500', wind_sub)):
        chi2, p_chi2, k, dof = chi_square_test(sample, frozen, N_FREE_PARAMS)
        ks, p_ks = stats.kstest(sample, frozen.cdf)
        row.update({f'bins_{label}': k, f'dof_{label}': dof, f'chi2_{label}': chi2, f'p_chi2_{label}': p_chi2,
                    f'ks_{label}': float(ks), f'p_ks_{label}': float(p_ks)})
    rows.append(row)
df_fit = pd.DataFrame(rows)
df_fit_out = df_fit.copy()
num_cols = df_fit_out.select_dtypes('number').columns
df_fit_out[num_cols] = df_fit_out[num_cols].apply(lambda col: col.map(lambda x: float(f'{x:.4g}')))
df_fit_out.to_csv(os.path.join(tables_dir, '02_wind_fit_comparison.csv'), index=False)
saved_tables.append('02_wind_fit_comparison.csv')

# ---------------------------------------------------------------- selection
# Selected: Weibull. All three fits respect the physical support v >= 0 (loc fixed at 0), but the
# Weibull follows the histogram body and the Q-Q line over the whole range, reproduces the data mean
# and standard deviation, and has the lowest chi-square / KS statistics on both samples; the lognormal
# and gamma tails are too heavy for the bounded upper wind speeds seen at the site. With n = 14k every
# test rejects small deviations, so the decision rests on the plots and the n = 500 subsample.
# Weibull is also the standard physical model for wind speed in wind-energy practice.
SELECTED_DISTRIBUTION = 'weibull_min'
shape_sel, loc_sel, scale_sel = (float(df_fit.loc[df_fit['distribution'] == SELECTED_DISTRIBUTION, c].iloc[0])
                                 for c in ('shape', 'loc', 'scale'))
save_selected_distribution(os.path.join(tables_dir, '02_selected_wind_distribution.csv'),
                           SELECTED_DISTRIBUTION, round(shape_sel, 4), round(loc_sel, 4), round(scale_sel, 4))
saved_tables.append('02_selected_wind_distribution.csv')

df_ranges = pd.read_csv(RANGES_CSV)
v_nominal = round(float(wind.mean()), 1)
df_ranges.loc[df_ranges['Parameter'] == 'v_mean_mps', 'Nominal'] = v_nominal
df_ranges.loc[df_ranges['Parameter'] == 'v_mean_mps', 'Source'] = \
    f'NASA POWER WS10M {START_YEAR}-{END_YEAR}, daytime mean (script 02)'
df_ranges.to_csv(RANGES_CSV, index=False)

# ---------------------------------------------------------------- figures
v_max_plot = float(np.ceil(wind.max()))
v_grid = np.linspace(0.0, v_max_plot, 400)

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.hist(wind, bins=np.arange(0.0, v_max_plot + 0.5, 0.5), density=True, color='0.8', edgecolor='0.6',
             label=f'NASA POWER daytime hourly (n = {len(wind)})')
for name in CANDIDATES:
    ax_plot.plot(v_grid, fits[name].pdf(v_grid), label=LABELS[name])
pdf_max = max(float(np.max(fits[name].pdf(v_grid[1:]))) for name in CANDIDATES)
hist_max = float(np.max(np.histogram(wind, bins=np.arange(0.0, v_max_plot + 0.5, 0.5), density=True)[0]))
ax_plot.set_xlim(0, v_max_plot)
ax_plot.set_ylim(0, np.ceil(max(pdf_max, hist_max) * 1.1 / 0.05) * 0.05)
ax_plot.set_xlabel('Wind speed at 10 m (m/s)')
ax_plot.set_ylabel('Probability density (s/m)')
ax_plot.set_title(f'Madinah daytime wind speed {START_YEAR}-{END_YEAR} with fitted distributions')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
fig.savefig(os.path.join(figures_dir, '02_wind_histogram_fits.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('02_wind_histogram_fits.png')

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
v_sorted = np.sort(wind)
ax_plot.step(v_sorted, np.arange(1, len(v_sorted) + 1) / len(v_sorted), where='post', color='k',
             label='empirical CDF')
for name in CANDIDATES:
    ax_plot.plot(v_grid, fits[name].cdf(v_grid), '--', label=LABELS[name])
ax_plot.set_xlim(0, v_max_plot)
ax_plot.set_ylim(0, 1)
ax_plot.set_xlabel('Wind speed at 10 m (m/s)')
ax_plot.set_ylabel('Cumulative probability (-)')
ax_plot.set_title('Empirical vs fitted cumulative distribution functions')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
fig.savefig(os.path.join(figures_dir, '02_wind_cdf.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('02_wind_cdf.png')

probs = (np.arange(1, len(v_sorted) + 1) - 0.5) / len(v_sorted)
q_max = float(np.ceil(max(v_sorted[-1], max(fits[name].ppf(probs[-1]) for name in CANDIDATES))))
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
for idx, name in enumerate(CANDIDATES):
    axs[idx].plot(fits[name].ppf(probs), v_sorted, '.', ms=2, label='data quantiles')
    axs[idx].plot([0, q_max], [0, q_max], 'k--', lw=1, label='y = x')
    axs[idx].set_xlim(0, q_max)
    axs[idx].set_ylim(0, q_max)
    axs[idx].set_aspect('equal', adjustable='box')
    axs[idx].set_xlabel(f'{LABELS[name]} quantile (m/s)')
    axs[idx].set_ylabel('Data quantile (m/s)')
    axs[idx].set_title(f'Q-Q plot: {LABELS[name]}')
    axs[idx].legend(loc='upper left')
    axs[idx].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, '02_wind_qq.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('02_wind_qq.png')

# ---------------------------------------------------------------- summary
print('\n=== Goodness of fit ===')
for _, row in df_fit.iterrows():
    print(f"  {row['distribution']:12s} shape={row['shape']:.4f} scale={row['scale']:.4f} "
          f"mean {row['mean_fit_mps']:.3f} (data {row['mean_data_mps']:.3f}) sd {row['sd_fit_mps']:.3f} "
          f"(data {row['sd_data_mps']:.3f})")
    print(f"      full: chi2={row['chi2_full']:.1f} (k={row['bins_full']}) p={row['p_chi2_full']:.3g} "
          f"KS={row['ks_full']:.4f} p={row['p_ks_full']:.3g} | n500: chi2={row['chi2_n500']:.1f} "
          f"(k={row['bins_n500']}) p={row['p_chi2_n500']:.3g} KS={row['ks_n500']:.4f} p={row['p_ks_n500']:.3g}")
print(f'  selected: {SELECTED_DISTRIBUTION}; v_mean_mps Nominal set to {v_nominal} m/s')

print('\n=== Files saved ===')
for name in saved_tables:
    print(f'  table : results/tables/{name}')
for name in saved_figures:
    print(f'  figure: results/figures/{name}')
print('  data  : data/disturbance_ranges.csv (v_mean_mps Nominal updated)')
