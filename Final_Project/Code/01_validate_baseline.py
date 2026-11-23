# File: 01_validate_baseline.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Validate the ideal (disturbance-free) model: sun position, PV model, reference days, numerics
# Inputs: data/site_parameters.csv, data/pv_parameters.csv, data/drive_parameters.csv,
#         data/controller_parameters.csv, data/disturbance_ranges.csv
# Outputs: tables 01_spa_validation.csv, 01_pv_validation.csv (and 01_pv_validation_nmot.csv when the parameter
#          folder has pv_nmot_reference.csv, as data_v2/ does), 01_baseline_days.csv,
#          01_numerical_verification.csv; figures 01_sun_path_reference_days.png, 01_simple_vs_spa_error.png,
#          01_pv_iv_curves.png, 01_pv_pv_curves.png, 01_tracking_angles_equinox.png,
#          01_tracking_error_equinox.png, 01_power_tracking_vs_fixed.png, 01_event_vs_continuous.png
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from params import SEED, load_parameters, load_disturbance_ranges, project_paths, axis_params
from model_checks import check_parameters, check_levels
from solar_position import sun_position_spa, sun_position_simple, pointing_error_deg
from pv_model import fit_single_diode, iv_curve, pv_power, plane_of_array_irradiance, cell_temperature
from disturbances import dead_zone, wind_torque, hold_offset_rad
from drive_model import axis_derivatives, motor_time_constant_s, SETTLE_OMEGA_RAD_S
from day_simulation import (simulate_day, fixed_tilt_energy_Wh, ideal_levels, nominal_levels, _sun_and_sky,
                            TIMEZONE, WIND_PERIOD_CV, GRID_STEP_S)

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']
data_dir = paths['data']

# ---------------------------------------------------------------- parameters
site = load_parameters(os.path.join(data_dir, 'site_parameters.csv'))
pv = load_parameters(os.path.join(data_dir, 'pv_parameters.csv'))
drive = load_parameters(os.path.join(data_dir, 'drive_parameters.csv'))
ctrl = load_parameters(os.path.join(data_dir, 'controller_parameters.csv'))
ranges = load_disturbance_ranges(os.path.join(data_dir, 'disturbance_ranges.csv'))
check_parameters(site, pv, drive, ctrl)
sd = fit_single_diode(pv)
T_AIR_C = site['t_air_ref_c']
REFERENCE_DAYS = ('2026-03-20', '2026-06-21', '2026-12-21')
EQUINOX = '2026-03-20'
RTOL_LEVELS = (1e-4, 1e-6, 1e-8)
ATOL_PER_RTOL = 1e-3            # atol = 1e-3*rtol (rad-scale states)
HOUR_START_S = 11 * 3600.0      # 11:00 local
HOUR_LENGTH_S = 3600.0
STEP_VOLTAGE_V = ctrl['V_max']  # no-load step for the motor time constant check
IRRADIANCE_LEVELS = (200.0, 400.0, 600.0, 800.0, 1000.0)
T_STC_C = 25.0
NREL_CASE = {'time': '2003-10-17 12:30:30', 'tz': 'Etc/GMT+7', 'lat': 39.742476, 'lon': -105.1786,
             'alt': 1830.14, 'pressure_pa': 82000.0, 'temperature_c': 11.0, 'delta_t': 67.0,
             'azimuth_ref': 194.34024, 'zenith_ref': 50.11162}   # NREL/TP-560-34302, Table A4.1

print('=== Parameters ===')
for label, block in (('site', site), ('pv', pv), ('drive', drive), ('controller', ctrl)):
    for name, value in block.items():
        print(f'  {label:10s} {name:16s} = {value:g}')
print(f'  single-diode fit: ' + ', '.join(f'{k}={v:.4g}' for k, v in sd.items()))
print(f'  air temperature (reference days) = {T_AIR_C:g} degC, SEED = {SEED}')

saved_tables = []
saved_figures = []


def save_table(df_out, name):
    df_out.to_csv(os.path.join(tables_dir, name), index=False)
    saved_tables.append(name)


def save_figure(fig, name):
    fig.savefig(os.path.join(figures_dir, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    saved_figures.append(name)


def nice_upper(value, step):
    """Smallest multiple of step that is >= value (fixed axis limit that contains the data)."""
    return step * np.ceil(value / step)


# ---------------------------------------------------------------- local model functions
def motor_step_response(ax, volt, t_end):
    """No-load DC motor step: L di/dt = V - R i - Ke w, Jm dw/dt = Kt i - b w."""
    def rhs(t, y):
        i, omega = y
        return [(volt - ax['R'] * i - ax['Ke'] * omega) / ax['L'], (ax['Kt'] * i - ax['b'] * omega) / ax['Jm']]
    return solve_ivp(rhs, (0.0, t_end), [0.0, 0.0], method='LSODA', rtol=1e-10, atol=1e-12, dense_output=True)


def locked_axis_derivatives(t, y, theta_out, ax, T_wind, backlash_rad):
    """Worm locked (motor side fixed at theta_out = theta_m/N): flexible panel with backlash and wind."""
    theta_p, omega_p = y
    shaft = ax['ks'] * float(dead_zone(theta_out - theta_p, backlash_rad)) - ax['cs'] * omega_p
    return [omega_p, (shaft - T_wind) / ax['Jp']]


def continuous_axis(theta_m0, theta_p0, refs, starts, t_end, ax, ctrl, Tc, t_wind, backlash_rad, t_samples):
    """One continuous integration of an axis: at each update instant the controller runs while |e| >= tol
    (until |e| < tol and |omega_m/N| < 1e-4, or timeout_s); between moves the worm is locked and the
    flexible panel dynamics are still integrated. Returns the panel angle (rad) at t_samples."""
    tol = np.radians(ctrl['tol_deg'])
    n_gear = ax['N']
    state = np.array([0.0, theta_m0 * n_gear, 0.0, theta_p0, 0.0, 0.0])
    theta_p = np.full(len(t_samples), np.nan)
    for k, t0 in enumerate(starts):
        t1 = starts[k + 1] if k + 1 < len(starts) else t_end
        in_k = (t_samples >= t0) & (t_samples < t1)
        driven = np.zeros_like(in_k)
        t_now = t0
        if abs(refs[k] - state[3]) >= tol:
            state[0] = 0.0
            state[5] = 0.0

            def settled(t, x, ref, *unused):   # solve_ivp passes the RHS args to events as well
                return max(abs(ref - x[3]) / tol, abs(x[2] / n_gear) / SETTLE_OMEGA_RAD_S) - 1.0
            settled.terminal = True
            settled.direction = -1
            sol = solve_ivp(axis_derivatives, (t0, min(t0 + ctrl['timeout_s'], t1)), state, method='LSODA',
                            args=(refs[k], ax, ctrl, Tc, t_wind[k], backlash_rad), events=settled,
                            dense_output=True, rtol=1e-6, atol=1e-9)
            t_now = sol.t[-1]
            driven = in_k & (t_samples <= t_now)
            if np.any(driven):
                theta_p[driven] = sol.sol(t_samples[driven])[3]
            state = sol.y[:, -1].copy()
        locked = in_k & ~driven
        if t_now < t1:
            theta_out = state[1] / n_gear
            sol = solve_ivp(locked_axis_derivatives, (t_now, t1), state[3:5], method='LSODA',
                            args=(theta_out, ax, t_wind[k], backlash_rad), dense_output=True,
                            rtol=1e-6, atol=1e-9)
            if np.any(locked):
                theta_p[locked] = sol.sol(t_samples[locked])[0]
            state = np.array([0.0, state[1], 0.0, sol.y[0, -1], sol.y[1, -1], 0.0])
    return theta_p


def event_model_draws(secs, ctrl, levels, rng):
    """Re-create simulate_day's random draws in the same order: hold sides, then per-period wind."""
    sides = {'az': float(rng.choice([-1.0, 1.0])), 'el': float(rng.choice([-1.0, 1.0]))}
    period_starts = np.arange(secs[0], secs[-1] + 1e-9, ctrl['update_s'])
    wind = np.maximum(0.0, levels['v_mean_mps'] * (1.0 + WIND_PERIOD_CV * rng.standard_normal(len(period_starts))))
    return sides, period_starts, wind


# ---------------------------------------------------------------- 1) sun position
t0_clock = time.perf_counter()
nrel_times = pd.DatetimeIndex([NREL_CASE['time']], tz=NREL_CASE['tz'])
nrel_sp = sun_position_spa(nrel_times, NREL_CASE['lat'], NREL_CASE['lon'], NREL_CASE['alt'],
                           pressure_pa=NREL_CASE['pressure_pa'], temperature_c=NREL_CASE['temperature_c'],
                           delta_t=NREL_CASE['delta_t'])
nrel_az = float(nrel_sp['azimuth_deg'].iloc[0])
nrel_zen = 90.0 - float(nrel_sp['elevation_deg'].iloc[0])

year_times = pd.date_range('2026-01-01 00:00', '2026-12-31 23:00', freq='1h', tz=TIMEZONE)
year_spa = sun_position_spa(year_times, site['lat_deg'], site['lon_deg'], site['altitude_m'])
year_simple = sun_position_simple(year_times, site['lat_deg'], site['lon_deg'])
daylight = year_spa['elevation_deg'].to_numpy() > 0.0
simple_diff = pointing_error_deg(year_simple['azimuth_deg'].to_numpy()[daylight],
                                 year_simple['elevation_deg'].to_numpy()[daylight],
                                 year_spa['azimuth_deg'].to_numpy()[daylight],
                                 year_spa['elevation_deg'].to_numpy()[daylight])

df_spa = pd.DataFrame({
    'Case': ['NREL report example', 'NREL report example', 'Simplified vs SPA, 2026 daylight hours',
             'Simplified vs SPA, 2026 daylight hours', 'Simplified vs SPA, 2026 daylight hours'],
    'Quantity': ['azimuth_deg', 'zenith_deg', 'mean_angular_difference_deg', 'rms_angular_difference_deg',
                 'max_angular_difference_deg'],
    'Computed': [nrel_az, nrel_zen, np.mean(simple_diff), np.sqrt(np.mean(simple_diff ** 2)), np.max(simple_diff)],
    'Reference': [NREL_CASE['azimuth_ref'], NREL_CASE['zenith_ref'], np.nan, np.nan, np.nan],
})
df_spa['Difference_deg'] = df_spa['Computed'] - df_spa['Reference']
df_spa[['Computed', 'Reference', 'Difference_deg']] = df_spa[['Computed', 'Reference', 'Difference_deg']].round(5)
save_table(df_spa, '01_spa_validation.csv')

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
for date in REFERENCE_DAYS:
    day_times = pd.date_range(f'{date} 00:00', periods=24 * 60, freq='1min', tz=TIMEZONE)
    day_sp = sun_position_spa(day_times, site['lat_deg'], site['lon_deg'], site['altitude_m'])
    up = day_sp['elevation_deg'] > 0
    ax_plot.plot(day_sp['azimuth_deg'][up], day_sp['elevation_deg'][up], label=date)
ax_plot.set_xlim(0, 360)
ax_plot.set_ylim(0, 90)
ax_plot.set_xticks(np.arange(0, 361, 45))
ax_plot.set_xlabel('Sun azimuth (deg, clockwise from north)')
ax_plot.set_ylabel('Sun elevation (deg)')
ax_plot.set_title('Sun path at Madinah on the three reference days (NREL SPA)')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_sun_path_reference_days.png')

doy = year_times[daylight].dayofyear.to_numpy()
daily_max = pd.Series(simple_diff).groupby(doy).max()
daily_mean = pd.Series(simple_diff).groupby(doy).mean()
fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.plot(daily_max.index, daily_max.to_numpy(), label='daily maximum')
ax_plot.plot(daily_mean.index, daily_mean.to_numpy(), label='daily mean')
ax_plot.set_xlim(1, 365)
ax_plot.set_ylim(0, nice_upper(daily_max.max(), 0.1))
ax_plot.set_xlabel('Day of year 2026')
ax_plot.set_ylabel('Angular difference simplified - SPA (deg)')
ax_plot.set_title('Simplified solar equations vs NREL SPA (hourly, sun above horizon)')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_simple_vs_spa_error.png')

# ---------------------------------------------------------------- 2) PV model vs datasheet
v_stc, i_stc = iv_curve(1000.0, T_STC_C, pv, sd, n_points=20001)
p_stc = v_stc * i_stc
mpp = int(np.argmax(p_stc))
model_stc = {'P_max': float(pv_power(1000.0, T_STC_C, pv, sd)[0]), 'V_oc': float(v_stc[-1]),
             'I_sc': float(i_stc[0]), 'V_mp': float(v_stc[mpp]), 'I_mp': float(i_stc[mpp])}
units = {'P_max': 'W', 'V_oc': 'V', 'I_sc': 'A', 'V_mp': 'V', 'I_mp': 'A'}
df_pv = pd.DataFrame({
    'Quantity': list(model_stc),
    'Unit': [units[k] for k in model_stc],
    'Datasheet': [pv[k] for k in model_stc],
    'Model': [model_stc[k] for k in model_stc],
})
df_pv['Error_pct'] = 100.0 * (df_pv['Model'] - df_pv['Datasheet']) / df_pv['Datasheet']
df_pv[['Model', 'Error_pct']] = df_pv[['Model', 'Error_pct']].round(4)
save_table(df_pv, '01_pv_validation.csv')

# independent check at NMOT, only where the parameter folder has the datasheet NMOT column (v2): the fit uses
# STC values only, so this compares the model with data it has not seen
NMOT_CSV = os.path.join(data_dir, 'pv_nmot_reference.csv')
if os.path.exists(NMOT_CSV):
    nmot = load_parameters(NMOT_CSV)
    v_nm, i_nm = iv_curve(nmot['irradiance_Wm2'], pv['noct_c'], pv, sd, n_points=20001)
    k_nm = int(np.argmax(v_nm * i_nm))
    model_nm = {'P_max': float(pv_power(nmot['irradiance_Wm2'], pv['noct_c'], pv, sd)[0]), 'V_oc': float(v_nm[-1]),
                'I_sc': float(i_nm[0]), 'V_mp': float(v_nm[k_nm]), 'I_mp': float(i_nm[k_nm])}
    df_nm = pd.DataFrame({'Quantity': list(model_nm), 'Unit': [units[k] for k in model_nm],
                          'Datasheet_NMOT': [nmot[k] for k in model_nm], 'Model': [model_nm[k] for k in model_nm]})
    df_nm['Error_pct'] = 100.0 * (df_nm['Model'] - df_nm['Datasheet_NMOT']) / df_nm['Datasheet_NMOT']
    df_nm[['Model', 'Error_pct']] = df_nm[['Model', 'Error_pct']].round(4)
    save_table(df_nm, '01_pv_validation_nmot.csv')

for kind, name in (('iv', '01_pv_iv_curves.png'), ('pv', '01_pv_pv_curves.png')):
    fig, ax_plot = plt.subplots(figsize=(7, 4.5))
    for g_level in IRRADIANCE_LEVELS:
        v_curve, i_curve = iv_curve(g_level, T_STC_C, pv, sd, n_points=400)
        ax_plot.plot(v_curve, i_curve if kind == 'iv' else v_curve * i_curve, label=f'{g_level:.0f} W/m$^2$')
    ax_plot.set_xlim(0, nice_upper(pv['V_oc'], 5.0))
    if kind == 'iv':
        ax_plot.set_ylim(0, nice_upper(pv['I_sc'], 1.0))
        ax_plot.set_ylabel('Current (A)')
        ax_plot.set_title('Module I-V curves at 25 degC cell temperature (De Soto model)')
    else:
        ax_plot.set_ylim(0, nice_upper(pv['P_max'], 50.0))
        ax_plot.set_ylabel('Power (W)')
        ax_plot.set_title('Module P-V curves at 25 degC cell temperature (De Soto model)')
    ax_plot.set_xlabel('Voltage (V)')
    ax_plot.legend()
    ax_plot.grid(alpha=0.3)
    save_figure(fig, name)

# ---------------------------------------------------------------- 3) reference days, no disturbances
levels_ideal = ideal_levels()
check_levels(levels_ideal, ranges)
rows = []
day_results = {}
for date in REFERENCE_DAYS:
    t_day = time.perf_counter()
    res = simulate_day(date, site, pv, sd, drive, ctrl, levels_ideal, T_AIR_C,
                       rng=np.random.default_rng(SEED), keep_series=(date == EQUINOX))
    day_results[date] = res
    fixed = fixed_tilt_energy_Wh(date, site, pv, sd, T_AIR_C, site['lat_deg'])
    rows.append({'date': date, 'rms_error_deg': res['rms_error_deg'], 'max_error_deg': res['max_error_deg'],
                 'pv_energy_Wh': res['pv_energy_Wh'], 'fixed_tilt_energy_Wh': fixed,
                 'gain_pct': 100.0 * (res['pv_energy_Wh'] - fixed) / fixed,
                 'motor_energy_Wh': res['motor_energy_Wh'], 'net_energy_Wh': res['net_energy_Wh']})
    print(f'  {date}: simulated in {time.perf_counter() - t_day:.1f} s')
df_days = pd.DataFrame(rows).round(4)
save_table(df_days, '01_baseline_days.csv')

series = day_results[EQUINOX]['series']
hours = (series['time'] - series['time'].iloc[0].normalize()).dt.total_seconds().to_numpy() / 3600.0
x_lo, x_hi = np.floor(hours[0]), np.ceil(hours[-1])

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.plot(hours, series['sun_az'], label='sun azimuth')
ax_plot.plot(hours, series['panel_az'], '--', label='panel azimuth')
ax_plot.plot(hours, series['sun_el'], label='sun elevation')
ax_plot.plot(hours, series['panel_el'], '--', label='panel elevation')
ax_plot.set_xlim(x_lo, x_hi)
ax_plot.set_ylim(0, 300)
ax_plot.set_xlabel('Local time (h, UTC+3)')
ax_plot.set_ylabel('Angle (deg)')
ax_plot.set_title('Sun and panel angles, 2026-03-20, no disturbances')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_tracking_angles_equinox.png')

# Per-update-period RMS shown over the raw 10 s error (the raw saw-tooth repeats every update_s).
period_rms = series['error_deg'].pow(2).groupby(
    (np.arange(len(series)) * GRID_STEP_S) // ctrl['update_s']).transform('mean').pow(0.5)
fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.plot(hours, series['error_deg'], lw=0.3, color='0.7', label='pointing error (10 s samples)')
ax_plot.plot(hours, period_rms, color='C0', label=f"RMS per {ctrl['update_s']:g} s update period")
ax_plot.axhline(ctrl['tol_deg'], color='k', ls=':', label=f"controller tolerance {ctrl['tol_deg']:g} deg")
ax_plot.set_xlim(x_lo, x_hi)
ax_plot.set_ylim(0, nice_upper(series['error_deg'].max(), 0.1) + 0.1)
ax_plot.set_xlabel('Local time (h, UTC+3)')
ax_plot.set_ylabel('Pointing error (deg)')
ax_plot.set_title('Tracking error, 2026-03-20, no disturbances')
ax_plot.legend(loc='upper center', ncol=2, fontsize=8)
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_tracking_error_equinox.png')

eq_day = _sun_and_sky(EQUINOX, site['lat_deg'], site['lon_deg'], site['altitude_m'])
n_eq = len(eq_day[1])
poa_fixed = plane_of_array_irradiance(eq_day[5], eq_day[6], eq_day[4], eq_day[2], eq_day[3],   # dni, dhi, ghi
                                      np.full(n_eq, 180.0), np.full(n_eq, 90.0 - site['lat_deg']), site['albedo'])
power_fixed = pv_power(poa_fixed, cell_temperature(poa_fixed, T_AIR_C, pv['noct_c']), pv, sd)
fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.plot(hours, series['power_w'], label='dual-axis tracker')
ax_plot.plot(hours, power_fixed, label=f"fixed, tilt {site['lat_deg']:.1f} deg, facing south")
ax_plot.set_xlim(x_lo, x_hi)
ax_plot.set_ylim(0, nice_upper(series['power_w'].max(), 50.0))
ax_plot.set_xlabel('Local time (h, UTC+3)')
ax_plot.set_ylabel('PV power (W)')
ax_plot.set_title('Module power, tracking vs fixed tilt, 2026-03-20 (clear sky)')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_power_tracking_vs_fixed.png')

# ---------------------------------------------------------------- 4) numerical verification
levels_nom = nominal_levels(ranges)
check_levels(levels_nom, ranges)
verification = []
rtol_results = {}
for rtol in RTOL_LEVELS:
    res = simulate_day(EQUINOX, site, pv, sd, drive, ctrl, levels_nom, T_AIR_C,
                       rng=np.random.default_rng(SEED), rtol=rtol, atol=ATOL_PER_RTOL * rtol)
    rtol_results[rtol] = res
reference_res = rtol_results[RTOL_LEVELS[-1]]
for rtol, res in rtol_results.items():
    for quantity, unit in (('rms_error_deg', 'deg'), ('pv_energy_Wh', 'Wh')):
        change = 100.0 * (res[quantity] - reference_res[quantity]) / reference_res[quantity]
        verification.append({'Check': 'solver tolerance (nominal disturbances, 2026-03-20)',
                             'Case': f'rtol={rtol:g}', 'Quantity': quantity, 'Value': res[quantity], 'Unit': unit})
        verification.append({'Check': 'solver tolerance (nominal disturbances, 2026-03-20)',
                             'Case': f'rtol={rtol:g}', 'Quantity': f'{quantity}_change_vs_rtol_1e-8',
                             'Value': change, 'Unit': '%'})

# Event-based vs continuous over one hour (nominal disturbances, same random draws).
event_res = simulate_day(EQUINOX, site, pv, sd, drive, ctrl, levels_nom, T_AIR_C,
                         rng=np.random.default_rng(SEED), keep_series=True)
secs, sun_az, sun_el = eq_day[1], eq_day[2], eq_day[3]
sides, period_starts, wind = event_model_draws(secs, ctrl, levels_nom, np.random.default_rng(SEED))
k_first = int(np.searchsorted(period_starts, HOUR_START_S))
t_start = period_starts[k_first]
t_stop = t_start + HOUR_LENGTH_S
k_hour = np.arange(k_first, int(np.searchsorted(period_starts, t_stop)))
in_hour = (secs >= t_start) & (secs < t_stop)
backlash_rad = np.radians(levels_nom['backlash_deg'])
area = pv['length_m'] * pv['width_m']
ref_times = period_starts + ctrl['update_s'] / 2.0 - levels_nom['tau_d_s']
event_series = event_res['series']
panel_cont = {}
for name, sun_deg, panel_col in (('az', sun_az, 'panel_az'), ('el', sun_el, 'panel_el')):
    ax_axis = axis_params(drive, name)
    refs = np.interp(ref_times, secs, np.radians(sun_deg))
    offsets = hold_offset_rad(wind, backlash_rad, ax_axis['ks'], site['air_density'], area,
                              drive['Cm'], drive['Lc'], sides[name])
    torques = -sides[name] * wind_torque(wind, site['air_density'], area, drive['Cm'], drive['Lc'])
    # Initial state = event model's hold at the end of the period before the hour.
    idx_before = int(np.searchsorted(secs, t_start)) - 1
    theta_p0 = np.radians(event_series[panel_col].to_numpy()[idx_before])
    theta_m0 = theta_p0 - offsets[k_first - 1]
    panel_cont[name] = np.degrees(continuous_axis(theta_m0, theta_p0, refs[k_hour], period_starts[k_hour], t_stop,
                                                  ax_axis, ctrl, levels_nom['Tc_Nm'], torques[k_hour],
                                                  backlash_rad, secs[in_hour]))
error_event = event_series['error_deg'].to_numpy()[in_hour]
error_cont = pointing_error_deg(panel_cont['az'], panel_cont['el'], sun_az[in_hour], sun_el[in_hour])
rms_event = float(np.sqrt(np.mean(error_event ** 2)))
rms_cont = float(np.sqrt(np.mean(error_cont ** 2)))
hour_label = f'{t_start / 3600:.3f}-{t_stop / 3600:.3f} h, nominal disturbances'
for case, value in (('event-based', rms_event), ('continuous', rms_cont), ('difference', rms_cont - rms_event)):
    verification.append({'Check': 'event-based vs continuous (one hour, 2026-03-20)', 'Case': f'{case} ({hour_label})',
                         'Quantity': 'rms_error_deg', 'Value': value, 'Unit': 'deg'})

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
hour_axis = secs[in_hour] / 3600.0
ax_plot.plot(hour_axis, error_event, label=f'event-based (RMS {rms_event:.3f} deg)')
ax_plot.plot(hour_axis, error_cont, '--', label=f'continuous (RMS {rms_cont:.3f} deg)')
ax_plot.set_xlim(t_start / 3600.0, t_stop / 3600.0)
ax_plot.set_ylim(0, nice_upper(max(error_event.max(), error_cont.max()), 0.1))
ax_plot.set_xlabel('Local time (h, UTC+3)')
ax_plot.set_ylabel('Pointing error (deg)')
ax_plot.set_title('Event-based vs continuous simulation, one hour, 2026-03-20')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
save_figure(fig, '01_event_vs_continuous.png')

# Motor time constant: simulated 63 % of the no-load step response vs Jm*R/(Kt*Ke).
ax_el = axis_params(drive, 'el')
tau_formula = motor_time_constant_s(ax_el)
step = motor_step_response(ax_el, STEP_VOLTAGE_V, 20.0 * tau_formula)
t_fine = np.linspace(0.0, 20.0 * tau_formula, 200001)
omega_fine = step.sol(t_fine)[1]
tau_sim = float(t_fine[np.argmax(omega_fine >= (1.0 - np.exp(-1.0)) * omega_fine[-1])])
for case, value in (('simulated (63 % of no-load step)', tau_sim), ('formula Jm*R/(Kt*Ke)', tau_formula)):
    verification.append({'Check': 'motor time constant', 'Case': case, 'Quantity': 'tau_s', 'Value': value,
                         'Unit': 's'})
verification.append({'Check': 'motor time constant', 'Case': 'simulated vs formula', 'Quantity': 'difference',
                     'Value': 100.0 * (tau_sim - tau_formula) / tau_formula, 'Unit': '%'})
df_verify = pd.DataFrame(verification)
df_verify['Value'] = df_verify['Value'].round(6)
save_table(df_verify, '01_numerical_verification.csv')

# ---------------------------------------------------------------- summary
print('\n=== Acceptance checks ===')
print(f"  NREL azimuth diff = {nrel_az - NREL_CASE['azimuth_ref']:+.5f} deg, "
      f"zenith diff = {nrel_zen - NREL_CASE['zenith_ref']:+.5f} deg (limit 0.01)")
print('  PV errors (%): ' + ', '.join(f'{q}={e:+.3f}' for q, e in zip(df_pv['Quantity'], df_pv['Error_pct'])) +
      ' (limit 1)')
for _, row in df_days.iterrows():
    print(f"  {row['date']}: rms={row['rms_error_deg']:.4f} deg (limit 0.5), gain={row['gain_pct']:.2f} % "
          f"(range 15-60)")
for quantity in ('rms_error_deg', 'pv_energy_Wh'):
    change = 100.0 * (rtol_results[1e-6][quantity] - reference_res[quantity]) / reference_res[quantity]
    print(f'  rtol 1e-6 vs 1e-8: {quantity} change = {change:+.6f} % (limit 0.1)')
print(f'  event vs continuous RMS: {rms_event:.5f} vs {rms_cont:.5f} deg, '
      f'difference {rms_cont - rms_event:+.5f} deg (limit 0.02)')
print(f'  motor time constant: simulated {tau_sim * 1e3:.3f} ms vs formula {tau_formula * 1e3:.3f} ms')
print(f'  simplified vs SPA: mean {np.mean(simple_diff):.3f}, max {np.max(simple_diff):.3f} deg')
print(f'  total run time {time.perf_counter() - t0_clock:.1f} s')

print('\n=== Files saved ===')
for name in saved_tables:
    print(f'  table : results/tables/{name}')
for name in saved_figures:
    print(f'  figure: results/figures/{name}')
