# File: day_simulation.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Event-based simulation of one tracking day (sunrise to sunset): full axis dynamics during
#          each move, algebraic hold model between moves, PV power on a 10 s grid, daily metrics.
# Inputs: date, site/pv/drive/controller parameter dicts, single-diode parameters, disturbance levels,
#         daytime air temperature, optional vibration curve, numpy Generator
# Outputs: dict of daily metrics (rms/max pointing error, PV/motor/net/ideal energy, loss %), optional series
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import functools
import numpy as np
import pandas as pd
from params import SEED, axis_params
from solar_position import sun_position_spa, pointing_error_deg
from pv_model import clear_sky_irradiance, plane_of_array_irradiance, cell_temperature, pv_power
from disturbances import wind_torque, hold_offset_rad
from drive_model import simulate_move
from model_checks import check_pv_output

GRID_STEP_S = 10.0
MIN_ELEVATION_DEG = 5.0
TIMEZONE = 'Asia/Riyadh'
WIND_PERIOD_CV = 0.2       # per-period wind variability: v = v_mean*(1 + 0.2*z)
LEVEL_KEYS = ('Tc_Nm', 'v_mean_mps', 'backlash_deg', 'tau_d_s')


def ideal_levels():
    """All disturbances off."""
    return {key: 0.0 for key in LEVEL_KEYS}


def nominal_levels(ranges):
    """Nominal disturbance levels from disturbance_ranges.csv."""
    return {key: float(ranges.loc[key, 'Nominal']) for key in LEVEL_KEYS}


def load_vibration_curve(csv_path):
    """Linear interpolation v_mean_mps -> rms_vibration_deg, clamped at both ends."""
    df_curve = pd.read_csv(csv_path).sort_values('v_mean_mps')
    v_points = df_curve['v_mean_mps'].to_numpy(dtype=float)
    rms_points = df_curve['rms_vibration_deg'].to_numpy(dtype=float)
    return lambda v: float(np.interp(v, v_points, rms_points))


@functools.lru_cache(maxsize=16)
def _sun_and_sky(date, lat_deg, lon_deg, altitude_m):
    """10 s grid of the sunlit part of the day (elevation > 5 deg): times, seconds, sun angles, clear sky."""
    times = pd.date_range(f'{date} 00:00', periods=int(86400 / GRID_STEP_S), freq=f'{int(GRID_STEP_S)}s',
                          tz=TIMEZONE)
    sun = sun_position_spa(times, lat_deg, lon_deg, altitude_m)
    up = sun['elevation_deg'].to_numpy() > MIN_ELEVATION_DEG
    sky = clear_sky_irradiance(times[up], lat_deg, lon_deg, altitude_m)
    secs = (times[up] - times[0]).total_seconds().to_numpy()
    az = np.degrees(np.unwrap(np.radians(sun['azimuth_deg'].to_numpy()[up])))
    arrays = (secs, az, sun['elevation_deg'].to_numpy()[up],
              sky['ghi'].to_numpy(), sky['dni'].to_numpy(), sky['dhi'].to_numpy())
    for arr in arrays:
        arr.setflags(write=False)
    return (times[up],) + arrays


def _energy_Wh(secs, power_w):
    return float(np.trapezoid(power_w, secs) / 3600.0)


def _power(day, panel_az, panel_el, site, pv, sd, t_air_c):
    _, secs, sun_az, sun_el, ghi, dni, dhi = day
    poa = plane_of_array_irradiance(dni, dhi, ghi, sun_az, sun_el, panel_az, panel_el, site['albedo'])
    t_cell = cell_temperature(poa, t_air_c, pv['noct_c'])
    return poa, pv_power(poa, t_cell, pv, sd)


def _freeze(d):
    return tuple(sorted(d.items()))


@functools.lru_cache(maxsize=64)
def _ideal_energy_cached(date, site_key, pv_key, sd_key, t_air_c):
    site, pv, sd = dict(site_key), dict(pv_key), dict(sd_key)
    day = _sun_and_sky(date, site['lat_deg'], site['lon_deg'], site['altitude_m'])
    _, power = _power(day, day[2], day[3], site, pv, sd, t_air_c)
    return _energy_Wh(day[1], power)


def simulate_day(date, site, pv, sd, drive, ctrl, levels, t_air_c, vibration_curve=None, rng=None,
                 keep_series=False, rtol=1e-6, atol=1e-9):
    """Simulate one tracking day and return the daily performance metrics (rtol/atol: move solver)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    day = _sun_and_sky(date, site['lat_deg'], site['lon_deg'], site['altitude_m'])
    times, secs, sun_az, sun_el = day[0], day[1], day[2], day[3]
    area = pv['length_m'] * pv['width_m']
    backlash_rad = np.radians(levels['backlash_deg'])
    axes = {'az': axis_params(drive, 'az'), 'el': axis_params(drive, 'el')}
    sun_track = {'az': np.radians(sun_az), 'el': np.radians(sun_el)}

    # Random draws: hold side per axis (once per day), then one wind factor per update period.
    sides = {'az': float(rng.choice([-1.0, 1.0])), 'el': float(rng.choice([-1.0, 1.0]))}
    update_s = ctrl['update_s']
    period_starts = np.arange(secs[0], secs[-1] + 1e-9, update_s)
    n_periods = len(period_starts)
    wind = np.maximum(0.0, levels['v_mean_mps'] * (1.0 + WIND_PERIOD_CV * rng.standard_normal(n_periods)))

    # Reference: sun at the middle of the coming period, delayed by the timing error tau_d.
    ref_times = period_starts + update_s / 2.0 - levels['tau_d_s']
    refs = {name: np.interp(ref_times, secs, sun_track[name]) for name in axes}
    theta = {name: refs[name][0] for name in axes}   # parked on the first reference before sunrise
    panel_now = dict(theta)                           # true held panel angle (rad)
    period_index = np.minimum(((secs - secs[0]) // update_s).astype(int), n_periods - 1)

    motor_energy_J = 0.0
    n_moves = 0
    n_unsettled = 0
    hold = {name: np.empty(n_periods) for name in axes}
    for k in range(n_periods):
        torque = wind_torque(wind[k], site['air_density'], area, drive['Cm'], drive['Lc'])
        for name, ax in axes.items():
            # Static flank offset under this period's wind: a non-zero wind torque presses the panel onto
            # the side flank (theta_m/N + offset); with zero torque nothing moves it, so it stays where it
            # was (user decision 2026-11-18). The move starts from this true panel angle (PID measures it).
            pressed = torque > 0.0
            offset = hold_offset_rad(wind[k], backlash_rad, ax['ks'], site['air_density'],
                                     area, drive['Cm'], drive['Lc'], sides[name])
            if pressed:
                panel_now[name] = theta[name] + offset
            # Wind torque sign chosen so that the static twist is +side*T/ks, as in hold_offset_rad.
            move = simulate_move(theta[name], refs[name][k], ax, ctrl, levels['Tc_Nm'],
                                 -sides[name] * torque, backlash_rad, rtol=rtol, atol=atol,
                                 theta_p0_rad=panel_now[name])
            theta[name] = move['theta_end_rad']
            motor_energy_J += move['energy_J']
            n_moves += int(move['duration_s'] > 0.0)
            n_unsettled += int(not move['settled'])
            # Unpressed panel rests where the move left it, but no farther than a flank (the spring
            # engages beyond +/-b): theta_m/N + clip(theta_p - theta_m/N, -b, b); b = 0 gives theta_m/N.
            free = float(np.clip(move['x'][3, -1] - theta[name], -backlash_rad, backlash_rad))
            panel_now[name] = theta[name] + offset if pressed else theta[name] + free
            hold[name][k] = panel_now[name]
    panel = {name: np.degrees(hold[name][period_index]) for name in axes}

    error = pointing_error_deg(panel['az'], panel['el'], sun_az, sun_el)
    if vibration_curve is not None:
        vib = np.array([vibration_curve(v) for v in wind])[period_index]
        error = np.sqrt(error ** 2 + vib ** 2)   # vibration enters the error metric only

    poa, power = _power(day, panel['az'], panel['el'], site, pv, sd, t_air_c)
    check_pv_output(power, poa, area)
    pv_energy = _energy_Wh(secs, power)
    ideal_energy = _ideal_energy_cached(date, _freeze(site), _freeze(pv), _freeze(sd), float(t_air_c))
    motor_energy = motor_energy_J / 3600.0
    result = {
        'rms_error_deg': float(np.sqrt(np.mean(error ** 2))),
        'max_error_deg': float(np.max(error)),
        'pv_energy_Wh': pv_energy,
        'motor_energy_Wh': motor_energy,
        'net_energy_Wh': pv_energy - motor_energy,
        'ideal_energy_Wh': ideal_energy,
        'energy_loss_pct': 100.0 * (ideal_energy - pv_energy) / ideal_energy,
        'n_moves': n_moves,
        'n_unsettled': n_unsettled,
    }
    if keep_series:
        result['series'] = pd.DataFrame({'time': times, 'sun_az': np.mod(sun_az, 360.0), 'sun_el': sun_el,
                                         'panel_az': np.mod(panel['az'], 360.0), 'panel_el': panel['el'],
                                         'error_deg': error, 'poa': poa, 'power_w': power})
    return result


def fixed_tilt_energy_Wh(date, site, pv, sd, t_air_c, tilt_deg, azimuth_deg=180.0):
    """Daily energy of a fixed panel with the given tilt, facing azimuth_deg."""
    day = _sun_and_sky(date, site['lat_deg'], site['lon_deg'], site['altitude_m'])
    n_points = len(day[1])
    _, power = _power(day, np.full(n_points, azimuth_deg), np.full(n_points, 90.0 - tilt_deg),
                      site, pv, sd, t_air_c)
    return _energy_Wh(day[1], power)
