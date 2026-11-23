# File: model_checks.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Automatic model/code checks that reject nonphysical inputs and outputs with ValueError.
# Inputs: site, pv, drive, controller parameter dicts; disturbance levels and ranges; PV power series
# Outputs: None (raises ValueError on any violation)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import numpy as np

POSITIVE_DRIVE = ('R', 'L', 'Kt', 'Ke', 'Jm', 'omega_s', 'N', 'eta', 'T_rated',
                  'Jp_el', 'Jp_az', 'ks_el', 'ks_az', 'cs_el', 'cs_az', 'Cm', 'Lc')
NONNEGATIVE_DRIVE = ('b',)
POSITIVE_PV = ('P_max', 'V_mp', 'I_mp', 'V_oc', 'I_sc', 'cells_in_series',
               'noct_c', 'length_m', 'width_m')
POSITIVE_CTRL = ('Kp', 'V_max', 'tol_deg', 'update_s', 'timeout_s')
NONNEGATIVE_CTRL = ('Ki', 'Kd')


def _require(params, keys, source, strictly_positive):
    for key in keys:
        if key not in params:
            raise ValueError(f'{source}: missing parameter {key}')
        value = params[key]
        if not np.isfinite(value):
            raise ValueError(f'{source}: parameter {key} = {value} is not finite')
        if strictly_positive and value <= 0:
            raise ValueError(f'{source}: parameter {key} = {value} must be > 0')
        if not strictly_positive and value < 0:
            raise ValueError(f'{source}: parameter {key} = {value} must be >= 0')


def check_parameters(site, pv, drive, ctrl):
    """Reject nonphysical site, PV, drive and controller parameters."""
    lat = site['lat_deg']
    lon = site['lon_deg']
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f'site_parameters.csv: lat_deg = {lat} outside [-90, 90]')
    if not -180.0 <= lon <= 180.0:
        raise ValueError(f'site_parameters.csv: lon_deg = {lon} outside [-180, 180]')
    if not 0.0 <= site['albedo'] <= 1.0:
        raise ValueError(f"site_parameters.csv: albedo = {site['albedo']} outside [0, 1]")
    _require(site, ('air_density',), 'site_parameters.csv', True)
    _require(pv, POSITIVE_PV, 'pv_parameters.csv', True)
    _require(drive, POSITIVE_DRIVE, 'drive_parameters.csv', True)
    _require(drive, NONNEGATIVE_DRIVE, 'drive_parameters.csv', False)
    if drive['eta'] > 1.0:
        raise ValueError(f"drive_parameters.csv: eta = {drive['eta']} must be <= 1")
    _require(ctrl, POSITIVE_CTRL, 'controller_parameters.csv', True)
    _require(ctrl, NONNEGATIVE_CTRL, 'controller_parameters.csv', False)


def check_levels(levels, ranges):
    """Reject a disturbance level that is unknown or outside its [Min, Max] range."""
    for name, value in levels.items():
        if name not in ranges.index:
            raise ValueError(f'disturbance level {name} is not defined in disturbance_ranges.csv')
        low = float(ranges.loc[name, 'Min'])
        high = float(ranges.loc[name, 'Max'])
        if not np.isfinite(value) or value < low or value > high:
            raise ValueError(f'disturbance level {name} = {value} outside [{low}, {high}]')


def check_pv_output(power_w, poa_wm2, area_m2):
    """Reject negative PV power or power above the incident power G*A."""
    power = np.atleast_1d(np.asarray(power_w, dtype=float))
    poa = np.atleast_1d(np.asarray(poa_wm2, dtype=float))
    if area_m2 <= 0:
        raise ValueError(f'panel area = {area_m2} m^2 must be > 0')
    if np.any(~np.isfinite(power)):
        raise ValueError('PV power contains non-finite values')
    if np.any(power < 0):
        raise ValueError(f'PV power negative: min = {power.min():.4f} W')
    incident = np.maximum(poa, 0.0) * area_m2
    excess = power - incident
    if np.any(excess > 1e-9):
        idx = int(np.argmax(excess))
        raise ValueError(f'PV power {power[idx]:.4f} W exceeds incident power G*A = {incident[idx]:.4f} W')
