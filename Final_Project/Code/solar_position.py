# File: solar_position.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Sun position (NREL SPA via pvlib and a simplified astronomical model) and angular pointing error.
# Inputs: timezone-aware times, latitude/longitude/altitude, optional pressure/temperature/delta_t
# Outputs: DataFrames with azimuth_deg (from north, clockwise) and elevation_deg; pointing error in degrees
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import numpy as np
import pandas as pd
from pvlib import solarposition


def sun_position_spa(times, lat_deg, lon_deg, altitude_m, pressure_pa=None,
                     temperature_c=25.0, delta_t=None):
    """NREL SPA (pvlib.spa_python): apparent elevation (with refraction) and azimuth in degrees."""
    kwargs = {'temperature': temperature_c, 'atmos_refract': 0.5667}
    if pressure_pa is not None:
        kwargs['pressure'] = pressure_pa
    if delta_t is not None:
        kwargs['delta_t'] = delta_t
    sp = solarposition.spa_python(times, lat_deg, lon_deg, altitude=altitude_m, **kwargs)
    return pd.DataFrame({'azimuth_deg': sp['azimuth'].to_numpy(),
                         'elevation_deg': sp['apparent_elevation'].to_numpy()}, index=times)


def sun_position_simple(times, lat_deg, lon_deg):
    """Simplified model: Spencer declination and equation of time, hour angle, no refraction."""
    times_utc = times.tz_convert('UTC')
    doy = times_utc.dayofyear.to_numpy()
    hour_utc = (times_utc.hour + times_utc.minute / 60.0 + times_utc.second / 3600.0).to_numpy()
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (hour_utc - 12.0) / 24.0)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eot_min = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    solar_time_min = hour_utc * 60.0 + eot_min + 4.0 * lon_deg
    hour_angle = np.radians(solar_time_min / 4.0 - 180.0)
    lat = np.radians(lat_deg)
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    elevation = np.arcsin(np.clip(sin_el, -1.0, 1.0))
    # Azimuth from north, clockwise (east = 90 deg).
    azimuth = np.arctan2(-np.cos(decl) * np.sin(hour_angle),
                         np.cos(lat) * np.sin(decl) - np.sin(lat) * np.cos(decl) * np.cos(hour_angle))
    return pd.DataFrame({'azimuth_deg': np.mod(np.degrees(azimuth), 360.0),
                         'elevation_deg': np.degrees(elevation)}, index=times)


def unit_vector(az_deg, el_deg):
    """East-North-Up unit vector for azimuth (from north, clockwise) and elevation in degrees."""
    az = np.radians(np.asarray(az_deg, dtype=float))
    el = np.radians(np.asarray(el_deg, dtype=float))
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], axis=-1)


def pointing_error_deg(az1_deg, el1_deg, az2_deg, el2_deg):
    """Angle between two directions in degrees (atan2 form, accurate for small angles)."""
    u1 = unit_vector(az1_deg, el1_deg)
    u2 = unit_vector(az2_deg, el2_deg)
    cross_norm = np.linalg.norm(np.cross(u1, u2), axis=-1)
    dot = np.sum(u1 * u2, axis=-1)
    return np.degrees(np.arctan2(cross_norm, dot))
