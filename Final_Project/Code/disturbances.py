# File: disturbances.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Disturbance models: Coulomb+viscous friction, wind torque, backlash dead zone,
#          Ornstein-Uhlenbeck turbulent wind, static hold offset (backlash + wind twist).
# Inputs: speeds, torques and parameters passed as arguments (no hard-coded physical values)
# Outputs: torques (N*m), angles (rad), wind series (m/s)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import numpy as np


def friction_torque(omega, Tc, b, omega_s):
    """Smooth Coulomb + viscous friction: Tc*tanh(omega/omega_s) + b*omega (N*m)."""
    return Tc * np.tanh(omega / omega_s) + b * omega


def wind_torque(v, rho, area, Cm, Lc):
    """Aerodynamic torque about the axis: 0.5*rho*v^2*A*Cm*Lc (N*m)."""
    return 0.5 * rho * np.square(v) * area * Cm * Lc


def dead_zone(x, delta):
    """Backlash dead zone: zero inside +/-delta, x -/+ delta outside."""
    return np.where(x > delta, x - delta, np.where(x < -delta, x + delta, 0.0)) * 1.0 \
        if np.ndim(x) else (x - delta if x > delta else (x + delta if x < -delta else 0.0))


def ou_wind_series(v_mean, Iu, Lu, duration_s, dt, rng):
    """Turbulent wind v = v_mean + v', v' an Ornstein-Uhlenbeck process (sigma = Iu*v_mean,
    tau = Lu/v_mean) with the exact discrete update; v >= 0. Zero mean wind gives zeros."""
    t = np.arange(0.0, duration_s + 0.5 * dt, dt)
    if v_mean <= 0.0:
        return t, np.zeros_like(t)
    sigma = Iu * v_mean
    tau = Lu / v_mean
    decay = np.exp(-dt / tau)
    kick = sigma * np.sqrt(1.0 - decay ** 2)
    noise = rng.standard_normal(len(t))
    fluct = np.empty_like(t)
    fluct[0] = sigma * noise[0]
    for k in range(1, len(t)):
        fluct[k] = decay * fluct[k - 1] + kick * noise[k]
    return t, np.maximum(v_mean + fluct, 0.0)


def hold_offset_rad(v, backlash_rad, ks, rho, area, Cm, Lc, side):
    """Static panel offset while the worm drive holds: side*(backlash + T_wind/ks) (rad)."""
    return side * (backlash_rad + wind_torque(v, rho, area, Cm, Lc) / ks)
