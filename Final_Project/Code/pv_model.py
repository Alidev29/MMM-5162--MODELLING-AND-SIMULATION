# File: pv_model.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: PV panel model: clear-sky irradiance, plane-of-array irradiance, NOCT cell temperature,
#          De Soto single-diode power at the maximum power point, I-V curve.
# Inputs: pv parameter dict (pv_parameters.csv), irradiance components, sun and panel angles
# Outputs: irradiance (W/m^2), cell temperature (degC), power (W), I-V arrays
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import numpy as np
from pvlib import ivtools, pvsystem
from pvlib.location import Location

SD_KEYS = ('I_L_ref', 'I_o_ref', 'R_s', 'R_sh_ref', 'a_ref')


def fit_single_diode(pv):
    """Fit De Soto single-diode reference parameters from datasheet values (pvlib fit_desoto)."""
    if not (pv['V_mp'] < pv['V_oc'] and pv['I_mp'] < pv['I_sc']):
        raise ValueError('pv_parameters.csv: single-diode fit failed - need V_mp < V_oc and I_mp < I_sc '
                         f"(V_mp={pv['V_mp']}, V_oc={pv['V_oc']}, I_mp={pv['I_mp']}, I_sc={pv['I_sc']})")
    try:
        fit, result = ivtools.sdm.fit_desoto(
            v_mp=pv['V_mp'], i_mp=pv['I_mp'], v_oc=pv['V_oc'], i_sc=pv['I_sc'],
            alpha_sc=pv['alpha_sc'], beta_voc=pv['beta_voc'],
            cells_in_series=int(pv['cells_in_series']),
            root_kwargs={'method': 'lm'})  # default 'hybr' stalls on 72-cell datasheets
    except Exception as exc:  # pvlib raises RuntimeError on non-convergence
        raise ValueError(f'pv_parameters.csv: single-diode fit failed ({exc})') from exc
    sd = {key: float(fit[key]) for key in SD_KEYS}
    if (not getattr(result, 'success', True)) or any(not np.isfinite(v) or v <= 0 for v in sd.values()):
        raise ValueError(f'pv_parameters.csv: single-diode fit failed - unphysical parameters {sd}')
    return sd


def clear_sky_irradiance(times, lat_deg, lon_deg, altitude_m):
    """Ineichen clear-sky GHI, DNI, DHI (W/m^2) at the site."""
    site = Location(lat_deg, lon_deg, tz=str(times.tz), altitude=altitude_m)
    return site.get_clearsky(times, model='ineichen')[['ghi', 'dni', 'dhi']]


def plane_of_array_irradiance(dni, dhi, ghi, sun_az_deg, sun_el_deg, panel_az_deg, panel_el_deg, albedo):
    """POA irradiance (W/m^2): beam + isotropic sky diffuse + ground reflected.
    panel_el_deg is the elevation of the panel normal; tilt beta = 90 - panel_el."""
    sun_az = np.radians(np.asarray(sun_az_deg, dtype=float))
    sun_el = np.radians(np.asarray(sun_el_deg, dtype=float))
    pan_az = np.radians(np.asarray(panel_az_deg, dtype=float))
    pan_el = np.radians(np.asarray(panel_el_deg, dtype=float))
    cos_theta = (np.sin(sun_el) * np.sin(pan_el)
                 + np.cos(sun_el) * np.cos(pan_el) * np.cos(sun_az - pan_az))
    cos_beta = np.sin(pan_el)  # cos(90 - el)
    beam = np.asarray(dni, dtype=float) * np.maximum(cos_theta, 0.0)
    diffuse = np.asarray(dhi, dtype=float) * (1.0 + cos_beta) / 2.0
    reflected = np.asarray(ghi, dtype=float) * albedo * (1.0 - cos_beta) / 2.0
    return beam + diffuse + reflected


def cell_temperature(poa, t_air_c, noct_c):
    """NOCT model: T_cell = T_air + (NOCT - 20)/800 * POA."""
    return np.asarray(t_air_c, dtype=float) + (noct_c - 20.0) / 800.0 * np.asarray(poa, dtype=float)


def _desoto_params(poa, t_cell_c, pv, sd):
    return pvsystem.calcparams_desoto(poa, t_cell_c, pv['alpha_sc'], sd['a_ref'], sd['I_L_ref'],
                                      sd['I_o_ref'], sd['R_sh_ref'], sd['R_s'])


def pv_power(poa, t_cell_c, pv, sd):
    """Maximum power point power (W); zero where poa <= 0."""
    poa_arr = np.atleast_1d(np.asarray(poa, dtype=float))
    t_cell = np.broadcast_to(np.asarray(t_cell_c, dtype=float), poa_arr.shape)
    power = np.zeros_like(poa_arr)
    lit = poa_arr > 0
    if np.any(lit):
        il, io, rs, rsh, nnsvth = _desoto_params(poa_arr[lit], t_cell[lit], pv, sd)
        out = pvsystem.singlediode(il, io, rs, rsh, nnsvth)
        power[lit] = np.maximum(np.asarray(out['p_mp'], dtype=float), 0.0)
    return power


def iv_curve(poa, t_cell_c, pv, sd, n_points=100):
    """I-V curve (V, A) from short circuit to open circuit."""
    il, io, rs, rsh, nnsvth = _desoto_params(poa, t_cell_c, pv, sd)
    out = pvsystem.singlediode(il, io, rs, rsh, nnsvth)
    voc = float(np.asarray(out['v_oc']))
    v = np.linspace(0.0, voc, n_points)
    i = np.asarray(pvsystem.i_from_v(v, il, io, rs, rsh, nnsvth), dtype=float)
    return v, np.maximum(i, 0.0)
