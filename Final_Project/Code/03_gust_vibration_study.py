# File: 03_gust_vibration_study.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Gust-induced torsional vibration of the locked panel (continuous simulation) as a function of
#          mean wind speed; produces the vibration curve used by the event-based day simulation
# Inputs: data/site_parameters.csv, data/pv_parameters.csv, data/drive_parameters.csv;
#         turbulence intensity Iu and integral length scale Lu (parameter section below)
# Outputs: tables vibration_curve.csv, 03_vibration_by_axis.csv, 03_verification.csv;
#          figures 03_wind_series_example.png, 03_panel_deflection_example.png, 03_vibration_curve.png,
#          03_free_decay_fft.png
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from params import SEED, load_parameters, project_paths, axis_params
from model_checks import check_parameters
from disturbances import ou_wind_series, wind_torque
from drive_model import locked_panel_derivatives, natural_frequency_hz

script_dir = os.path.dirname(os.path.abspath(__file__))
paths = project_paths(__file__)
tables_dir = paths['tables']
figures_dir = paths['figures']

# ---------------------------------------------------------------- parameters
site = load_parameters(os.path.join(paths['data'], 'site_parameters.csv'))
pv = load_parameters(os.path.join(paths['data'], 'pv_parameters.csv'))
drive = load_parameters(os.path.join(paths['data'], 'drive_parameters.csv'))
ctrl = load_parameters(os.path.join(paths['data'], 'controller_parameters.csv'))
check_parameters(site, pv, drive, ctrl)
IU = 0.20                 # turbulence intensity (-): IEC 61400-1 class A reference ~0.16-0.2, open terrain
LU = 50.0                 # longitudinal integral length scale (m): near-ground value, ESDU/Eurocode order
SPEEDS_MPS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0)
DURATION_S = 600.0
DT_WIND_S = 0.01
MAX_STEP_S = 0.01
MAX_STEP_FINE_S = 0.005
DISCARD_S = 30.0          # initial transient discarded
EXAMPLE_SPEED_MPS = 10.0
FREE_DECAY_S = 200.0
FREE_DECAY_DT_S = 0.001
FREE_DECAY_PHI0_DEG = 0.1
LIMIT_POINTING_DEG = 0.5
VISIBLE_VIBRATION_DEG = 0.1
AXES = ('az', 'el')
area = pv['length_m'] * pv['width_m']

print('=== Parameters ===')
print(f'  Iu = {IU:g} (-), Lu = {LU:g} m, duration = {DURATION_S:g} s, wind dt = {DT_WIND_S:g} s, '
      f'max_step = {MAX_STEP_S:g} s, discard = {DISCARD_S:g} s, SEED = {SEED}')
for name in AXES:
    ax = axis_params(drive, name)
    print(f"  axis {name}: Jp = {ax['Jp']:g} kg*m^2, ks = {ax['ks']:g} N*m/rad, cs = {ax['cs']:g} N*m*s/rad, "
          f'f_n = {natural_frequency_hz(ax):.3f} Hz')
print(f"  rho = {site['air_density']:g} kg/m^3, A = {area:.4f} m^2, Cm = {drive['Cm']:g}, Lc = {drive['Lc']:g} m")

saved_tables = []
saved_figures = []


# ---------------------------------------------------------------- functions
def gust_response(ax, v_mean, max_step, seed):
    """Locked panel under turbulent wind torque; returns t, v(t), phi(t) (rad) on the wind grid."""
    rng = np.random.default_rng(seed)
    t_wind, v_wind = ou_wind_series(v_mean, IU, LU, DURATION_S, DT_WIND_S, rng)
    torque = wind_torque(v_wind, site['air_density'], area, drive['Cm'], drive['Lc'])
    torque_func = lambda t: np.interp(t, t_wind, torque)
    sol = solve_ivp(locked_panel_derivatives, (0.0, DURATION_S), [0.0, 0.0], args=(ax, torque_func),
                    t_eval=t_wind, max_step=max_step, rtol=1e-6, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'gust simulation failed at v_mean={v_mean}: {sol.message}')
    return t_wind, v_wind, sol.y[0]


def vibration_stats(t, phi):
    keep = t >= DISCARD_S
    phi_deg = np.degrees(phi[keep])
    return float(np.mean(phi_deg)), float(np.std(phi_deg)), float(np.max(np.abs(phi_deg)))


# ---------------------------------------------------------------- gust study
rows = []
examples = {}
for name in AXES:
    ax = axis_params(drive, name)
    for v_mean in SPEEDS_MPS:
        t, v, phi = gust_response(ax, v_mean, MAX_STEP_S, SEED)     # common random numbers over v and axes
        mean_def, rms_vib, max_def = vibration_stats(t, phi)
        rows.append({'axis': name, 'v_mean_mps': v_mean, 'mean_deflection_deg': mean_def,
                     'rms_vibration_deg': rms_vib, 'max_deflection_deg': max_def})
        if v_mean == EXAMPLE_SPEED_MPS:
            examples[name] = (t, v, phi)
        print(f'  axis {name}, v = {v_mean:4.1f} m/s: mean {mean_def:.4f}, rms {rms_vib:.4f}, max {max_def:.4f} deg')
df_axis = pd.DataFrame(rows)
df_axis.round(5).to_csv(os.path.join(tables_dir, '03_vibration_by_axis.csv'), index=False)
saved_tables.append('03_vibration_by_axis.csv')

df_curve = df_axis.groupby('v_mean_mps', as_index=False)['rms_vibration_deg'].max()
df_curve.round(5).to_csv(os.path.join(tables_dir, 'vibration_curve.csv'), index=False)
saved_tables.append('vibration_curve.csv')

# ---------------------------------------------------------------- verification
verification = []
for name in AXES:
    ax = axis_params(drive, name)
    t_free = np.arange(0.0, FREE_DECAY_S, FREE_DECAY_DT_S)
    sol = solve_ivp(locked_panel_derivatives, (0.0, FREE_DECAY_S), [np.radians(FREE_DECAY_PHI0_DEG), 0.0],
                    args=(ax, lambda t: 0.0), t_eval=t_free, max_step=0.01, rtol=1e-9, atol=1e-13)
    n_fft = 8 * len(t_free)                                  # zero padding for a finer frequency grid
    spectrum = np.abs(np.fft.rfft(sol.y[0] - np.mean(sol.y[0]), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, FREE_DECAY_DT_S)
    f_sim = float(freqs[np.argmax(spectrum)])
    f_formula = natural_frequency_hz(ax)
    verification += [
        {'Check': f'natural frequency, axis {name}', 'Case': 'free decay FFT peak', 'Value': f_sim, 'Unit': 'Hz'},
        {'Check': f'natural frequency, axis {name}', 'Case': 'sqrt(ks/Jp)/(2 pi)', 'Value': f_formula, 'Unit': 'Hz'},
        {'Check': f'natural frequency, axis {name}', 'Case': 'difference', 'Value': 100.0 * (f_sim - f_formula)
         / f_formula, 'Unit': '%'},
    ]
    if name == 'el':
        decay_example = (t_free, sol.y[0], freqs, spectrum, f_sim, f_formula)
    rms_coarse = float(df_axis.query('axis == @name and v_mean_mps == @EXAMPLE_SPEED_MPS')['rms_vibration_deg'].iloc[0])
    t, _, phi_fine = gust_response(ax, EXAMPLE_SPEED_MPS, MAX_STEP_FINE_S, SEED)
    rms_fine = vibration_stats(t, phi_fine)[1]
    verification += [
        {'Check': f'step halving at {EXAMPLE_SPEED_MPS:g} m/s, axis {name}', 'Case': f'max_step={MAX_STEP_S:g} s',
         'Value': rms_coarse, 'Unit': 'deg'},
        {'Check': f'step halving at {EXAMPLE_SPEED_MPS:g} m/s, axis {name}', 'Case': f'max_step={MAX_STEP_FINE_S:g} s',
         'Value': rms_fine, 'Unit': 'deg'},
        {'Check': f'step halving at {EXAMPLE_SPEED_MPS:g} m/s, axis {name}', 'Case': 'change',
         'Value': 100.0 * (rms_coarse - rms_fine) / rms_fine, 'Unit': '%'},
    ]
df_verify = pd.DataFrame(verification)
df_verify['Value'] = df_verify['Value'].round(6)
df_verify.to_csv(os.path.join(tables_dir, '03_verification.csv'), index=False)
saved_tables.append('03_verification.csv')

# ---------------------------------------------------------------- figures
t_ex, v_ex, _ = examples['el']
fig, ax_plot = plt.subplots(figsize=(7, 4.5))
ax_plot.plot(t_ex, v_ex, lw=0.6, label=f'turbulent wind, mean {EXAMPLE_SPEED_MPS:g} m/s')
ax_plot.axhline(EXAMPLE_SPEED_MPS, color='k', ls='--', label='mean')
ax_plot.set_xlim(0, DURATION_S)
ax_plot.set_ylim(0, np.ceil(v_ex.max() / 5.0) * 5.0)
ax_plot.set_xlabel('Time (s)')
ax_plot.set_ylabel('Wind speed (m/s)')
ax_plot.set_title(f'Ornstein-Uhlenbeck wind series (Iu = {IU:g}, Lu = {LU:g} m)')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
fig.savefig(os.path.join(figures_dir, '03_wind_series_example.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('03_wind_series_example.png')

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
phi_max = 0.0
for name in AXES:
    t_ex, _, phi_ex = examples[name]
    ax_plot.plot(t_ex, np.degrees(phi_ex), lw=0.6, label=f'{name} axis')
    phi_max = max(phi_max, float(np.max(np.abs(np.degrees(phi_ex)))))
ax_plot.axvline(DISCARD_S, color='k', ls=':', label=f'first {DISCARD_S:g} s discarded')
ax_plot.set_xlim(0, DURATION_S)
phi_lim = np.ceil(phi_max * 1.1 / 0.05) * 0.05
ax_plot.set_ylim(-phi_lim, phi_lim)      # deflection sign follows the wind-torque sign convention
ax_plot.set_xlabel('Time (s)')
ax_plot.set_ylabel('Panel torsional deflection (deg)')
ax_plot.set_title(f'Locked-panel deflection under gusts, mean wind {EXAMPLE_SPEED_MPS:g} m/s')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
fig.savefig(os.path.join(figures_dir, '03_panel_deflection_example.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('03_panel_deflection_example.png')

fig, ax_plot = plt.subplots(figsize=(7, 4.5))
for name in AXES:
    sub = df_axis[df_axis['axis'] == name]
    ax_plot.plot(sub['v_mean_mps'], sub['rms_vibration_deg'], 'o-', label=f'{name} axis')
ax_plot.axhline(VISIBLE_VIBRATION_DEG, color='k', ls=':', label=f'{VISIBLE_VIBRATION_DEG:g} deg')
ax_plot.set_xlim(0, 16)
ax_plot.set_ylim(0, max(np.ceil(df_axis['rms_vibration_deg'].max() * 1.1 / 0.05) * 0.05, 0.15))
ax_plot.set_xlabel('Mean wind speed (m/s)')
ax_plot.set_ylabel('RMS vibration (deg)')
ax_plot.set_title('Gust-induced RMS torsional vibration of the locked panel')
ax_plot.legend()
ax_plot.grid(alpha=0.3)
fig.savefig(os.path.join(figures_dir, '03_vibration_curve.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('03_vibration_curve.png')

t_free, phi_free, freqs, spectrum, f_sim, f_formula = decay_example
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
axs[0].plot(t_free, np.degrees(phi_free), lw=0.6)
axs[0].set_xlim(0, 10)
axs[0].set_ylim(-FREE_DECAY_PHI0_DEG, FREE_DECAY_PHI0_DEG)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Deflection (deg)')
axs[0].set_title('Free decay, elevation axis (first 10 s)')
axs[0].grid(alpha=0.3)
axs[1].plot(freqs, spectrum / spectrum.max(), label='FFT magnitude')
axs[1].axvline(f_formula, color='k', ls='--', label=f'formula {f_formula:.3f} Hz')
axs[1].set_xlim(0, 10)
axs[1].set_ylim(0, 1.05)
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Normalised magnitude (-)')
axs[1].set_title(f'Spectrum: peak {f_sim:.3f} Hz')
axs[1].legend()
axs[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, '03_free_decay_fft.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
saved_figures.append('03_free_decay_fft.png')

# ---------------------------------------------------------------- summary
print('\n=== Acceptance checks ===')
for _, row in df_verify.iterrows():
    print(f"  {row['Check']:38s} {row['Case']:22s} {row['Value']:.6g} {row['Unit']}")
monotonic = all(np.all(np.diff(df_axis[df_axis['axis'] == name]['rms_vibration_deg'].to_numpy()) > 0)
                for name in AXES)
print(f'  curve increasing with mean wind: {monotonic}')

ranges = pd.read_csv(os.path.join(paths['data'], 'disturbance_ranges.csv')).set_index('Parameter')
v_site = float(ranges.loc['v_mean_mps', 'Nominal'])
vib_site = float(np.interp(v_site, df_curve['v_mean_mps'], df_curve['rms_vibration_deg']))
above = df_curve[df_curve['rms_vibration_deg'] > VISIBLE_VIBRATION_DEG]
v_cross = (float(np.interp(VISIBLE_VIBRATION_DEG, df_curve['rms_vibration_deg'], df_curve['v_mean_mps']))
           if len(above) and df_curve['rms_vibration_deg'].iloc[0] < VISIBLE_VIBRATION_DEG else None)
print('\n=== Conclusion ===')
print(f'  At the Madinah daytime mean wind ({v_site:g} m/s) the RMS vibration is {vib_site:.4f} deg, i.e. '
      f'{100 * vib_site / LIMIT_POINTING_DEG:.1f} % of the {LIMIT_POINTING_DEG:g} deg pointing limit '
      f'({"negligible" if vib_site < 0.1 * LIMIT_POINTING_DEG else "not negligible"}).')
if v_cross is not None:
    print(f'  The vibration exceeds {VISIBLE_VIBRATION_DEG:g} deg above a mean wind of about {v_cross:.1f} m/s.')
else:
    print(f'  The vibration stays below {VISIBLE_VIBRATION_DEG:g} deg up to '
          f'{SPEEDS_MPS[-1]:g} m/s' if len(above) == 0 else
          f'  The vibration already exceeds {VISIBLE_VIBRATION_DEG:g} deg at {SPEEDS_MPS[0]:g} m/s.')

print('\n=== Files saved ===')
for name in saved_tables:
    print(f'  table : results/tables/{name}')
for name in saved_figures:
    print(f'  figure: results/figures/{name}')
