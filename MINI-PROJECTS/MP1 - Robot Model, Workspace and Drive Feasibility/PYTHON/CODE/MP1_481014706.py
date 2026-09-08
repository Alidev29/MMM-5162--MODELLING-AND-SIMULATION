# File: MP1_481014706.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: how cycle time changes the required joint speed for Two-Link Robot
# Inputs: L1, L2, q1_start, q1_end, q2_start, q2_end, T
# Outputs: end-effector path plot, joint angle/velocity plots
# Author: Ali Khuzam
# Date: 09/09/2026
#==========================================================================
# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#==========================================================================
# files path
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(script_dir, '..', 'results')
tables_dir = os.path.join(results_dir, 'tables')
figures_dir = os.path.join(results_dir, 'figures')
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
#==========================================================================
# Parameters
S = 8
L1 = (0.38+(0.005*S)) #length in (m)
L2 = (0.30+(0.004*S)) #length in (m)
q1_s = (20+S)          #angle in d
q1_e = (75-(0.5*S))    #angle in d
q2_s = ((-55)+(0.8*S)) #angle in d
q2_e = (15+(0.5*S))    #angle in d
T = (2.4+(0.04*S))  #Time in (s)

# convert q from degree to radians
q1_start = np.radians(q1_s)
q1_end = np.radians(q1_e)
q2_start = np.radians(q2_s)
q2_end = np.radians(q2_e)

print("="*55)
print(f" L1            = {L1:.4f} m")
print(f" L2            = {L2:.4f} m")
print(f" q1_start      = {q1_s:.2f} deg = {q1_start:.4f} rad")
print(f" q1_end        = {q1_e:.2f} deg = {q1_end:.4f} rad")
print(f" q2_start      = {q2_s:.2f} deg = {q2_start:.4f} rad")
print(f" q2_end        = {q2_e:.2f} deg = {q2_end:.4f} rad")
print(f" T (nominal)   = {T:.4f} s")
print("="*55)
#============================================
# Equations
def s_of_u(u):
    return 3*u**2 - 2*u**3

def sdot_of_u(u):
    return 6*u - 6*u**2

def joint_trajectory(t, T, q_start, q_end):
    u = t / T
    s = s_of_u(u)
    sd = sdot_of_u(u)
    q = q_start + (q_end - q_start) * s
    qdot = (q_end - q_start) * sd * (1.0 / T)   
    return q, qdot

def forward_kinematics(q1, q2, L1, L2):
    x = L1*np.cos(q1) + L2*np.cos(q1 + q2)
    y = L1*np.sin(q1) + L2*np.sin(q1 + q2)
    return x, y

def validate_motion_inputs(t, T, L1, L2):
    #  Rejects nonphysical inputs
    if L1 <= 0 or L2 <= 0:
        raise ValueError(f"Nonphysical link length detected: L1={L1}, L2={L2} (both must be > 0 m)")
    if T <= 0:
        raise ValueError(f"Nonphysical cycle time detected: T={T} (must be > 0 s)")
    if np.any(t < 0) or np.any(t > T):
        raise ValueError(f"Time value(s) outside valid motion interval [0, {T}] s")
    return True

#============================================
# Simulate
N = 1000
t = np.linspace(0, T, N)

# Test before start calculations
validate_motion_inputs(t, T, L1, L2)
print(f"\nAutomatic check passed: L1, L2 > 0; T > 0; all t within [0, {T:.4f}] s")


q1_t, q1dot_t = joint_trajectory(t, T, q1_start, q1_end)
q2_t, q2dot_t = joint_trajectory(t, T, q2_start, q2_end)
x_t, y_t = forward_kinematics(q1_t, q2_t, L1, L2)

max_diff_q1 = np.max(np.abs(q1dot_t))
max_diff_q2 = np.max(np.abs(q2dot_t))

dx = np.diff(x_t); dy = np.diff(y_t)
path_length = np.sum(np.sqrt(dx**2 + dy**2))

print(f"\nEnd-effector start point: ({x_t[0]:.4f}, {y_t[0]:.4f}) m")
print(f"End-effector end point:   ({x_t[-1]:.4f}, {y_t[-1]:.4f}) m")
print(f"Path length (arc length): {path_length:.4f} m")
print(f"Max |q1dot| = {max_diff_q1:.4f} rad/s")
print(f"Max |q2dot| = {max_diff_q2:.4f} rad/s")

#============================================
# Tables
nominal_results_df = pd. DataFrame({
    'Metric': ['Max |q1dot| (rad/s)', 'Max |q2dot| (rad/s)', 
               'Path length (m)', 'x_start (m)', 'y_start (m)',
               'x_end (m)', 'y_end (m)'],
    'Value': [np.max(np.abs(q1dot_t)), np.max(np.abs(q2dot_t)),
              path_length, x_t[0], y_t[0], x_t[-1], y_t[-1]]})
nominal_results_df.to_csv(os.path.join(tables_dir, 'results_summary.csv'), index=False)

df_timeseries = pd.DataFrame({
    't': t,
    'q1': q1_t,
    'q2': q2_t,
    'q1dot': q1dot_t,
    'q2dot': q2dot_t,
    'x': x_t,
    'y': y_t
})
df_timeseries.to_csv(os.path.join(tables_dir, 'timeseries_data.csv'), index=False)

#============================================
# Plot
# (a) End-effector path in workspace
plt.figure(figsize=(7, 6))
plt.plot(x_t, y_t, 'b-', lw=2)
plt.plot(x_t[0], y_t[0], 'go', ms=10, label='Start (pick)')
plt.plot(x_t[-1], y_t[-1], 'ro', ms=10, label='End (place)')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.title('End-Effector Path in Workspace')
plt.axis('equal'); plt.grid(True, alpha=0.3); plt.legend()
plt.xlim(0.0, 0.8); plt.ylim(0.0, 0.8)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig1_workspace_path.png'), dpi=300)
plt.close()
 
# (b) Joint angles vs time
plt.figure(figsize=(7, 6))
plt.plot(t, np.degrees(q1_t), label='q1(t)')
plt.plot(t, np.degrees(q2_t), label='q2(t)')
plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)')
plt.title('Joint Angles vs Time')
plt.grid(True, alpha=0.3); plt.legend()
plt.xlim(0.0, 3.0); plt.ylim(-60, 80)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig2_joint_angles.png'), dpi=300)
plt.close()
 
# (c) Joint velocities vs time
plt.figure(figsize=(7, 6))
plt.plot(t, q1dot_t, label='q1dot(t)')
plt.plot(t, q2dot_t, label='q2dot(t)')
plt.xlabel('Time (s)'); plt.ylabel('Angular velocity (rad/s)')
plt.title('Joint Angular Velocities vs Time')
plt.grid(True, alpha=0.3); plt.legend()
plt.xlim(0.0, 3.0); plt.ylim(0.0, 0.7)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig3_joint_velocities.png'), dpi=300)
plt.close()
 
# (d) x(t), y(t) vs time
plt.figure(figsize=(7, 6))
plt.plot(t, x_t, label='x(t)')
plt.plot(t, y_t, label='y(t)')
plt.xlabel('Time (s)'); plt.ylabel('Position (m)')
plt.title('End-Effector Coordinates vs Time')
plt.grid(True, alpha=0.3); plt.legend()
plt.xlim(0.0, 3.0); plt.ylim(0.0, 0.8)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig4_coordinates_vs_time.png'), dpi=300)
plt.close()

#============================================
# # PARAMETER STUDY: cycle times 0.8T, T, 1.2T
# Required (maximum |q1|, maximum |q2|, path length or end-point coordinates.)

factors = [0.8, 1.0, 1.2]
colors = ['tab:orange', 'tab:blue', 'tab:green']
results = {}
 
fig2, axs2 = plt.subplots(1, 2, figsize=(13, 5))
for f, c in zip(factors, colors):
    T_case = f * T
    t_case = np.linspace(0, T_case, N)
    validate_motion_inputs(t_case, T_case, L1, L2) 
    q1c, q1dc = joint_trajectory(t_case, T_case, q1_start, q1_end)
    q2c, q2dc = joint_trajectory(t_case, T_case, q2_start, q2_end)
    xc, yc = forward_kinematics(q1c, q2c, L1, L2)
 
    dxc = np.diff(xc); dyc = np.diff(yc)
    path_len_c = np.sum(np.sqrt(dxc**2 + dyc**2))
 
    results[f] = dict(T=T_case,
                       max_q1dot=np.max(np.abs(q1dc)),
                       max_q2dot=np.max(np.abs(q2dc)),
                       path_length=path_len_c,
                       end_point=(xc[-1], yc[-1]))
 
    axs2[0].plot(t_case, q1dc, color=c, label=f'{f}T  (T={T_case:.2f}s)')
    axs2[1].plot(t_case, q2dc, color=c, label=f'{f}T  (T={T_case:.2f}s)')
 
axs2[0].set_xlabel('Time (s)'); axs2[0].set_ylabel('q1dot (rad/s)')
axs2[0].set_title('Joint 1 Velocity: Effect of Cycle Time')
axs2[0].grid(True, alpha=0.3); axs2[0].legend()
axs2[0].set_xlim(0.0, 3.5); axs2[0].set_ylim(0.0, 0.9)
 
axs2[1].set_xlabel('Time (s)'); axs2[1].set_ylabel('q2dot (rad/s)')
axs2[1].set_title('Joint 2 Velocity: Effect of Cycle Time')
axs2[1].grid(True, alpha=0.3); axs2[1].legend()
axs2[1].set_xlim(0.0, 3.5); axs2[1].set_ylim(0.0, 0.9)
 
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'fig5_parameter_study.png'), dpi=300)
plt.close()
 
#============================================
# SUMMARY TABLE OF PERFORMANCE MEASURES
print("\n" + "="*70)
print(" PARAMETER STUDY SUMMARY (cycle times 0.8T, T, 1.2T)")
print("="*70)
print(f"{'Case':<10}{'T (s)':<10}{'max|q1dot|':<14}{'max|q2dot|':<14}{'Path len (m)':<14}")
for f in factors:
    r = results[f]
    print(f"{str(f)+'T':<10}{r['T']:<10.3f}{r['max_q1dot']:<14.4f}{r['max_q2dot']:<14.4f}{r['path_length']:<14.4f}")
 
for f in factors:
    r = results[f]
    print(f"  {f}T -> ({r['end_point'][0]:.4f}, {r['end_point'][1]:.4f}) m")

param_study_df = pd.DataFrame([
    {'Case': f'{f}T', 'T (s)': round(results[f]['T'], 3),
     'max_|q1dot| (rad/s)': round(results[f]['max_q1dot'], 4),
     'max_|q2dot| (rad/s)': round(results[f]['max_q2dot'], 4)}
    for f in factors
])
param_study_df.to_csv(os.path.join(tables_dir, 'parameter_study_results.csv'), index=False)
 
print("\nDone. Plots saved: fig1_workspace_path.png, fig2_joint_angles.png,")
print("fig3_joint_velocities.png, fig4_coordinates_vs_time.png, fig5_parameter_study.png")
print("Tables saved: results_summary.csv, timeseries_data.csv, parameter_study_results.csv")