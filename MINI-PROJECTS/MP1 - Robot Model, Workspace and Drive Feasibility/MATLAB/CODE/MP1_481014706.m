%% File: MP1_481014706.py
% Course: MMM 5162 Modelling and Simulation
% Purpose: how cycle time changes the required joint speed for Two-Link Robot
% Inputs: L1, L2, q1_start, q1_end, q2_start, q2_end, T
% Outputs: end-effector path plot, joint angle/velocity plots
% Author: Ali Khuzam
% Date: 09/09/2026
% Student parameter S = 8

clear; clc; close all;
%==========================================================================
%% files path
script_dir = fileparts(mfilename('fullpath'));
results_dir = fullfile(script_dir, '..', 'results');
tables_dir = fullfile(results_dir, 'tables');
figures_dir = fullfile(results_dir, 'figures');
if ~exist(tables_dir, 'dir'); mkdir(tables_dir); end
if ~exist(figures_dir, 'dir'); mkdir(figures_dir); end
%==========================================================================

%% Parameters
S = 8;
L1 = 0.38 + (0.005*S);   % length in (m)
L2 = 0.30 + (0.004*S);   % length in (m)
q1_s = 20 + S;           % angle in deg
q1_e = 75 - (0.5*S);     % angle in deg
q2_s = (-55) + (0.8*S);  % angle in deg
q2_e = 15 + (0.5*S);     % angle in deg
T = 2.4 + (0.04*S);      % Time in (s)

% convert q from degree to radians
q1_start = deg2rad(q1_s);
q1_end   = deg2rad(q1_e);
q2_start = deg2rad(q2_s);
q2_end   = deg2rad(q2_e);

fprintf('=======================================================\n');
fprintf(' L1            = %.4f m\n', L1);
fprintf(' L2            = %.4f m\n', L2);
fprintf(' q1_start      = %.2f deg = %.4f rad\n', q1_s, q1_start);
fprintf(' q1_end        = %.2f deg = %.4f rad\n', q1_e, q1_end);
fprintf(' q2_start      = %.2f deg = %.4f rad\n', q2_s, q2_start);
fprintf(' q2_end        = %.2f deg = %.4f rad\n', q2_e, q2_end);
fprintf(' T (nominal)   = %.4f s\n', T);
fprintf('=======================================================\n');

%============================================
%% FUNCTIONS
function s = s_of_u(u)
    s = 3*u.^2 - 2*u.^3;
end
 
function sd = sdot_of_u(u)
    sd = 6*u - 6*u.^2;
end
 
function [q, qdot] = joint_trajectory(t, T, q_start, q_end)
    u = t / T;
    s = s_of_u(u);
    sd = sdot_of_u(u);
    q = q_start + (q_end - q_start) * s;
    qdot = (q_end - q_start) * sd * (1.0 / T);
end
 
function [x, y] = forward_kinematics(q1, q2, L1, L2)
    x = L1*cos(q1) + L2*cos(q1 + q2);
    y = L1*sin(q1) + L2*sin(q1 + q2);
end

function ok = validate_motion_inputs(t, T, L1, L2)
    % Rejects nonphysical inputs

    if L1 <= 0 || L2 <= 0
        error('validate_motion_inputs:badLength', ...
            'Nonphysical link length detected: L1=%.4f, L2=%.4f (both must be > 0 m)', L1, L2);
    end
    if T <= 0
        error('validate_motion_inputs:badTime', ...
            'Nonphysical cycle time detected: T=%.4f (must be > 0 s)', T);
    end
    if any(t < 0) || any(t > T)
        error('validate_motion_inputs:badInterval', ...
            'Time value(s) outside valid motion interval [0, %.4f] s', T);
    end
    ok = true;
end

%============================================
%% Simulate
N = 1000;
t = linspace(0, T, N);

% Test before start calculations
validate_motion_inputs(t, T, L1, L2);
fprintf('\nAutomatic check passed: L1, L2 > 0; T > 0; all t within [0, %.4f] s\n', T);

[q1_t, q1dot_t] = joint_trajectory(t, T, q1_start, q1_end);
[q2_t, q2dot_t] = joint_trajectory(t, T, q2_start, q2_end);
[x_t, y_t] = forward_kinematics(q1_t, q2_t, L1, L2);
 
max_diff_q1 = max(abs(q1dot_t));
max_diff_q2 = max(abs(q2dot_t));
 
dx = diff(x_t); dy = diff(y_t);
path_length = sum(sqrt(dx.^2 + dy.^2));
 
fprintf('\nEnd-effector start point: (%.4f, %.4f) m\n', x_t(1), y_t(1));
fprintf('End-effector end point:   (%.4f, %.4f) m\n', x_t(end), y_t(end));
fprintf('Path length (arc length): %.4f m\n', path_length);
fprintf('Max |q1dot| = %.4f rad/s\n', max_diff_q1);
fprintf('Max |q2dot| = %.4f rad/s\n', max_diff_q2);

%============================================
%% Tables
Metric = {'Max |q1dot| (rad/s)'; 'Max |q2dot| (rad/s)'; ...
          'Path length (m)'; 'x_start (m)'; 'y_start (m)'; ...
          'x_end (m)'; 'y_end (m)'};
Value = [max_diff_q1; max_diff_q2; path_length; ...
         x_t(1); y_t(1); x_t(end); y_t(end)];
nominal_results_table = table(Metric, Value);
writetable(nominal_results_table, fullfile(tables_dir, 'results_summary.csv'));
 
timeseries_table = table(t', q1_t', q2_t', q1dot_t', q2dot_t', x_t', y_t', ...
    'VariableNames', {'t', 'q1', 'q2', 'q1dot', 'q2dot', 'x', 'y'});
writetable(timeseries_table, fullfile(tables_dir, 'timeseries_data.csv'));
 
%============================================
%% Plot
% (a) End-effector path in workspace
figure('Position', [100 100 700 600]);
plot(x_t, y_t, 'b-', 'LineWidth', 2); hold on;
plot(x_t(1), y_t(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot(x_t(end), y_t(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('x (m)'); ylabel('y (m)');
title('End-Effector Path in Workspace');
axis equal; grid on;
legend('Path', 'Start (pick)', 'End (place)', 'Location', 'best');
axis([0 0.8 0.0 0.8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig1_workspace_path.png'), 'Resolution', 300);

 
% (b) Joint angles vs time
figure('Position', [100 100 700 600]);
plot(t, rad2deg(q1_t), 'LineWidth', 1.5); hold on;
plot(t, rad2deg(q2_t), 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Angle (deg)');
title('Joint Angles vs Time');
grid on; legend('q1(t)', 'q2(t)', 'Location', 'best');
axis([0 3 -60 80]);
exportgraphics(gcf, fullfile(figures_dir, 'fig2_joint_angles.png'), 'Resolution', 300);

 
% (c) Joint velocities vs time
figure('Position', [100 100 700 600]);
plot(t, q1dot_t, 'LineWidth', 1.5); hold on;
plot(t, q2dot_t, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Angular velocity (rad/s)');
title('Joint Angular Velocities vs Time');
grid on; legend('q1dot(t)', 'q2dot(t)', 'Location', 'best');
axis([0 3 0 0.7]);
exportgraphics(gcf, fullfile(figures_dir, 'fig3_joint_velocities.png'), 'Resolution', 300);

 
% (d) x(t), y(t) vs time
figure('Position', [100 100 700 600]);
plot(t, x_t, 'LineWidth', 1.5); hold on;
plot(t, y_t, 'LineWidth', 1.5);
xlabel('Time (s)'); ylabel('Position (m)');
title('End-Effector Coordinates vs Time');
grid on; legend('x(t)', 'y(t)', 'Location', 'best');
axis([0 3 0 0.8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig4_coordinates_vs_time.png'), 'Resolution', 300);



%============================================
%% PARAMETER STUDY: cycle times 0.8T, T, 1.2T
% Required (maximum |q1|, maximum |q2|, path length or end-point coordinates.)
 
factors = [0.8, 1.0, 1.2];
colors = {[0.85 0.33 0.10], [0 0.45 0.74], [0.47 0.67 0.19]};
results = struct();
 
fig2 = figure('Position', [100 100 1300 500]);
ax1 = subplot(1,2,1); hold(ax1, 'on');
ax2 = subplot(1,2,2); hold(ax2, 'on');
 
for k = 1:length(factors)
    f = factors(k);
    T_case = f * T;
    t_case = linspace(0, T_case, N);
    validate_motion_inputs(t_case, T_case, L1, L2);
    [q1c, q1dc] = joint_trajectory(t_case, T_case, q1_start, q1_end);
    [q2c, q2dc] = joint_trajectory(t_case, T_case, q2_start, q2_end);
    [xc, yc] = forward_kinematics(q1c, q2c, L1, L2);
 
    dxc = diff(xc); dyc = diff(yc);
    path_len_c = sum(sqrt(dxc.^2 + dyc.^2));
 
    fname = sprintf('f%d', round(f*10));   % f8, f10, f12
    results.(fname).T = T_case;
    results.(fname).max_q1dot = max(abs(q1dc));
    results.(fname).max_q2dot = max(abs(q2dc));
    results.(fname).path_length = path_len_c;
    results.(fname).end_point = [xc(end), yc(end)];
 
    plot(ax1, t_case, q1dc, 'Color', colors{k}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%.1fT  (T=%.2fs)', f, T_case));
    plot(ax2, t_case, q2dc, 'Color', colors{k}, 'LineWidth', 1.5, ...
        'DisplayName', sprintf('%.1fT  (T=%.2fs)', f, T_case));
end
 
xlabel(ax1, 'Time (s)'); ylabel(ax1, 'q1dot (rad/s)');
title(ax1, 'Joint 1 Velocity: Effect of Cycle Time');
xlim(ax1, [0.0 3.5]); ylim(ax1, [0.0 0.9]);
grid(ax1, 'on'); legend(ax1, 'show', 'Location', 'best');
 
xlabel(ax2, 'Time (s)'); ylabel(ax2, 'q2dot (rad/s)');
title(ax2, 'Joint 2 Velocity: Effect of Cycle Time');
axis([0 3.5 0 0.9]);
grid(ax2, 'on'); legend(ax2, 'show', 'Location', 'best');
 
exportgraphics(fig2, fullfile(figures_dir, 'fig5_parameter_study.png'), 'Resolution', 300);

 
%============================================
%% SUMMARY TABLE OF PERFORMANCE MEASURES
fprintf('\n======================================================================\n');
fprintf(' PARAMETER STUDY SUMMARY (cycle times 0.8T, T, 1.2T)\n');
fprintf('======================================================================\n');
fprintf('%-10s%-10s%-14s%-14s%-14s\n', 'Case', 'T (s)', 'max|q1dot|', 'max|q2dot|', 'Path len (m)');
 
fnames = {'f8', 'f10', 'f12'};
labels = {'0.8T', '1.0T', '1.2T'};
for k = 1:length(factors)
    r = results.(fnames{k});
    fprintf('%-10s%-10.3f%-14.4f%-14.4f%-14.4f\n', labels{k}, r.T, r.max_q1dot, r.max_q2dot, r.path_length);
end
 
for k = 1:length(factors)
    r = results.(fnames{k});
    fprintf('  %s -> (%.4f, %.4f) m\n', labels{k}, r.end_point(1), r.end_point(2));
end
 
Case = labels';
T_s = zeros(3,1); max_q1dot = zeros(3,1); max_q2dot = zeros(3,1);
for k = 1:length(factors)
    r = results.(fnames{k});
    T_s(k) = round(r.T, 3);
    max_q1dot(k) = round(r.max_q1dot, 4);
    max_q2dot(k) = round(r.max_q2dot, 4);
end
param_study_table = table(Case, T_s, max_q1dot, max_q2dot, ...
    'VariableNames', {'Case', 'T_s', 'max_q1dot_rad_s', 'max_q2dot_rad_s'});
writetable(param_study_table, fullfile(tables_dir, 'parameter_study_results.csv'));
 
fprintf('\nDone. Plots saved: fig1_workspace_path.png, fig2_joint_angles.png,\n');
fprintf('fig3_joint_velocities.png, fig4_coordinates_vs_time.png, fig5_parameter_study.png\n');
fprintf('Tables saved: results_summary.csv, timeseries_data.csv, parameter_study_results.csv\n');