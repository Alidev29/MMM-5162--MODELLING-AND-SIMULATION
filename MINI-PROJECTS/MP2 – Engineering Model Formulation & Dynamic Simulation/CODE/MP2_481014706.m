%% File: MP2_481014706.m
% Course: MMM 5162 Modelling and Simulation
% Author: Ali Khuzam
% Date: 27/09/2026
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
%% Parameters (S = 8)
S = 8;
drone.m = 1.60 + 0.03*S; %Mass in (kg)
drone.c = 0.120 + 0.005*S; 
drone.tau_T = 0.160 + 0.005*S; %time constant in (s)
drone.r = 1.180 + 0.005*S;

sys.g = 9.81; %Acceleration in (m/s^2)
sys.pulse_start = 1.0;              % s
sys.pulse_duration = 1.40 + 0.03*S; % s
sys.mg = drone.m*sys.g; %Force in (N)

t0 = 0.0; tf = 8.0; % simulation interval in (s)
x0 = [0.0; 0.0; sys.mg];% initial state z,v,T
%============================================
%% FUNCTIONS
function Tc = Tcmd(t, drone, sys)
    if t >= sys.pulse_start && t < sys.pulse_start + sys.pulse_duration
        Tc = drone.r*sys.mg;
    else
        Tc = sys.mg;
    end
end

function dxdt = drone_ode(t, x, drone, sys)
    z = x(1); v = x(2); T = x(3); 
    dz = v;
    dv = (T - sys.mg - drone.c*v*abs(v)) / drone.m;
    dT = (Tcmd(t, drone, sys) - T) / drone.tau_T;
    dxdt = [dz; dv; dT];
end

function [t, x] = forward_euler(f, x0, t0, tf, dt)
% f for function
    n = round((tf - t0)/dt);
    t = linspace(t0, t0 + n*dt, n+1)'; % U can use t = linspace(t0, tf, n+1)'; but to avoid small mathmatical errors
    x = zeros(n+1, length(x0));
    x(1,:) = x0';
    for k = 1:n
        dx = f(t(k), x(k,:)');
        x(k+1,:) = x(k,:) + dt*dx';
    end
end
%==========================================================================
%% State equations 
odefun = @(t, x) drone_ode(t, x, drone, sys);
%% Standard ODE solver 
options = odeset('RelTol', 1e-9, 'AbsTol', 1e-11);
[t_ref, x_ref] = ode45(odefun, [t0 tf], x0, options);
%U can simplify it like :[t_ref, x_ref] = ode45(odefun, [t0 tf], x0); but
%for very small error to use it as reference 
z_ref = x_ref(:,1);
v_ref = x_ref(:,2);
T_ref = x_ref(:,3);

[v_max_ref, v_index] = max(v_ref);
t_vmax_ref = t_ref(v_index);
z_final_ref = z_ref(end);
%==========================================================================
%% Baseline Plotting

fprintf('\n[Standard solver - ode45, reference]\n');
fprintf(' Max climb velocity : %.4f m/s at t = %.3f s\n', v_max_ref, t_vmax_ref);
fprintf(' Altitude at t=8 s  : %.4f m\n', z_final_ref);

figure('Position', [100 100 700 500]);
plot(t_ref, z_ref, 'b-', 'LineWidth', 1.8);
xlabel('Time (s)'); ylabel('Altitude z (m)');
title('Drone Altitude vs Time (Standard Solver, ode45)');
grid on; xlim([0 8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig1_altitude_response.png'), 'Resolution', 300);

figure('Position', [100 100 700 500]);
plot(t_ref, v_ref, 'b-', 'LineWidth', 1.8);
xlabel('Time (s)'); ylabel('Vertical velocity v (m/s)');
title('Drone Vertical Velocity vs Time (Standard Solver, ode45)');
grid on; xlim([0 8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig2_velocity_response.png'), 'Resolution', 300);

figure('Position', [100 100 700 500]);
plot(t_ref, T_ref, 'b-', 'LineWidth', 1.8); hold on;
yline(sys.mg, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xlabel('Time (s)'); ylabel('Thrust T (N)');
title('Actuator Thrust vs Time (Standard Solver, ode45)');
legend('T(t) actual thrust', 'mg (hover thrust)', 'Location', 'best');
grid on; xlim([0 8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig3_thrust_response.png'), 'Resolution', 300);

%==========================================================================
%% Forward Euler implementation and comparison at multiple time steps
dt_list = [0.1, 0.05, 0.01];   
n_dt = length(dt_list);
euler_t = cell(1, n_dt);
euler_z = cell(1, n_dt);
euler_v = cell(1, n_dt);
% MAKE EMPTY MATRIX CELLS 
euler_vmax = zeros(1, n_dt);
euler_zfinal = zeros(1, n_dt);

for k = 1:n_dt
    dt = dt_list(k);
    [t_e, x_e] = forward_euler(odefun, x0, t0, tf, dt);
    z_e = x_e(:,1);
    v_e = x_e(:,2);
    euler_t{k} = t_e;
    euler_z{k} = z_e;
    euler_v{k} = v_e;
    euler_vmax(k) = max(v_e);
    euler_zfinal(k) = z_e(end);
    fprintf('\n[Forward Euler, dt = %.3f s]\n', dt);
    fprintf(' Max climb velocity : %.4f m/s\n', euler_vmax(k));
    fprintf(' Altitude at t=8 s  : %.4f m\n', euler_zfinal(k));
end

% --- standard-solver vs forward-Euler comparison plots ---
colors = {[0.85 0.33 0.10], [0.47 0.67 0.19], [0.64 0.08 0.18]};

figure('Position', [100 100 750 550]);
plot(t_ref, v_ref, 'b-', 'LineWidth', 2.2, 'DisplayName', 'ode45 (reference)'); hold on;
for k = 1:n_dt
    plot(euler_t{k}, euler_v{k}, '--', 'Color', colors{k}, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('Euler dt=%.3fs', dt_list(k)));
end
xlabel('Time (s)'); ylabel('Vertical velocity v (m/s)');
title('Standard Solver vs Forward Euler: Velocity Response');
grid on; legend('show', 'Location', 'best'); xlim([0 8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig4_velocity_comparison.png'), 'Resolution', 300);

figure('Position', [100 100 750 550]);
plot(t_ref, z_ref, 'b-', 'LineWidth', 2.2, 'DisplayName', 'ode45 (reference)'); hold on;
for k = 1:n_dt
    plot(euler_t{k}, euler_z{k}, '--', 'Color', colors{k}, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('Euler dt=%.2fs', dt_list(k)));
end
xlabel('Time (s)'); ylabel('Altitude z (m)');
title('Standard Solver vs Forward Euler: Altitude Response');
grid on; legend('show', 'Location', 'best'); xlim([0 8]);
exportgraphics(gcf, fullfile(figures_dir, 'fig5_altitude_comparison.png'), 'Resolution', 300);


%==========================================================================
% Numerical verification 
Method = cell(n_dt+1, 1);
v_max_col = zeros(n_dt+1, 1);
z8_col = zeros(n_dt+1, 1);
err_vmax_col = zeros(n_dt+1, 1);
err_z8_col = zeros(n_dt+1, 1);

for k = 1:n_dt
    Method{k} = sprintf('Forward Euler (dt=%.3fs)', dt_list(k));
    v_max_col(k) = round(euler_vmax(k), 4);
    z8_col(k) = round(euler_zfinal(k), 4);
    err_vmax_col(k) = round(100*abs(euler_vmax(k)-v_max_ref)/abs(v_max_ref), 4);
    err_z8_col(k) = round(100*abs(euler_zfinal(k)-z_final_ref)/abs(z_final_ref), 4);
end
Method{n_dt+1} = 'ode45 (reference)';
v_max_col(n_dt+1) = round(v_max_ref, 4);
z8_col(n_dt+1) = round(z_final_ref, 4);
err_vmax_col(n_dt+1) = 0.0;
err_z8_col(n_dt+1) = 0.0;

verification_table = table(Method, v_max_col, z8_col, err_vmax_col, err_z8_col, ...
    'VariableNames', {'Method', 'v_max_m_s', 'z8_m', 'RelErr_vmax_pct', 'RelErr_z8_pct'});
writetable(verification_table, fullfile(tables_dir, 'numerical_verification.csv'));

fprintf('\n======================================================================\n');
fprintf(' NUMERICAL VERIFICATION TABLE (Euler vs ode45 reference)\n');
fprintf('======================================================================\n');
disp(verification_table);

% quantitative resolution check on the finest Euler step 
[finest_dt, idx_finest] = min(dt_list);
err_vmax_finest = err_vmax_col(idx_finest);
err_zfin_finest = err_z8_col(idx_finest);
TOL_PCT = 1.0;   % engineering tolerance for "sufficiently resolved"

fprintf('\n=======================================================\n');
fprintf(' RESOLUTION STATEMENT\n');
fprintf('=======================================================\n');
if err_vmax_finest < TOL_PCT && err_zfin_finest < TOL_PCT
    fprintf([' The finest Euler step (dt=%.3fs) differs from the ode45 reference by ' ...
        '%.4f%% (v_max) and %.4f%% (z(8)), both below the %.1f%% engineering tolerance. ' ...
        'The reported engineering result (v_max = %.4f m/s, z(8) = %.4f m) is therefore ' ...
        'judged to be sufficiently resolved and not an artifact of the numerical settings.\n'], ...
        finest_dt, err_vmax_finest, err_zfin_finest, TOL_PCT, v_max_ref, z_final_ref);
else
    fprintf([' The finest Euler step (dt=%.3fs) differs from the ode45 reference by ' ...
        '%.4f%% (v_max) and %.4f%% (z(8)), which exceeds the %.1f%% engineering tolerance. ' ...
        'A finer time step or the ode45 solution alone should be used as the reported result.\n'], ...
        finest_dt, err_vmax_finest, err_zfin_finest, TOL_PCT);
end

%==========================================================================
%% Results summary table
Metric = {'Max climb velocity (m/s)'; 'Time of max velocity (s)'; 'Altitude at t=8s (m)'; ...
          'm (kg)'; 'c'; 'tau_T (s)'; 'r'; 'pulse duration (s)'};
Value = [round(v_max_ref,4); round(t_vmax_ref,4); round(z_final_ref,4); drone.m; drone.c; drone.tau_T; drone.r; sys.pulse_duration];
summary_table = table(Metric, Value);
writetable(summary_table, fullfile(tables_dir, 'results_summary.csv'));

timeseries_table = table(t_ref, z_ref, v_ref, T_ref, 'VariableNames', {'t','z','v','T'});
writetable(timeseries_table, fullfile(tables_dir, 'timeseries_data.csv'));

fprintf('\nDone. Plots saved\n');
fprintf('Tables saved\n');


