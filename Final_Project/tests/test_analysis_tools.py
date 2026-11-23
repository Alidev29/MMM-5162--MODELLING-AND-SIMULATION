# File: test_analysis_tools.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Tests of the pure analysis functions used by scripts 07-10 (known answers and properties).
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18
import math
import numpy as np
import pytest
from scipy import stats
from analysis_tools import (t_mean_ci, wilson_ci, proportion_se, required_n, prcc, pareto_mask,
                            random_points, local_search, step_metrics, frequency_response, loop_margins)


def test_t_mean_ci_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi, se = t_mean_ci(x)
    assert mean == 3.0
    assert se == pytest.approx(math.sqrt(2.5 / 5))
    half = stats.t.ppf(0.975, 4) * se            # 2.776 * 0.7071
    assert hi - mean == pytest.approx(half)
    assert mean - lo == pytest.approx(half)


def test_wilson_ci_known_values():
    p, lo, hi = wilson_ci(29, 1000)               # values of the report (2.9 % [2.0, 4.1] %)
    assert p == 0.029
    assert lo == pytest.approx(0.020266, abs=1e-6)     # same as results/tables/05_risk_convergence.csv
    assert hi == pytest.approx(0.041338, abs=1e-6)
    p0, lo0, hi0 = wilson_ci(0, 1000)
    assert p0 == 0.0 and lo0 == 0.0 and 0.0 < hi0 < 0.005


def test_proportion_se_and_required_n():
    assert proportion_se(0.5, 100) == pytest.approx(0.05)
    assert required_n(0.029, 0.005) == 4327
    assert required_n(0.029, 0.01) == 1082
    # the chosen N really meets the half-width, N - 1 does not
    n = required_n(0.2, 0.01)
    assert 1.959964 * math.sqrt(0.2 * 0.8 / n) <= 0.01 < 1.959964 * math.sqrt(0.2 * 0.8 / (n - 1))


def test_prcc_detects_monotone_and_irrelevant_inputs():
    rng = np.random.default_rng(1)
    X = rng.random((2000, 3))
    y = np.exp(3 * X[:, 0]) - 2 * X[:, 1] + 0.01 * rng.standard_normal(2000)   # x2 irrelevant
    r = prcc(X, y)
    assert r[0] > 0.9
    assert r[1] < -0.7                       # monotone but not rank-additive, so below 0.9 in size
    assert abs(r[2]) < 0.1
    # a single monotone (nonlinear) input gives a PRCC of 1
    assert prcc(X, X[:, 1] ** 3)[1] == pytest.approx(1.0)


def test_pareto_mask_simple_case():
    energy = np.array([10.0, 9.0, 8.0, 10.0, 7.0])
    rms = np.array([0.30, 0.20, 0.10, 0.40, 0.15])
    mask = pareto_mask(energy, rms)
    # point 3 is dominated by point 0 (same energy, larger rms); point 4 by point 2
    assert mask.tolist() == [True, True, True, False, False]


def test_random_points_inside_bounds_and_seeded():
    bounds = [(60.0, 1200.0), (0.05, 0.5)]
    a = random_points(bounds, 48, np.random.default_rng(5162))
    b = random_points(bounds, 48, np.random.default_rng(5162))
    assert a.shape == (48, 2)
    assert np.array_equal(a, b)
    assert np.all(a[:, 0] >= 60.0) and np.all(a[:, 0] <= 1200.0)
    assert np.all(a[:, 1] >= 0.05) and np.all(a[:, 1] <= 0.5)


def test_local_search_climbs_to_lattice_maximum():
    # concave objective with maximum at (3, 2); points with x > 4 are infeasible
    evaluate = lambda p: (-(p[0] - 3.0) ** 2 - (p[1] - 2.0) ** 2, p[0] <= 4.0)
    path, n_eval = local_search(evaluate, (0.0, 0.0), (1.0, 1.0), [(0.0, 10.0), (0.0, 10.0)])
    assert path[-1][0] == (3.0, 2.0)
    assert all(path[i + 1][1] > path[i][1] for i in range(len(path) - 1))
    assert n_eval > len(path)


def test_local_search_respects_bounds_and_feasibility():
    evaluate = lambda p: (p[0], p[0] <= 2.0)           # J grows with x but x > 2 is infeasible
    path, _ = local_search(evaluate, (0.0,), (1.0,), [(0.0, 5.0)])
    assert path[-1][0] == (2.0,)


def test_step_metrics_first_order_system():
    tau = 0.5
    t = np.linspace(0.0, 5.0, 50001)
    y = 1.0 - np.exp(-t / tau)
    m = step_metrics(t, y, 0.0, 1.0)
    assert m['rise_time_s'] == pytest.approx(tau * math.log(9.0), rel=1e-3)   # 2.197 tau
    assert m['overshoot_pct'] == 0.0
    assert m['settling_time_s'] == pytest.approx(-tau * math.log(0.02), rel=1e-3)   # 3.91 tau


def test_step_metrics_second_order_overshoot():
    zeta, wn = 0.5, 4.0
    t = np.linspace(0.0, 10.0, 100001)
    wd = wn * math.sqrt(1 - zeta ** 2)
    y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / math.sqrt(1 - zeta ** 2) * np.sin(wd * t))
    m = step_metrics(t, y, 0.0, 1.0)
    assert m['overshoot_pct'] == pytest.approx(100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2)), rel=1e-3)


def test_loop_margins_integrator_lag():
    # L(s) = 1/(s(s+1)): crossover where w*sqrt(w^2+1) = 1, PM = 90 - atan(w_c), no -180 crossing
    w = np.logspace(-3, 3, 20001)
    s = 1j * w
    m = loop_margins(1.0 / (s * (s + 1.0)), w)
    w_c = math.sqrt((math.sqrt(5.0) - 1.0) / 2.0)
    assert m['w_c'] == pytest.approx(w_c, rel=1e-3)
    assert m['phase_margin_deg'] == pytest.approx(90.0 - math.degrees(math.atan(w_c)), abs=0.05)
    assert m['gain_margin_db'] == math.inf


def test_loop_margins_third_order_gain_margin():
    # L(s) = 4/(s+1)^3: phase -180 deg at w = sqrt(3), |L| = 4/8 there, so GM = 20 log10(2) dB
    w = np.logspace(-3, 3, 20001)
    m = loop_margins(4.0 / (1j * w + 1.0) ** 3, w)
    assert m['w_180'] == pytest.approx(math.sqrt(3.0), rel=1e-3)
    assert m['gain_margin_db'] == pytest.approx(20.0 * math.log10(2.0), abs=0.01)


def test_frequency_response_first_order():
    A, B, C = np.array([[-2.0]]), np.array([1.0]), np.array([1.0])
    w = np.array([0.0, 2.0])
    g = frequency_response(A, B, C, w)
    assert np.allclose(g, [0.5, 1.0 / (2.0 + 2.0j)])
