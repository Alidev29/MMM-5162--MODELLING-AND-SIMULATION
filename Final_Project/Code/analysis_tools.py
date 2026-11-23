# File: analysis_tools.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Pure analysis functions used by scripts 07-10: output statistics (t interval, Wilson
#          interval, required sample size), global sensitivity (PRCC), Pareto filtering, simple
#          search methods and step-response metrics
# Inputs: numpy arrays passed by the calling script
# Outputs: numbers and arrays (no files)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import math
import numpy as np
from scipy import stats

Z_95 = 1.959964


# ---------------------------------------------------------------- output analysis (Topic 5)
def t_mean_ci(values, conf=0.95):
    """Sample mean with a Student-t confidence interval. Returns (mean, low, high, standard error)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n))
    half = float(stats.t.ppf(0.5 + conf / 2.0, n - 1)) * se
    return mean, mean - half, mean + half, se


def wilson_ci(n_success, n, z=Z_95):
    """Wilson score interval for a binomial proportion. Returns (p_hat, low, high)."""
    p_hat = n_success / n
    denom = 1.0 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
    low = 0.0 if n_success == 0 else max(0.0, centre - half)       # exact bounds at the extremes
    high = 1.0 if n_success == n else min(1.0, centre + half)
    return p_hat, low, high


def proportion_se(p_hat, n):
    """Standard error of a binomial proportion."""
    return math.sqrt(p_hat * (1.0 - p_hat) / n)


def required_n(p, half_width, z=Z_95):
    """Smallest N whose normal-approximation 95 % half-width on p does not exceed half_width."""
    return int(math.ceil(z ** 2 * p * (1.0 - p) / half_width ** 2))


# ---------------------------------------------------------------- global sensitivity (Topic 7)
def prcc(X, y):
    """Partial rank correlation coefficient of each column of X with y: rank-transform everything,
    remove the linear effect of the other inputs from both x_i and y, correlate the residuals."""
    R = np.column_stack([stats.rankdata(col) for col in np.asarray(X, dtype=float).T])
    ry = stats.rankdata(y)
    n, k = R.shape
    out = np.empty(k)
    for i in range(k):
        others = np.column_stack([np.ones(n), np.delete(R, i, axis=1)])
        res_x = R[:, i] - others @ np.linalg.lstsq(others, R[:, i], rcond=None)[0]
        res_y = ry - others @ np.linalg.lstsq(others, ry, rcond=None)[0]
        out[i] = np.corrcoef(res_x, res_y)[0, 1]
    return out


# ---------------------------------------------------------------- multi-objective (Topic 8)
def pareto_mask(maximise, minimise):
    """True for the non-dominated points when `maximise` is to be maximised and `minimise` minimised."""
    a = np.asarray(maximise, dtype=float)
    b = np.asarray(minimise, dtype=float)
    mask = np.ones(len(a), dtype=bool)
    for i in range(len(a)):
        dominated = (a >= a[i]) & (b <= b[i]) & ((a > a[i]) | (b < b[i]))
        mask[i] = not dominated.any()
    return mask


# ---------------------------------------------------------------- search methods (Topic 8)
def random_points(bounds, n, rng):
    """n points drawn uniformly inside the box bounds = [(low, high), ...]."""
    low = np.array([b[0] for b in bounds])
    high = np.array([b[1] for b in bounds])
    return low + rng.random((n, len(bounds))) * (high - low)


def local_search(evaluate, start, steps, bounds, max_iter=100):
    """Steepest-ascent local search on a lattice. evaluate(point) -> (J, feasible).
    From the current point, try +/- one step in each coordinate (inside the bounds), move to the best
    feasible neighbour that improves J, stop when none does. Returns (path, n_evaluations), where path
    is a list of (point, J) starting at `start`."""
    current = tuple(float(v) for v in start)
    j_cur, _ = evaluate(current)
    path = [(current, j_cur)]
    n_eval = 1
    for _ in range(max_iter):
        best = None
        for d, step in enumerate(steps):
            for sign in (-1.0, 1.0):
                cand = list(current)
                cand[d] = round(cand[d] + sign * step, 9)
                if not bounds[d][0] - 1e-12 <= cand[d] <= bounds[d][1] + 1e-12:
                    continue
                j, feasible = evaluate(tuple(cand))
                n_eval += 1
                if feasible and j > j_cur and (best is None or j > best[1]):
                    best = (tuple(cand), j)
        if best is None:
            break
        current, j_cur = best
        path.append(best)
    return path, n_eval


# ---------------------------------------------------------------- step response (Topic 3)
def step_metrics(t, y, y0, y_final, band=0.02):
    """Rise time (10-90 %), percent overshoot and settling time (last exit from the +/-band band
    around y_final, as a fraction of the step) of a step response y(t) from y0 to y_final."""
    t = np.asarray(t, dtype=float)
    step = y_final - y0
    frac = (np.asarray(y, dtype=float) - y0) / step
    t10 = float(np.interp(0.1, *_first_rise(t, frac, 0.1)))
    t90 = float(np.interp(0.9, *_first_rise(t, frac, 0.9)))
    overshoot = max(0.0, float(frac.max() - 1.0)) * 100.0
    outside = np.nonzero(np.abs(frac - 1.0) > band)[0]
    if len(outside) == 0:
        t_settle = float(t[0])
    elif outside[-1] == len(t) - 1:
        t_settle = float('nan')                  # never settled inside the window
    else:
        t_settle = float(t[outside[-1] + 1])
    return {'rise_time_s': t90 - t10, 'overshoot_pct': overshoot, 'settling_time_s': t_settle,
            't10_s': t10, 't90_s': t90}


def _first_rise(t, frac, level):
    """The two samples around the first upward crossing of `level` (for linear interpolation)."""
    idx = int(np.argmax(frac >= level))
    if idx == 0:
        return frac[:1], t[:1]
    return frac[idx - 1:idx + 1], t[idx - 1:idx + 1]


# ---------------------------------------------------------------- loop margins (Topic 3)
def frequency_response(A, B, C, w):
    """G(jw) = C (jwI - A)^-1 B of a single-input single-output state-space model, for each w (rad/s)."""
    n = A.shape[0]
    mats = 1j * np.asarray(w)[:, None, None] * np.eye(n)[None] - A[None]
    return np.linalg.solve(mats, np.broadcast_to(B, (len(w), n))[..., None])[..., 0] @ C


def loop_margins(loop, w):
    """Gain crossover, phase margin and gain margin of an open-loop frequency response `loop` on the
    ascending grid w (rad/s). The phase is unwrapped from the lowest frequency; crossings are refined by
    linear interpolation in log w. Returns dict with w_c (rad/s), phase_margin_deg (at the first gain
    crossover), w_180 (rad/s) and gain_margin_db (smallest over the -180 deg crossings; inf if none)."""
    w = np.asarray(w, float)
    mag = np.abs(loop)
    phase = np.degrees(np.unwrap(np.angle(loop)))
    lw = np.log10(w)
    out = {'w_c': float('nan'), 'phase_margin_deg': float('nan'), 'w_180': float('nan'),
           'gain_margin_db': float('inf')}
    lm = np.log10(mag)
    idx = np.nonzero((lm[:-1] > 0) & (lm[1:] <= 0))[0]
    if len(idx):
        k = idx[0]
        f = lm[k] / (lm[k] - lm[k + 1])
        out['w_c'] = float(10 ** (lw[k] + f * (lw[k + 1] - lw[k])))
        out['phase_margin_deg'] = float(180.0 + phase[k] + f * (phase[k + 1] - phase[k]))
    p = phase + 180.0
    for k in np.nonzero(np.sign(p[:-1]) != np.sign(p[1:]))[0]:
        f = p[k] / (p[k] - p[k + 1])
        gm = -20.0 * (lm[k] + f * (lm[k + 1] - lm[k]))
        if gm < out['gain_margin_db']:
            out['gain_margin_db'] = float(gm)
            out['w_180'] = float(10 ** (lw[k] + f * (lw[k + 1] - lw[k])))
    return out
