# -*- coding: utf-8 -*-
"""
Scaling study with goodness-of-fit (merged paper, Section 7.5).

Re-runs the structured O(d) vs dense O(d^2) interval-Krawczyk image timing
over an extended dimension grid and reports, for each implementation, the
log-log slope AND the R^2 of the linear fit, so that the "effectively
linear" claim of the structured form can be judged by the fit quality
(rather than by the slope alone).

Self-contained layout (see joint_protocol.py):
  <project>/experiments/scaling_r2.py
  <project>/experiments/vendor/
  <project>/results/
"""
import os
import sys
import json
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)

import verifier_lib as vl  # noqa: E402
import verifier_cnn as vc  # noqa: E402

OUT_DIR = os.path.join(BASE, '..', 'results')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

# extended grid: 2^6 .. 2^14, overlapping the paper's 2^8..2^12 window
DS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
REPEATS = {64: 30, 128: 30, 256: 30, 512: 25, 1024: 20,
           2048: 15, 4096: 10, 8192: 6, 16384: 3}


def measure_image(d):
    X_lo = -torch.rand(d) * 0.5
    X_hi = torch.rand(d) * 0.5
    m = (X_lo + X_hi) / 2.0
    g_m = 0.3
    J_lo = -torch.rand(d)
    J_hi = torch.rand(d)
    J_mid = (J_lo + J_hi) / 2.0
    C = vl.moore_penrose(J_mid)
    reps = REPEATS[d]
    t0 = time.perf_counter()
    for _ in range(reps):
        vl.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    t_dense = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        vc.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    t_struct = (time.perf_counter() - t0) / reps
    return t_struct, t_dense


def fit(logd, logt):
    """OLS slope + R^2 on log-log data."""
    A = np.vstack([logd, np.ones_like(logd)]).T
    coef, _, _, _ = np.linalg.lstsq(A, logt, rcond=None)
    slope, intercept = coef
    pred = slope * logd + intercept
    ss_res = float(np.sum((logt - pred) ** 2))
    ss_tot = float(np.sum((logt - np.mean(logt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def main():
    rows = []
    for d in DS:
        t_s, t_d = measure_image(d)
        print(f'd={d}: struct={t_s*1e3:.4f}ms dense={t_d*1e3:.4f}ms', flush=True)
        rows.append({'d': d, 't_struct_s': t_s, 't_dense_s': t_d})

    logd_all = np.log([r['d'] for r in rows])
    lt_s = np.log([r['t_struct_s'] for r in rows])
    lt_d = np.log([r['t_dense_s'] for r in rows])
    s_s, _, r2_s = fit(logd_all, lt_s)
    s_d, _, r2_d = fit(logd_all, lt_d)

    # paper window d in 2^8 .. 2^12
    win = [r for r in rows if 256 <= r['d'] <= 4096]
    logd_win = np.log([r['d'] for r in win])
    lt_s_win = np.log([r['t_struct_s'] for r in win])
    lt_d_win = np.log([r['t_dense_s'] for r in win])
    ws_s, _, wr2_s = fit(logd_win, lt_s_win)
    ws_d, _, wr2_d = fit(logd_win, lt_d_win)

    summary = {
        'slope_struct_all': s_s, 'r2_struct_all': r2_s,
        'slope_dense_all': s_d, 'r2_dense_all': r2_d,
        'slope_struct_2_8_2_12': ws_s, 'r2_struct_2_8_2_12': wr2_s,
        'slope_dense_2_8_2_12': ws_d, 'r2_dense_2_8_2_12': wr2_d,
        'rows': rows,
    }
    fn = os.path.join(OUT_DIR, 'scaling_r2.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print('SUMMARY:', json.dumps(
        {k: v for k, v in summary.items() if k != 'rows'}, indent=1),
        flush=True)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    main()
