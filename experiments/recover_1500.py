# -*- coding: utf-8 -*-
"""Recover results/cnn_rates_1500.json from the run log of cnn_rates_1500.py.

The original run completed all 3 seeds x 4 eps but crashed in the pooled
stage (stale 'crown' key) before writing JSON. This script:
  1. parses _cnn1500_run.log for per-(seed, eps) counts -> per_seed_eps
  2. rebuilds pooled rates (mean/std over seeds)
  3. recomputes exact McNemar discordant pairs (SAFE2 vs IBP) with a light
     rerun restricted to ibp+safe2 at eps in {2/255, 4/255} only, on the
     same 500 correctly-classified samples per seed (deterministic index
     selection identical to cnn_rates_1500.py), then writes the merged JSON.
"""
import os
import re
import sys
import json
import time

import numpy as np
import torch
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
LOG_PATH = os.path.join(BASE, 'results', '_cnn1500_run.log')
OUT_PATH = os.path.join(BASE, 'results', 'cnn_rates_1500.json')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc                    # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402

INPUT_SHAPE = (3, 32, 32)
EPS_GRID = [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0]
MCNEMAR_EPS = [2.0 / 255.0, 4.0 / 255.0]
SEEDS = [0, 1, 2]
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12
N_SAMPLES = 500

LINE_RE = re.compile(
    r'\[seed (\d) eps (\d+\.\d+)\] ibp=(\d+)/\d+ krawczyk=(\d+)/\d+ safe2=(\d+)/\d+')


def parse_log():
    per_seed_eps = {}
    with open(LOG_PATH, encoding='utf-8') as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            s, eps_str = int(m.group(1)), float(m.group(2))
            ibp, kraw, safe2 = int(m.group(3)), int(m.group(4)), int(m.group(5))
            n = N_SAMPLES
            rec = {'ibp': {'robust': ibp, 'n': n, 'rate': ibp / n},
                   'krawczyk': {'robust': kraw, 'n': n, 'rate': kraw / n},
                   'safe2': {'robust': safe2, 'n': n, 'rate': safe2 / n}}
            per_seed_eps[f'{s}_{eps_str}'] = rec
    return per_seed_eps


def pooled_from(per_seed_eps):
    pooled = {}
    for eps in EPS_GRID:
        e = f'{eps:.4f}'
        pooled[e] = {}
        for m in ('ibp', 'krawczyk', 'safe2'):
            rates = [per_seed_eps[f'{s}_{e}'][m]['rate'] for s in SEEDS]
            pooled[e][m] = {'rate_mean': float(np.mean(rates)),
                            'rate_std': float(np.std(rates)),
                            'n_models': len(rates)}
    return pooled


def run_mcnemar():
    """Light rerun: ibp+safe2 only, eps in MCNEMAR_EPS, 3 seeds x 500 samples."""
    _, _, x_te, y_te = load_cifar10()
    mcn = {}
    for s in SEEDS:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        print(f'[seed {s}] n_samples={len(idx)}', flush=True)
        for eps in MCNEMAR_EPS:
            b = c = 0
            for si in idx:
                x0 = x_te[si].flatten()
                cc = int(y_te[si].item())
                r_ibp = vc.verify_ibp(model, x0, cc, eps, INPUT_SHAPE,
                                      clamp01=True)
                r_s2 = vc.verify_safe2(
                    model, x0, cc, eps, INPUT_SHAPE, ALPHA_MIN, ALPHA_FACTOR,
                    MAX_ITER_R, MAX_ITER_E, clamp01=True)[0]
                if r_s2 == 'robust' and r_ibp != 'robust':
                    b += 1
                if r_ibp == 'robust' and r_s2 != 'robust':
                    c += 1
            key = f'{eps:.4f}'
            mcn.setdefault(key, {'b': 0, 'c': 0})
            mcn[key]['b'] += b
            mcn[key]['c'] += c
            print(f'[seed {s} eps {eps:.4f}] b={b} c={c}', flush=True)
    for eps in MCNEMAR_EPS:
        key = f'{eps:.4f}'
        b, c = mcn[key]['b'], mcn[key]['c']
        if b + c > 0:
            p = 2.0 * stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
            mcn[key]['p_exact'] = float(min(p, 1.0))
            mcn[key]['discordant_pairs'] = b + c
        print(f'[mcnemar eps {eps:.4f}] {json.dumps(mcn[key])}', flush=True)
    return mcn


def main():
    per_seed_eps = parse_log()
    print('parsed', len(per_seed_eps), 'per_seed_eps entries', flush=True)
    pooled = pooled_from(per_seed_eps)
    print('POOLED', json.dumps(pooled, default=str), flush=True)
    mcn = run_mcnemar()
    out = {'n_samples_per_seed': N_SAMPLES, 'eps_grid': EPS_GRID,
           'seeds': SEEDS, 'per_seed_eps': per_seed_eps,
           'pooled': pooled, 'mcnemar_safe2_vs_ibp': mcn,
           'recovered_from_log': True}
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
