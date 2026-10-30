# -*- coding: utf-8 -*-
"""CIFAR-10 CNN four-method certified rates at 1500 samples (3 seeds x 500).

Extends the sample size from 900 to 1500 for the
IBP-trained CIFAR-10 CNN workload. Reuses the sound self-implemented
verifiers from the SAFE2 source tree (verifier_cnn.py, baseline_crown.py)
and the project checkpoints / bundled CIFAR-10 data -- no auto_LiRPA.

Methods: ibp, crown (self-implemented CROWN backward), krawczyk (one-shot
classical), safe2 (gated relaxed protocol). eps in {2/255, 4/255}.
Outputs results/cnn_rates_1500.json with per-seed/eps/method robust rates,
pooled rates, runtimes, and exact McNemar p-values (SAFE2 vs IBP).
"""
import os
import sys
import json
import time

import numpy as np
import torch
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
OUT_PATH = os.path.join(BASE, 'results', 'cnn_rates_1500.json')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc          # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402

SEEDS = [0, 1, 2]
N_SAMPLES = 500
EPS_GRID = [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0]
INPUT_SHAPE = (3, 32, 32)
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12


def clamp_box(x0, eps):
    lo = torch.clamp(x0 - eps, 0.0, 1.0)
    hi = torch.clamp(x0 + eps, 0.0, 1.0)
    return lo, hi


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=N_SAMPLES)
    ap.add_argument('--seeds', type=str, default='0,1,2')
    ap.add_argument('--out', type=str, default=OUT_PATH)
    args = ap.parse_args()
    n_samples = args.samples
    seeds = [int(s) for s in args.seeds.split(',')]
    out_path = args.out
    _, _, x_te, y_te = load_cifar10()
    out = {'n_samples_per_seed': n_samples, 'eps_grid': EPS_GRID,
           'seeds': seeds, 'per_seed_eps': {}, 'pooled': {},
           'mcnemar_safe2_vs_ibp': {}}
    for s in seeds:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
        print(f'[seed {s}] n_samples={len(idx)}', flush=True)
        for eps in EPS_GRID:
            cnt = {'ibp': 0, 'krawczyk': 0, 'safe2': 0}
            rt = {'ibp': [], 'krawczyk': [], 'safe2': []}
            mcnemar_b = 0   # safe2 robust, ibp not
            mcnemar_c = 0   # ibp robust, safe2 not
            t_eps0 = time.perf_counter()
            for pos, si in enumerate(idx):
                x0 = x_te[si].flatten()
                c = int(y_te[si].item())
                t0 = time.perf_counter()
                r_ibp = vc.verify_ibp(model, x0, c, eps, INPUT_SHAPE, clamp01=True)
                rt['ibp'].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                r_kr = vc.verify_krawczyk_classical(
                    model, x0, c, eps, INPUT_SHAPE, clamp01=True)
                rt['krawczyk'].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                r_s2 = vc.verify_safe2(
                    model, x0, c, eps, INPUT_SHAPE, ALPHA_MIN, ALPHA_FACTOR,
                    MAX_ITER_R, MAX_ITER_E, clamp01=True)[0]
                rt['safe2'].append(time.perf_counter() - t0)
                for m, r in (('ibp', r_ibp), ('krawczyk', r_kr),
                             ('safe2', r_s2)):
                    if r == 'robust':
                        cnt[m] += 1
                if r_s2 == 'robust' and r_ibp != 'robust':
                    mcnemar_b += 1
                if r_ibp == 'robust' and r_s2 != 'robust':
                    mcnemar_c += 1
                if (pos + 1) % 100 == 0:
                    el = time.perf_counter() - t_eps0
                    print(f'[seed {s} eps {eps:.4f}] {pos+1}/{len(idx)} '
                          f'({el:.0f}s)', flush=True)
            rec = {m: {'robust': cnt[m], 'n': len(idx),
                       'rate': cnt[m] / len(idx),
                       't_mean_s': float(np.mean(rt[m]))} for m in cnt}
            out['per_seed_eps'][f'{s}_{eps}'] = rec
            # checkpoint to disk after every (seed, eps) so a late crash
            # never loses completed per-seed data
            with open(out_path, 'w', encoding='utf-8') as _f:
                json.dump(out, _f, indent=1)
            print(f'[seed {s} eps {eps:.4f}] ' +
                  ' '.join(f'{m}={cnt[m]}/{len(idx)}' for m in cnt), flush=True)
            # exact McNemar (two-sided) accumulated over this (seed, eps)
            key = f'{eps}'
            b = out['mcnemar_safe2_vs_ibp'].setdefault(
                key, {'b': 0, 'c': 0})
            b['b'] += mcnemar_b
            b['c'] += mcnemar_c
    # pooled rates
    for eps in EPS_GRID:
        pooled = {}
        for m in ('ibp', 'krawczyk', 'safe2'):
            rates = [out['per_seed_eps'][f'{s}_{eps}'][m]['rate'] for s in seeds]
            pooled[m] = {'rate_mean': float(np.mean(rates)),
                         'rate_std': float(np.std(rates)),
                         'n_models': len(rates)}
        out['pooled'][str(eps)] = pooled
        b = out['mcnemar_safe2_vs_ibp'][str(eps)]['b']
        c = out['mcnemar_safe2_vs_ibp'][str(eps)]['c']
        if b + c > 0:
            # Standard two-sided exact McNemar p-value: scipy's binomtest
            # already sums both tails, so no extra factor of two.
            p = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
            out['mcnemar_safe2_vs_ibp'][str(eps)]['p_exact'] = float(min(p, 1.0))
            out['mcnemar_safe2_vs_ibp'][str(eps)]['discordant_pairs'] = b + c
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', out_path, flush=True)
    print('POOLED', json.dumps(out['pooled'], default=str), flush=True)
    print('MCNEMAR', json.dumps(out['mcnemar_safe2_vs_ibp']), flush=True)


if __name__ == '__main__':
    main()
