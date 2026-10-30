# -*- coding: utf-8 -*-
"""
Self-implemented CROWN-IBP rerun on the SAME 600-sample MNIST certification
set as the IBP/Krawczyk/SAFE2 columns of Table tab:rates (3 seeds x 200 x 6
radii), using baseline_crown.crown_bounds_logits with refine=True
(RefineIntermediateBounds: per-ReLU pre-activation bounds tightened by a
CROWN backward from the source affine output, then merged with the IBP
bounds -- the standard CROWN-IBP refinement used by auto-LiRPA). Unifies
the CROWN-IBP column across the MNIST and CIFAR tables.
"""
import os
import sys
import json
import time

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from baseline_crown import crown_bounds_logits      # noqa: E402
from train import MLP, load_mnist                   # noqa: E402

SEEDS = [0, 1, 2]
N_SAMPLES = 200
EPS_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def crown_margin_clamped(model, x0, c, eps):
    # MNIST box: NO clamp to [0,1], matching verifier_lib.verify_safe2 and the
    # auto-LiRPA reference (run_baselines_lirpa.py uses PerturbationLpNorm on
    # the raw x0 for MNIST). Clamping would shrink the box and inflate the rate.
    lo = x0 - eps
    hi = x0 + eps
    lo_b, hi_b = crown_bounds_logits(model, lo, hi, (1, 28, 28), refine=True)
    others = [j for j in range(lo_b.shape[0]) if j != c]
    return float((lo_b[c] - torch.max(hi_b[others])).item())


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    out = {'n_per_seed': N_SAMPLES, 'eps_grid': EPS_GRID,
           'per_seed_eps': {}, 'pooled': {}}
    for s in SEEDS:
        model = MLP([784, 64, 64, 10])
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'mlp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        print(f'[seed {s}] n={len(idx)}', flush=True)
        for eps in EPS_GRID:
            robust = 0
            t0 = time.perf_counter()
            for si in idx:
                x0 = x_te[si].flatten()
                c = int(y_te[si].item())
                if crown_margin_clamped(model, x0, c, eps) > 0:
                    robust += 1
            rate = robust / len(idx)
            out['per_seed_eps'][f'{s}_{eps}'] = {'robust': robust,
                                                 'n': len(idx), 'rate': rate}
            print(f'[seed {s} eps {eps}] robust={robust}/{len(idx)} '
                  f'rate={rate:.4f} ({time.perf_counter()-t0:.0f}s)',
                  flush=True)
    import statistics
    for eps in EPS_GRID:
        rates = [out['per_seed_eps'][f'{s}_{eps}']['rate'] for s in SEEDS]
        out['pooled'][str(eps)] = {'rate_mean': float(sum(rates) / len(rates)),
                                   'rate_std': float(statistics.stdev(rates))}
        print(f'POOLED {eps} mean={out["pooled"][str(eps)]["rate_mean"]:.4f} '
              f'std={out["pooled"][str(eps)]["rate_std"]:.4f}', flush=True)
    with open(os.path.join(BASE, 'results', 'crown_ibp_mnist_600.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED results/crown_ibp_mnist_600.json (refined CROWN-IBP)', flush=True)


if __name__ == '__main__':
    main()
