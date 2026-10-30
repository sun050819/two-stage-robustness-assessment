# -*- coding: utf-8 -*-
"""CROWN-IBP self-implemented rerun on the SAME 1500-sample certification
set as the SAFE2/IBP/Krawczyk columns (3 seeds x 500 x 4 radii).

Uses baseline_crown.crown_bounds_logits with refine=True
(RefineIntermediateBounds: per-ReLU pre-activation bounds tightened by a
CROWN backward from the source affine output and merged with the IBP
bounds, the standard CROWN-IBP refinement used by auto-LiRPA). The
900-set alignment vs. the historical auto-LiRPA values
(results/crown_ibp_align_900.json) validates the implementation before
this same-set run.
"""
import os
import sys
import json
import time
import argparse

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from baseline_crown import crown_bounds_logits      # noqa: E402
from train_cnn import make_cnn, load_cifar10        # noqa: E402

INPUT_SHAPE = (3, 32, 32)
SEEDS = [0, 1, 2]
N_SAMPLES = 500
EPS_GRID = [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0]
EPS_LABEL = {1.0 / 255.0: '1/255', 2.0 / 255.0: '2/255',
             4.0 / 255.0: '4/255', 8.0 / 255.0: '8/255'}


def crown_margin_clamped(model, x0, c, eps):
    lo = torch.clamp(x0 - eps, 0.0, 1.0)
    hi = torch.clamp(x0 + eps, 0.0, 1.0)
    lo_b, hi_b = crown_bounds_logits(model, lo, hi, INPUT_SHAPE, refine=True)
    others = [j for j in range(lo_b.shape[0]) if j != c]
    return float((lo_b[c] - torch.max(hi_b[others])).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    out_path = os.path.join(BASE, 'results',
                            args.out or 'crown_ibp_1500.json')
    _, _, x_te, y_te = load_cifar10()
    out = {'n_per_seed': N_SAMPLES, 'eps_grid': [EPS_LABEL[e] for e in EPS_GRID],
           'per_seed_eps': {}, 'pooled': {}}
    for s in seeds:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
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
            out['per_seed_eps'][f'{s}_{EPS_LABEL[eps]}'] = {'robust': robust,
                                                            'n': len(idx),
                                                            'rate': rate}
            print(f'[seed {s} eps {EPS_LABEL[eps]}] robust={robust}/{len(idx)} '
                  f'rate={rate:.4f} ({time.perf_counter()-t0:.0f}s)', flush=True)
    for eps in EPS_GRID:
        key = EPS_LABEL[eps]
        rates = [out['per_seed_eps'][f'{s}_{key}']['rate'] for s in seeds]
        pooled = {'rate_mean': float(sum(rates) / len(rates)),
                  'rate_std': float(__import__('statistics').stdev(rates))}
        out['pooled'][key] = pooled
        print(f'POOLED {key} mean={pooled["rate_mean"]:.4f} '
              f'std={pooled["rate_std"]:.4f}', flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'SAVED {out_path} (refined CROWN-IBP)', flush=True)


if __name__ == '__main__':
    main()
