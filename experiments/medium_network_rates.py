# -*- coding: utf-8 -*-
"""Certification rates of the two-stage joint protocol on a medium-scale
IBP-trained MLP (FC-256 family, [784,256,256,10]), MNIST, 3 seeds.

Stage 1: random-direction screening (screen_rs, 50 forward passes).
Stage 2: SAFE2 gated verification on the remaining candidates.
Outputs screened-vulnerable / certified-robust / certified-vulnerable /
undecided rates per eps (pooled over seeds and per seed), written to
results/medium_mlp_rates.json.

The protocol is identical to the small-MLP joint workload (Table 14);
only the network width changes.
"""
import json
import os
import sys
import time

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
sys.path.insert(0, VENDOR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import MLP, load_mnist             # noqa: E402
from verifier_lib import verify_safe2         # noqa: E402
from joint_protocol import screen_rs, BUDGET  # noqa: E402

DIMS = [784, 256, 256, 10]
CKPT = os.path.join(BASE, 'data', 'checkpoints')
EPS_LIST = [0.01, 0.02, 0.05]
SEEDS = [0, 1, 2]
N_PER_SEED = 200
OUT = os.path.join(BASE, 'results', 'medium_mlp_rates.json')


def log(msg):
    print(msg, flush=True)


def main():
    torch.set_num_threads(4)
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]

    pooled = {eps: {'screened_vuln': 0, 'cert_robust': 0, 'cert_vuln': 0,
                    'undecided': 0, 'errors': 0, 'n': 0}
              for eps in EPS_LIST}
    per_seed = {}
    t_screen = 0.0
    t_cert = 0.0

    for s in SEEDS:
        model = MLP(DIMS)
        model.load_state_dict(torch.load(
            os.path.join(CKPT, f'mlp_medium_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_PER_SEED]
        log(f'[medium seed {s}] n_samples={len(idx)}')

        for eps in EPS_LIST:
            cnt = {'screened_vuln': 0, 'cert_robust': 0, 'cert_vuln': 0,
                   'undecided': 0, 'errors': 0}
            for pos, si in enumerate(idx.tolist()):
                x0 = x_te[si]
                c = int(y_te[si].item())
                seed_base = 4242 + s * 100000 + int(eps * 1e6) + pos
                t0 = time.perf_counter()
                verdict, min_marg = screen_rs(model, x0, c, eps, BUDGET, seed_base)
                t_screen += time.perf_counter() - t0
                if verdict == 'vulnerable':
                    cnt['screened_vuln'] += 1
                    continue
                t0 = time.perf_counter()
                cert, _, _, _, _ = verify_safe2(model, x0, c, eps)
                t_cert += time.perf_counter() - t0
                if cert == 'robust':
                    cnt['cert_robust'] += 1
                elif cert == 'exist':
                    cnt['cert_vuln'] += 1
                elif cert == 'undecided':
                    cnt['undecided'] += 1
                else:
                    cnt['errors'] += 1
            for k in cnt:
                pooled[eps][k] += cnt[k]
                pooled[eps]['n'] += len(idx)
            per_seed.setdefault(s, {})[eps] = dict(cnt)
            log(f'[medium seed {s}] eps={eps} {cnt}')

    res = {
        'workload': 'MNIST medium-scale MLP [784,256,256,10] (IBP-trained, 3 seeds)',
        'n_per_seed': N_PER_SEED,
        'eps_list': EPS_LIST,
        'pooled_rates': {},
        'per_seed': {str(s): {str(eps): d for eps, d in d2.items()}
                     for s, d2 in per_seed.items()},
        'timing_s': {'screening': round(t_screen, 2), 'safe2': round(t_cert, 2)},
    }
    for eps in EPS_LIST:
        p = pooled[eps]
        n = max(p['n'], 1)
        res['pooled_rates'][str(eps)] = {
            'n': p['n'],
            'screened_vuln_rate': round(p['screened_vuln'] / n, 4),
            'certified_robust_rate': round(p['cert_robust'] / n, 4),
            'certified_vuln_rate': round(p['cert_vuln'] / n, 4),
            'undecided_rate': round(p['undecided'] / n, 4),
            'counts': {k: p[k] for k in ('screened_vuln', 'cert_robust',
                                         'cert_vuln', 'undecided', 'errors')},
        }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f'SAVED {OUT}')


if __name__ == '__main__':
    main()
