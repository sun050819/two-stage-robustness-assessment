# -*- coding: utf-8 -*-
"""
Extended budget sweep with exponential-convergence fit (novelty upgrade).

For each MLP workload/radius, run the nested random-sampling screen at
budgets N in {50, 100, 200, 500, 1000, 2000} and fit the theoretical shape
E[S(N)] = n * (1 - (1-p)^N) (uniform sampling in a box where the vulnerable
region occupies fraction p of the volume).  The fit gives an estimated
per-sample discovery rate p_hat and the implied convergence rate of the
upper endpoint U(N) = 1 - S(N)/n toward its limit, turning the monotone
tightening (Proposition: prop:tighten) into a *quantified* budget-interval
exchange law.  Same seed protocol as budget_sweep.py / joint_protocol.py.

Outputs results/budget_sweep_ext.json (MLP; CIFAR keeps the published 4-budget
table budget_sweep.json).
"""
import os
import sys
import json
import math

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from joint_protocol import screen_rs, OUT_DIR, CKPT          # noqa: E402
from train import MLP, load_mnist                            # noqa: E402

BUDGETS = [50, 100, 200, 500, 1000, 2000]
EPS_MLP = [0.01, 0.02, 0.05]
N_PER_SEED = 200


def fit_exponential(n, budget_counts):
    """Fit S(N) ~ n*(1-(1-p)^N) by least squares over (N, S) pairs.

    Returns {'p_hat', 'pred_saturated', 'rmse', 'r2_shape'}: r2_shape is the
    coefficient of determination of the fitted curve against the observed
    S(N) (shape agreement with the exponential law).
    """
    import numpy as np
    Ns = np.array([b for b in BUDGETS], dtype=float)
    Ss = np.array([budget_counts[b] for b in BUDGETS], dtype=float)
    # residual sum of squares over a grid of p
    best = None
    for p in np.linspace(1e-6, 0.5, 4000):
        pred = n * (1.0 - (1.0 - p) ** Ns)
        rss = float(((Ss - pred) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, p)
    rss, p_hat = best
    pred = n * (1.0 - (1.0 - p_hat) ** Ns)
    ss_tot = float(((Ss - Ss.mean()) ** 2).sum())
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float('nan')
    return {'p_hat': round(float(p_hat), 6),
            'pred_saturated': round(n, 0),
            'rmse': round(math.sqrt(rss / len(Ns)), 3),
            'r2_shape': round(r2, 4) if not math.isnan(r2) else None}


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    models = {}
    for s in (0, 1, 2):
        model = MLP([784, 64, 64, 10])
        model.load_state_dict(torch.load(
            os.path.join(CKPT, f'mlp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        models[s] = model

    out = {str(eps): {b: {'S': 0, 'n': 0, 'U': None} for b in BUDGETS}
           for eps in EPS_MLP}
    for s in (0, 1, 2):
        model = models[s]
        with torch.no_grad():
            preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_PER_SEED]
        n = len(idx)
        for eps in EPS_MLP:
            for b in BUDGETS:
                cnt = 0
                for pos, si in enumerate(idx.tolist()):
                    x0 = x_te[si]
                    c = int(y_te[si].item())
                    seed_base = 4242 + s * 100000 + int(eps * 1e6) + pos
                    v, _ = screen_rs(model, x0, c, eps, b, seed_base)
                    if v == 'vulnerable':
                        cnt += 1
                out[str(eps)][b]['S'] += cnt
                out[str(eps)][b]['n'] += n
        print(f'[mlp seed {s}] done', flush=True)

    for eps in EPS_MLP:
        for b in BUDGETS:
            d = out[str(eps)][b]
            d['U'] = 1.0 - d['S'] / d['n']
            d['S_rate'] = d['S'] / d['n']
        out[str(eps)]['fit'] = fit_exponential(
            out[str(eps)][BUDGETS[0]]['n'],
            {b: out[str(eps)][b]['S'] for b in BUDGETS})
        print(f'[mlp eps {eps:.4f}] ' +
              ' '.join(f'{b}:{out[str(eps)][b]["S"]}' for b in BUDGETS) +
              f' fit={out[str(eps)]["fit"]}', flush=True)

    fn = os.path.join(OUT_DIR, 'budget_sweep_ext.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', fn, flush=True)


if __name__ == '__main__':
    main()
