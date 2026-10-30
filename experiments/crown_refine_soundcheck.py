# -*- coding: utf-8 -*-
"""Large-sample soundness check of the REFINED CROWN-IBP (refine=True).

30 correctly-classified samples x 3 seeds x 1500 uniform draws/logit at
eps=0.01 (plus 5 samples at eps=0.02), no-clamp box (the exact 600-set
rerun口径). Reports worst per-logit lo/hi violation and margin checks.
Zero violation = the 78.2% certified rate is sound.
"""
import os
import sys
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from baseline_crown import crown_bounds_logits      # noqa: E402
from train import MLP, load_mnist                   # noqa: E402

N_SAMPLES = 30
N_DRAWS = 1500
SEEDS = [0, 1, 2]


def check(model, x_te, y_te, n_samples, eps):
    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
    worst_lo = 0.0
    worst_hi = 0.0
    n_robust_refined = 0
    for si in idx:
        x0 = x_te[si].flatten()
        c = int(y_te[si].item())
        lo = x0 - eps
        hi = x0 + eps
        lo_b, hi_b = crown_bounds_logits(model, lo, hi, (1, 28, 28), refine=True)
        others = [j for j in range(10) if j != c]
        if float(lo_b[c] - torch.max(hi_b[others])) > 0:
            n_robust_refined += 1
        torch.manual_seed(int(si) + 13)
        samples = lo + (hi - lo) * torch.rand(N_DRAWS, 784)
        logits = model(samples)
        emp_min = logits.min(0).values
        emp_max = logits.max(0).values
        lo_viol = (lo_b - emp_min).clamp(min=0.0)
        hi_viol = (emp_max - hi_b).clamp(min=0.0)
        worst_lo = max(worst_lo, float(lo_viol.max()))
        worst_hi = max(worst_hi, float(hi_viol.max()))
    return worst_lo, worst_hi, n_robust_refined, len(idx)


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    for s in SEEDS:
        model = MLP([784, 64, 64, 10])
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'mlp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        wl, wh, nrob, n = check(model, x_te, y_te, N_SAMPLES, 0.01)
        print(f'[seed {s} eps=0.01] refined-robust {nrob}/{n} | '
              f'worst lo-viol={wl:.7f} hi-viol={wh:.7f}', flush=True)
    # eps=0.02 spot check on seed 0
    model = MLP([784, 64, 64, 10])
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'mlp_seed0.pt'), map_location='cpu'))
    model.eval()
    wl, wh, nrob, n = check(model, x_te, y_te, 5, 0.02)
    print(f'[seed 0 eps=0.02] refined-robust {nrob}/{n} | '
          f'worst lo-viol={wl:.7f} hi-viol={wh:.7f}', flush=True)
    print('ZERO violation required for soundness', flush=True)


if __name__ == '__main__':
    main()
