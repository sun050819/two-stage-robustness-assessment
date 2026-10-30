# -*- coding: utf-8 -*-
"""MNIST MLP CROWN soundness check (training weights, seed 0).

For 5 correctly-classified samples and ALL logits, verify that the CROWN
interval [lo, hi] of each logit encloses the empirical range over many
uniform samples from the box (sampling-based soundness check), the same
criterion applied to the CIFAR CNN path after the conv-forward fix.
Prints the maximum violation per logit (lo > sampled-min or hi < sampled-max).
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

EPS = 0.01
N_SAMPLES = 5
N_DRAWS = 2000


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    model = MLP([784, 64, 64, 10])
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'mlp_seed0.pt'), map_location='cpu'))
    model.eval()
    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]

    worst_lo_viol = 0.0
    worst_hi_viol = 0.0
    with torch.no_grad():
        for si in idx:
            x0 = x_te[si].flatten()
            c = int(y_te[si].item())
            lo = torch.clamp(x0 - EPS, 0.0, 1.0)
            hi = torch.clamp(x0 + EPS, 0.0, 1.0)
            lo_b, hi_b = crown_bounds_logits(model, lo, hi, (1, 28, 28))
            # empirical range per logit over uniform draws
            torch.manual_seed(int(si) + 7)
            samples = lo + (hi - lo) * torch.rand(N_DRAWS, 784)
            logits = model(samples)
            emp_min = logits.min(0).values
            emp_max = logits.max(0).values
            lo_viol = (lo_b - emp_min).clamp(min=0.0)
            hi_viol = (emp_max - hi_b).clamp(min=0.0)
            worst_lo_viol = max(worst_lo_viol, float(lo_viol.max()))
            worst_hi_viol = max(worst_hi_viol, float(hi_viol.max()))
            print(f'sample {si} c={c}: '
                  f'worst lo-viol={float(lo_viol.max()):.6f} '
                  f'worst hi-viol={float(hi_viol.max()):.6f} '
                  f'| lo_b[c]={float(lo_b[c]):.4f} emp_min[c]={float(emp_min[c]):.4f} '
                  f'hi_b[c]={float(hi_b[c]):.4f} emp_max[c]={float(emp_max[c]):.4f}',
                  flush=True)
            # margin soundness: lo_b[c] > max hi_b[others]?
            others = [j for j in range(10) if j != c]
            margin_lo = float(lo_b[c] - torch.max(hi_b[others]))
            print(f'  margin_lo={margin_lo:.6f} (robust iff >0)', flush=True)
    print(f'WORST over samples: lo-viol={worst_lo_viol:.6f} '
          f'hi-viol={worst_hi_viol:.6f}', flush=True)


if __name__ == '__main__':
    main()
