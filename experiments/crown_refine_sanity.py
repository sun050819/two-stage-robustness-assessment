# -*- coding: utf-8 -*-
"""Sanity: refined CROWN-IBP (RefineIntermediateBounds) vs fixed-relaxation.

- margin comparison on 6 MNIST MLP samples (no clamp, seed 0 model)
- sampling-based soundness of the refined bounds (2000 draws/logit)
- refined margin must be >= plain margin (tightening is sound), and the
  refined bounds must still enclose the empirical logit range.
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
N_SAMPLES = 6
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
    n_tighter = 0
    with torch.no_grad():
        for si in idx:
            x0 = x_te[si].flatten()
            c = int(y_te[si].item())
            lo = x0 - EPS          # no clamp: same box as the 600-set rerun
            hi = x0 + EPS
            lo_p, hi_p = crown_bounds_logits(model, lo, hi, (1, 28, 28), refine=False)
            lo_r, hi_r = crown_bounds_logits(model, lo, hi, (1, 28, 28), refine=True)
            others = [j for j in range(10) if j != c]
            m_p = float(lo_p[c] - torch.max(hi_p[others]))
            m_r = float(lo_r[c] - torch.max(hi_r[others]))
            width_p = float(torch.max(hi_p - lo_p))
            width_r = float(torch.max(hi_r - lo_r))
            tighter = m_r >= m_p - 1e-9
            n_tighter += int(tighter)
            print(f'sample {si} c={c}: plain margin={m_p:.5f} '
                  f'refined margin={m_r:.5f} (tighter={tighter}) | '
                  f'max logit width plain={width_p:.4f} refined={width_r:.4f}',
                  flush=True)
            # sampling soundness of the REFINED bounds (no clamp box)
            torch.manual_seed(int(si) + 7)
            samples = lo + (hi - lo) * torch.rand(N_DRAWS, 784)
            logits = model(samples)
            emp_min = logits.min(0).values
            emp_max = logits.max(0).values
            lo_viol = (lo_r - emp_min).clamp(min=0.0)
            hi_viol = (emp_max - hi_r).clamp(min=0.0)
            worst_lo_viol = max(worst_lo_viol, float(lo_viol.max()))
            worst_hi_viol = max(worst_hi_viol, float(hi_viol.max()))
    print(f'RESULT: refined>=plain on {n_tighter}/{N_SAMPLES} samples | '
          f'worst lo-viol={worst_lo_viol:.6f} worst hi-viol={worst_hi_viol:.6f} '
          f'(0 = sound)', flush=True)


if __name__ == '__main__':
    main()
