# -*- coding: utf-8 -*-
"""Soundness check of the REFINED CROWN-IBP on the CIFAR-10 CNN (clamp box,
the exact 1500-set rerun口径). 12 samples x 1000 draws/logit at 2/255 per
seed, worst per-logit lo/hi violation; zero violation = sound.
"""
import os
import sys
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from baseline_crown import crown_bounds_logits      # noqa: E402
from train_cnn import make_cnn, load_cifar10        # noqa: E402

INPUT_SHAPE = (3, 32, 32)
N_SAMPLES = 12
N_DRAWS = 1000
EPS = 2.0 / 255.0


def main():
    _, _, x_te, y_te = load_cifar10()
    worst_lo = 0.0
    worst_hi = 0.0
    n_rob = 0
    for s in [0, 1, 2]:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        for si in idx:
            x0 = x_te[si].flatten()
            c = int(y_te[si].item())
            lo = torch.clamp(x0 - EPS, 0.0, 1.0)
            hi = torch.clamp(x0 + EPS, 0.0, 1.0)
            lo_b, hi_b = crown_bounds_logits(model, lo, hi, INPUT_SHAPE, refine=True)
            others = [j for j in range(10) if j != c]
            n_rob += int(float(lo_b[c] - torch.max(hi_b[others])) > 0)
            torch.manual_seed(int(si) + 29)
            samples = lo.flatten() + (hi - lo).flatten() * torch.rand(N_DRAWS, 3072)
            logits = model(samples.view(-1, *INPUT_SHAPE))
            emp_min = logits.min(0).values
            emp_max = logits.max(0).values
            lo_viol = (lo_b - emp_min).clamp(min=0.0)
            hi_viol = (emp_max - hi_b).clamp(min=0.0)
            worst_lo = max(worst_lo, float(lo_viol.max()))
            worst_hi = max(worst_hi, float(hi_viol.max()))
        print(f'[seed {s}] samples checked, worst so far lo={worst_lo:.7f} '
              f'hi={worst_hi:.7f}', flush=True)
    print(f'FINAL: worst lo-viol={worst_lo:.7f} hi-viol={worst_hi:.7f} '
          f'(0 = sound) | refined-robust total={n_rob}/{3 * N_SAMPLES}',
          flush=True)


if __name__ == '__main__':
    main()
