# -*- coding: utf-8 -*-
"""
Screen recall against an attack-based gold standard (Section 7.4 of the paper).

Same 100-box MNIST workload as screen_recall.py (eps=0.05): the screen
(budget 50) and a multi-restart PGD attack (10 steps, 2 restarts, the same
attack configuration as the joint protocol) run on identical boxes. PGD is
an incomplete but practical gold standard for "attackable" boxes; we report
how many screen-flagged boxes PGD confirms (false-positive check) and how
many PGD-vulnerable boxes the screen misses (recall proxy).
"""
import os
import sys
import json
import time
import argparse

import torch
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)

from train import MLP, load_mnist            # noqa: E402
import verifier_lib as vl                    # noqa: E402

OUT_DIR = os.path.abspath(os.path.join(BASE, '..', 'results'))
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = os.path.abspath(os.path.join(BASE, '..', 'data', 'checkpoints'))

SEED = 0
N_SAMPLES = 100
BUDGET = 50                # screening evaluations per box
PGD_STEPS = 10
PGD_RESTARTS = 2
PGD_STEP_SIZE = 0.05
SCREEN_SEED = 4242         # same screening stream family as the joint protocol


def pgd_attack(model, x, y, eps, steps=PGD_STEPS, restarts=PGD_RESTARTS,
               step_size=PGD_STEP_SIZE):
    """Multi-restart PGD in L-inf ball (same configuration as Table 19)."""
    x0 = x.clone()
    lo = torch.clamp(x0 - eps, 0, 1)
    hi = torch.clamp(x0 + eps, 0, 1)
    for _ in range(restarts):
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        adv = torch.clamp(x0 + delta, 0, 1).requires_grad_(True)
        for _ in range(steps):
            out = model(adv)
            loss = -out[0, y] + out[0].masked_fill(
                torch.arange(out.size(1), device=out.device) == y, -1e18).max()
            model.zero_grad()
            loss.backward()
            g = adv.grad.data
            adv = adv + step_size * g.sign()
            adv = torch.clamp(adv, lo, hi)
            adv = adv.detach().requires_grad_(True)
        with torch.no_grad():
            if model(adv).argmax(1).item() != y:
                return True, adv.detach()
    return False, None


def screen_verdict(model, x, y, eps, budget=BUDGET):
    """Uniform random sampling screen (Algorithm 1): vulnerable if any
    of the budget draws flips the label; otherwise likely robust."""
    for _ in range(budget):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        xa = torch.clamp(x + delta, 0, 1)
        with torch.no_grad():
            if model(xa).argmax(1).item() != y:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.05)
    ap.add_argument('--out', type=str, default='screen_recall_pgd.json')
    args = ap.parse_args()
    EPS = args.eps

    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]

    model = MLP([784, 64, 64, 10])
    ckpt = os.path.join(CKPT, 'mlp_seed0.pt')
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    # Same 100-box workload as screen_recall.py / complete_solver_compare.py:
    # the first 100 correctly classified test samples.
    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:N_SAMPLES]

    torch.manual_seed(SCREEN_SEED)
    np.random.seed(SCREEN_SEED)

    rows = []
    vuln = 0
    pgd_vuln = 0
    both = 0          # screen vuln AND pgd vuln
    screen_only = 0   # screen vuln but pgd robust
    pgd_only = 0      # pgd vuln but screen likely-robust
    for si in idx:
        x, y = x_te[si].unsqueeze(0), y_te[si]
        t0 = time.time()
        sv = screen_verdict(model, x, y, EPS)
        pv, _ = pgd_attack(model, x, y, EPS)
        dt = time.time() - t0
        vuln += int(sv)
        pgd_vuln += int(pv)
        both += int(sv and pv)
        screen_only += int(sv and not pv)
        pgd_only += int(pv and not sv)
        rows.append({'idx': si, 'screen_vulnerable': sv,
                     'pgd_vulnerable': pv})
        if (len(rows)) % 20 == 0:
            print(f'processed {len(rows)}/{len(idx)} screen_vuln={vuln} '
                  f'pgd_vuln={pgd_vuln} both={both} '
                  f'screen_only={screen_only} pgd_only={pgd_only}',
                  flush=True)

    recall = (both / pgd_vuln) if pgd_vuln > 0 else None
    precision = (both / vuln) if vuln > 0 else None
    summary = {
        'eps': EPS, 'n_samples': N_SAMPLES, 'screen_budget': BUDGET,
        'screen_vulnerable_rate': vuln / N_SAMPLES,
        'pgd_vulnerable_rate': pgd_vuln / N_SAMPLES,
        'both_rate': both / N_SAMPLES,
        'screen_only_rate': screen_only / N_SAMPLES,
        'pgd_only_rate': pgd_only / N_SAMPLES,
        'screen_recall_on_pgd_vulnerable': recall,
        'screen_precision_vs_pgd': precision,
        'wall_time_s': time.time() - t0,
    }
    out = {'summary': summary, 'details': rows}
    fn = os.path.join(OUT_DIR, args.out)
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print('SUMMARY:', json.dumps(summary, indent=1))
    print(f'saved {fn}')


if __name__ == '__main__':
    main()
