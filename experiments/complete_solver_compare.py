# -*- coding: utf-8 -*-
"""
End-to-end comparison against a complete solver on a real workload
(merged manuscript, Section 7.6 / 7.4).

Runs the complete neuron-branching BaB solver (vendor/verifier_bab.py:
CROWN bounds + exact affine-leaf LP checks, the same solver family as
alpha-beta-CROWN / VNN-COMP entrants, here implemented in-house) on the
MNIST MLP (seed 0) workload of the joint protocol at eps = 0.01 and
reports, for the same first-n correctly classified test samples:

  * certified robust rate of the complete solver;
  * nodes explored and wall time;
  * the SAFE2-Krawczyk rate on the same samples for comparison.

Together with the auto-LiRPA CROWN-IBP reference bounds already reported
(Table 4 of the manuscript, produced by run_baselines_lirpa.py), this
places the interval-method family against both a complete solver and an
optimized linear relaxation on identical models.

Self-contained layout (see joint_protocol.py).
"""
import os
import sys
import json
import time

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)

from train import MLP, load_mnist            # noqa: E402
import verifier_lib as vl                    # noqa: E402
import verifier_bab as vb                    # noqa: E402

OUT_DIR = os.path.join(BASE, '..', 'results')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = os.path.join(BASE, '..', 'data', 'checkpoints')
CKPT = os.path.abspath(CKPT)

SEED = 0
EPS = 0.01
N_SAMPLES = 100          # first-n correctly classified test samples
NODE_BUDGET = 2000       # per-box node cap
TIME_BUDGET_S = 180.0    # per-box wall-clock cap (pathological-LP guard)
ALPHA_MIN_MLP = 0.2
ALPHA_FACTOR_MLP = 0.8


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    model = MLP([784, 64, 64, 10])
    model.load_state_dict(torch.load(
        os.path.join(CKPT, f'mlp_seed{SEED}.pt'), map_location='cpu'))
    model.eval()

    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:N_SAMPLES]

    safe2_robust = 0
    bab_robust = 0
    bab_not_robust = 0
    bab_unknown = 0
    bab_nodes = 0
    bab_nodes_robust = []
    bab_time = 0.0
    safe2_time = 0.0
    details = []

    t_all = time.perf_counter()
    for si in idx:
        x0 = x_te[si]
        c = int(y_te[si].item())

        t0 = time.perf_counter()
        r_safe2, it, _, root, root_marg = vl.verify_safe2(
            model, x0, c, EPS, ALPHA_MIN_MLP, ALPHA_FACTOR_MLP)
        safe2_time += time.perf_counter() - t0
        if r_safe2 == 'robust':
            safe2_robust += 1

        t0 = time.perf_counter()
        verdict, nodes, cex = vb.verify_bab(
            model, x0, c, EPS, None, node_budget=NODE_BUDGET,
            time_budget=TIME_BUDGET_S)
        bab_time += time.perf_counter() - t0
        bab_nodes += nodes
        if verdict == 'robust':
            bab_robust += 1
            bab_nodes_robust.append(nodes)
        elif verdict == 'not_robust':
            bab_not_robust += 1
        else:
            bab_unknown += 1

        details.append({'sample': si, 'safe2': r_safe2, 'bab': verdict,
                        'bab_nodes': int(nodes)})
        if (si + 1) % 20 == 0:
            print(f'processed {si+1}/{len(idx)} '
                  f'safe2_robust={safe2_robust} bab_robust={bab_robust} '
                  f'bab_not_robust={bab_not_robust} '
                  f'bab_unknown={bab_unknown} total_nodes={bab_nodes}',
                  flush=True)

    total_s = time.perf_counter() - t_all
    n = len(idx)
    summary = {
        'seed': SEED, 'eps': EPS, 'n_samples': n,
        'node_budget': NODE_BUDGET,
        'safe2_certified_robust_rate': round(safe2_robust / n, 4),
        'safe2_total_time_s': round(safe2_time, 2),
        'bab_certified_robust_rate': round(bab_robust / n, 4),
        'bab_not_robust_rate': round(bab_not_robust / n, 4),
        'bab_unknown_rate': round(bab_unknown / n, 4),
        'bab_total_nodes': int(bab_nodes),
        'bab_mean_nodes_robust': round(
            sum(bab_nodes_robust) / max(len(bab_nodes_robust), 1), 1),
        'bab_total_time_s': round(bab_time, 2),
        'wall_time_s': round(total_s, 2),
    }
    out = {'summary': summary, 'details': details}
    fn = os.path.join(OUT_DIR, 'complete_solver_compare.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SUMMARY:', json.dumps(summary, indent=1), flush=True)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    main()
