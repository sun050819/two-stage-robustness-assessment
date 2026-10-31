# -*- coding: utf-8 -*-
"""
High-dimensional branch-and-bound completion of the joint protocol's
undecided boxes (merged paper, Section 7.6 / 7.4).

For the MNIST MLPs at eps = 0.02 the SAFE2-Krawczyk certification stage
certifies nothing and nearly every box is undecided.  This script takes
the undecided boxes of one model (seed 0), runs the complete
neuron-branching BaB solver (CROWN bounds + exact affine-leaf LP checks,
vendor/verifier_bab.py) on the first 30 such boxes, and reports how many
are decided by the complete solver, how many nodes it takes, and the
wall time -- closing the "undecided -> completed by branch-and-bound"
loop of the protocol on a high-dimensional workload.

Self-contained layout (same convention as joint_protocol.py):
  <project>/experiments/highdim_bab_completion.py  (this file)
  <project>/experiments/vendor/                    (SAFE2 + BaB modules)
  <project>/data/checkpoints/                      (mlp_seed*.pt)
  <project>/results/                               (outputs)
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

EPS = 0.02
SEED = 0
N_UNDECIDED = 30          # number of undecided boxes to complete
NODE_BUDGET = 2000        # per-box node cap for the complete solver
TIME_BUDGET_S = 180.0     # per-box wall-clock cap (pathological-LP guard)
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
    idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()
    print(f'correctly classified test samples: {len(idx)}', flush=True)

    # Stage 2 alone on the protocol candidates -> collect undecided boxes.
    undecided = []
    for pos, si in enumerate(idx):
        x0 = x_te[si]
        c = int(y_te[si].item())
        r, it, _, root, root_marg = vl.verify_safe2(
            model, x0, c, EPS, ALPHA_MIN_MLP, ALPHA_FACTOR_MLP)
        if r == 'undecided':
            undecided.append((si, x0, c))
    print(f'undecided boxes at eps={EPS}: {len(undecided)}', flush=True)
    undecided = undecided[:N_UNDECIDED]
    print(f'completing first {len(undecided)} undecided boxes '
          f'with the complete BaB solver (node_budget={NODE_BUDGET})',
          flush=True)

    records = []
    t0_all = time.perf_counter()
    for si, x0, c in undecided:
        t0 = time.perf_counter()
        verdict, nodes, cex = vb.verify_bab(
            model, x0, c, EPS, None, node_budget=NODE_BUDGET,
            time_budget=TIME_BUDGET_S)
        dt = time.perf_counter() - t0
        rec = {'sample': si, 'label': c, 'verdict': verdict,
               'nodes': int(nodes), 'time_s': round(dt, 4),
               'timed_out': dt >= TIME_BUDGET_S - 5.0,
               'counterexample_margin': None}
        if cex is not None:
            with torch.no_grad():
                logits = model(cex.unsqueeze(0)).squeeze(0)
                others = logits.clone()
                others[c] = float('-inf')
                rec['counterexample_margin'] = round(
                    float(logits[c] - others.max()), 6)
        records.append(rec)
        print(f'si={si} verdict={verdict} nodes={nodes} time={dt:.2f}s',
              flush=True)

    robust = [r for r in records if r['verdict'] == 'robust']
    not_robust = [r for r in records if r['verdict'] == 'not_robust']
    unknown = [r for r in records if r['verdict'] == 'unknown']
    total_s = time.perf_counter() - t0_all
    summary = {
        'eps': EPS, 'seed': SEED, 'n_boxes': len(records),
        'node_budget': NODE_BUDGET,
        'decided_robust': len(robust),
        'decided_not_robust': len(not_robust),
        'undecided_after_completion': len(unknown),
        'total_nodes': sum(r['nodes'] for r in records),
        'mean_nodes_decided': round(
            (sum(r['nodes'] for r in robust + not_robust) /
             max(len(robust) + len(not_robust), 1)), 1),
        'mean_nodes_unknown': round(
            (sum(r['nodes'] for r in unknown) / max(len(unknown), 1)), 1),
        'total_time_s': round(total_s, 2),
    }
    out = {'summary': summary, 'records': records}
    fn = os.path.join(OUT_DIR, 'highdim_bab.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SUMMARY:', json.dumps(summary, indent=1), flush=True)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    main()
