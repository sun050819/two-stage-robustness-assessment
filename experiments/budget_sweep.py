# -*- coding: utf-8 -*-
"""
Budget-tightening sweep for the certified two-sided interval (Section 7.7).

For each workload and radius, run the pure random-sampling screen at several
budgets N and count screened-vulnerable samples S(N). Since the screen budget
sequence is nested under the same per-sample seed (torch.manual_seed once,
then budget i.i.d. draws), S(N) is non-decreasing and the sound upper endpoint
U(N) = 1 - (S(N)+E)/n is non-increasing: the undecided region is a resource
that shrinks monotonically as the witness budget grows.

Certification counts (certified robust R, certified vulnerable E) do not
depend on the screen budget: screened samples carry a witness and can never be
certified robust, and E is zero on these workloads (the joint protocol reports
certified vulnerable 0.0%). So L = R/n is constant and U(N) moves with S(N).

Outputs results/budget_sweep.json (pooled per workload/eps/budget).
"""
import os
import sys
import json

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from joint_protocol import screen_rs, OUT_DIR, CKPT          # noqa: E402
from train import MLP, load_mnist                            # noqa: E402
from train_cnn import make_cnn, load_cifar10                 # noqa: E402
import verifier_cnn as vc                                    # noqa: E402

INPUT_SHAPE = (3, 32, 32)
BUDGETS = [50, 100, 200, 500]
EPS_MLP = [0.01, 0.02, 0.05]
EPS_CNN = [2.0 / 255.0, 4.0 / 255.0]


def sweep_workload(kind, eps_list, n_samples, ckpt_prefix, use_cnn):
    """Return dict: {eps: {budget: {'S': cnt, 'U': rate}}} pooled over seeds."""
    # load data/models (same protocol as joint_protocol.run_workload)
    if use_cnn:
        _, _, x_te, y_te = load_cifar10()
        models = {}
        for s in (0, 1, 2):
            model = make_cnn()
            model.load_state_dict(torch.load(
                os.path.join(CKPT, f'{ckpt_prefix}_seed{s}.pt'),
                map_location='cpu'))
            model.eval()
            models[s] = model
        x_view = lambda x: x.view(-1, *INPUT_SHAPE)
    else:
        x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
        models = {}
        for s in (0, 1, 2):
            model = MLP([784, 64, 64, 10])
            model.load_state_dict(torch.load(
                os.path.join(CKPT, f'{ckpt_prefix}_seed{s}.pt'),
                map_location='cpu'))
            model.eval()
            models[s] = model
        x_view = lambda x: x

    out = {str(eps): {b: {'S': 0, 'n': 0, 'U': None} for b in BUDGETS}
           for eps in eps_list}
    for s in (0, 1, 2):
        model = models[s]
        with torch.no_grad():
            preds = model(x_view(x_te)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
        n = len(idx)
        for eps in eps_list:
            for b in BUDGETS:
                cnt = 0
                for pos, si in enumerate(idx.tolist()):
                    x0 = x_te[si]
                    c = int(y_te[si].item())
                    seed_base = 4242 + s * 100000 + int(eps * 1e6) + pos
                    if use_cnn:
                        v, _ = screen_rs(model, x0.view(-1), c, eps, b,
                                         seed_base, INPUT_SHAPE, clamp=True)
                    else:
                        v, _ = screen_rs(model, x0, c, eps, b, seed_base)
                    if v == 'vulnerable':
                        cnt += 1
                out[str(eps)][b]['S'] += cnt
                out[str(eps)][b]['n'] += n
        print(f'[{kind} seed {s}] done', flush=True)
    # U = 1 - S/n  (E=0 on these workloads)
    for eps in eps_list:
        for b in BUDGETS:
            d = out[str(eps)][b]
            d['U'] = 1.0 - d['S'] / d['n']
            d['S_rate'] = d['S'] / d['n']
            print(f'[{kind} eps {eps:.4f} budget {b}] '
                  f'S={d["S"]}/{d["n"]} ({d["S_rate"]:.4f}) '
                  f'U={d["U"]:.4f}', flush=True)
    return out


def main():
    res = {}
    res['mlp'] = sweep_workload('mlp', EPS_MLP, 200, 'mlp', False)
    res['cnn'] = sweep_workload('cnn', EPS_CNN, 500, 'cnn_ibp', True)
    fn = os.path.join(OUT_DIR, 'budget_sweep.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('SAVED', fn, flush=True)


if __name__ == '__main__':
    main()
