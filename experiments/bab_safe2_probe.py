# -*- coding: utf-8 -*-
"""
Interval-Newton BaB probe (novelty upgrade feasibility test).

Question: on the joint protocol's high-dimensional undecided boxes (MNIST
MLP, eps=0.02, where a 2000-node CROWN-BaB decides only 1 of 30 boxes), does
running the root-preserving SAFE2 Phase-R contraction as a preprocessing
step before CROWN-BaB significantly improve the decided fraction?

Control group is the published result results/highdim_bab.json (30 boxes,
2000-node CROWN-BaB).  Treatment: contract_box_r (same schedule as SAFE2
Phase R, M_R=15) on each undecided box, then CROWN-BaB on the contracted
box with the same budget.  Soundness: Phase-R contraction preserves every
root of the original box (identical to the certified SAFE2 chain).

Self-contained layout (same convention as highdim_bab_completion.py):
  <project>/experiments/bab_safe2_probe.py   (this file)
  <project>/experiments/vendor/
  <project>/data/checkpoints/                (mlp_seed*.pt)
  <project>/results/                         (outputs)
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
N_UNDECIDED = 30          # same boxes as highdim_bab.json
NODE_BUDGET = 2000        # per-box node cap (same as the published control)
TIME_BUDGET_S = 180.0     # per-box wall-clock cap (same as the control)
CONTRACT_MAX_ITER = 15    # SAFE2 Phase R schedule for MNIST (M_R = 15)


def width_ratio(lo, hi):
    return float(((hi - lo) / 2.0).mean().item())


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

    # Same undecided-box collection as highdim_bab_completion.py.
    undecided = []
    for pos, si in enumerate(idx):
        x0 = x_te[si]
        c = int(y_te[si].item())
        r, it, _, root, root_marg = vl.verify_safe2(
            model, x0, c, EPS, 0.2, 0.8)
        if r == 'undecided':
            undecided.append((si, x0, c))
    print(f'undecided boxes at eps={EPS}: {len(undecided)}', flush=True)
    undecided = undecided[:N_UNDECIDED]

    # Published control (pure CROWN-BaB on the same 30 boxes).
    control_fn = os.path.join(OUT_DIR, 'highdim_bab.json')
    if os.path.exists(control_fn):
        with open(control_fn, 'r', encoding='utf-8') as f:
            control = json.load(f)
        control_records = control['records']
        print('control loaded from highdim_bab.json: '
              f"{control['summary'].get('decided_robust', 0)} robust, "
              f"{control['summary'].get('decided_not_robust', 0)} not_robust, "
              f"{control['summary'].get('undecided_after_completion', 0)} "
              'unknown', flush=True)
    else:
        control_records = None
        print('highdim_bab.json missing; control will be re-run '
              'alongside the treatment', flush=True)

    records = []
    t0_all = time.perf_counter()
    for k, (si, x0, c) in enumerate(undecided):
        t0 = time.perf_counter()
        X_lo0 = (x0 - EPS).detach().clone()
        X_hi0 = (x0 + EPS).detach().clone()
        w0 = width_ratio(X_lo0, X_hi0)
        X_lo, X_hi, n_itr, status = vl.contract_box_r(
            model, X_lo0, X_hi0, c, max_iter=CONTRACT_MAX_ITER)
        w1 = width_ratio(X_lo, X_hi)
        rec = {'sample': si, 'label': c,
               'contract_iter': int(n_itr), 'contract_status': status,
               'width0': round(w0, 6), 'width1': round(w1, 6),
               'shrink': round(1.0 - w1 / max(w0, 1e-12), 4)}
        if status == 'exist':
            rec['verdict'] = 'not_robust'
            rec['nodes'] = 0
            rec['via'] = 'contraction_exist'
        elif status == 'robust':
            rec['verdict'] = 'robust'
            rec['nodes'] = 0
            rec['via'] = 'contraction_robust'
        elif status == 'tiny':
            rec['verdict'] = 'unknown'
            rec['nodes'] = 0
            rec['via'] = 'contraction_tiny'
        else:
            t1 = time.perf_counter()
            verdict, nodes, cex = vb.verify_bab(
                model, x0, c, EPS, None, node_budget=NODE_BUDGET,
                time_budget=TIME_BUDGET_S, box=(X_lo, X_hi))
            dt_bab = time.perf_counter() - t1
            rec['verdict'] = verdict
            rec['nodes'] = int(nodes)
            rec['via'] = 'bab_on_contracted'
            rec['bab_time_s'] = round(dt_bab, 4)
        rec['time_s'] = round(time.perf_counter() - t0, 4)
        records.append(rec)
        print(f'[{k+1}/{len(undecided)}] si={si} via={rec["via"]} '
              f'verdict={rec["verdict"]} nodes={rec.get("nodes")} '
              f'shrink={rec["shrink"]} itr={rec["contract_iter"]} '
              f'time={rec["time_s"]}s', flush=True)

    robust = [r for r in records if r['verdict'] == 'robust']
    not_robust = [r for r in records if r['verdict'] == 'not_robust']
    unknown = [r for r in records if r['verdict'] == 'unknown']
    summary = {
        'eps': EPS, 'seed': SEED, 'n_boxes': len(records),
        'node_budget': NODE_BUDGET,
        'contract_max_iter': CONTRACT_MAX_ITER,
        'decided_robust': len(robust),
        'decided_not_robust': len(not_robust),
        'decided_total': len(robust) + len(not_robust),
        'undecided_after': len(unknown),
        'mean_shrink': round(sum(r['shrink'] for r in records) /
                             max(len(records), 1), 4),
        'median_shrink': round(
            sorted(r['shrink'] for r in records)[len(records) // 2], 4)
        if records else None,
        'via_contraction_exist': len([r for r in records
                                      if r.get('via') == 'contraction_exist']),
        'via_contraction_robust': len([r for r in records
                                       if r.get('via') == 'contraction_robust']),
        'via_bab_on_contracted': len([r for r in records
                                      if r.get('via') == 'bab_on_contracted']),
        'total_time_s': round(time.perf_counter() - t0_all, 2),
    }
    if control_records is not None:
        c_rob = len([r for r in control_records if r['verdict'] == 'robust'])
        c_nr = len([r for r in control_records if r['verdict'] == 'not_robust'])
        summary['control_decided_total'] = c_rob + c_nr
    out = {'summary': summary, 'records': records}
    fn = os.path.join(OUT_DIR, 'bab_safe2_probe.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SUMMARY:', json.dumps(summary, indent=1), flush=True)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    main()
