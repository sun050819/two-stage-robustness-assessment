# -*- coding: utf-8 -*-
"""
Screening recall against a complete solver (merged paper,
Section 7.4 of the paper).

On the same 100-box MNIST MLP (seed 0) workload at eps = 0.01 as the
complete-solver comparison (complete_solver_compare.py):

  * the fixed-budget random screen (N = 50) labels each box either
    "vulnerable" (explicit witness g(x) <= 0) or "likely robust";
  * the complete neuron-branching BaB solver (2000 nodes / 180 s budget)
    decides the box robust / not robust / undecided.

Reported:
  * screen recall on the boxes the complete solver can decide
    (not-robust boxes caught by the screen);
  * screen precision (screen-vulnerable boxes confirmed by BaB);
  * screen-witness boxes left undecided by the BaB budget (the screen
    provides information the budgeted solver does not);
  * consistency check: no screen-vulnerable box is certified robust by BaB
    (a witness is a concrete adversarial input, so this must hold).
"""
import os
import sys
import json
import time
import argparse

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)

from train import MLP, load_mnist            # noqa: E402
import verifier_bab as vb                    # noqa: E402

OUT_DIR = os.path.abspath(os.path.join(BASE, '..', 'results'))
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = os.path.abspath(os.path.join(BASE, '..', 'data', 'checkpoints'))

SEED = 0
N_SAMPLES = 100
BUDGET = 50                # screening evaluations per box
NODE_BUDGET = 2000
TIME_BUDGET_S = 180.0
SCREEN_SEED = 4242         # same screening stream family as the joint protocol


def screen_rs(model, x0, c, eps, budget=BUDGET):
    """Pure random sampling screen; returns (verdict, best_margin)."""
    best_margin = None
    torch.manual_seed(SCREEN_SEED + int(x0.sum().item() * 1e6) % 100000)
    with torch.no_grad():
        for _ in range(budget):
            delta = (torch.rand_like(x0) * 2.0 - 1.0) * eps
            x = torch.clamp(x0 + delta, 0.0, 1.0)
            logits = model(x.unsqueeze(0)).squeeze(0)
            others = logits.clone()
            others[c] = float('-inf')
            g = float(logits[c] - others.max())
            if best_margin is None or g < best_margin:
                best_margin = g
            if g <= 0:
                return 'vulnerable', g, x
    return 'likely_robust', best_margin, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--out', type=str, default='screen_recall.json')
    args = ap.parse_args()
    EPS = args.eps

    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    model = MLP([784, 64, 64, 10])
    model.load_state_dict(torch.load(
        os.path.join(CKPT, f'mlp_seed{SEED}.pt'), map_location='cpu'))
    model.eval()

    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:N_SAMPLES]

    rows = []          # per-box detail
    n_vuln = n_likely = 0
    n_bab_not = n_bab_robust = n_bab_unknown = 0
    catch = 0          # screen-vulnerable AND bab-not-robust
    confirmed = 0      # screen-vulnerable AND bab confirms (not_robust)
    undecided_after_witness = 0   # screen-vulnerable, bab undecided
    false_pos = 0      # screen-vulnerable but bab robust (must be 0)
    t0 = time.perf_counter()
    for si in idx:
        x0 = x_te[si]
        c = int(y_te[si].item())
        s_verdict, s_margin, s_wit = screen_rs(model, x0, c, EPS)
        verdict, nodes, cex = vb.verify_bab(
            model, x0, c, EPS, None, node_budget=NODE_BUDGET,
            time_budget=TIME_BUDGET_S)
        if s_verdict == 'vulnerable':
            n_vuln += 1
        else:
            n_likely += 1
        if verdict == 'not_robust':
            n_bab_not += 1
            if s_verdict == 'vulnerable':
                catch += 1
                confirmed += 1
        elif verdict == 'robust':
            n_bab_robust += 1
            if s_verdict == 'vulnerable':
                false_pos += 1
        else:
            n_bab_unknown += 1
            if s_verdict == 'vulnerable':
                undecided_after_witness += 1
        rows.append({'sample': si, 'screen': s_verdict,
                     'screen_min_margin': round(s_margin, 6),
                     'bab': verdict, 'bab_nodes': int(nodes)})
        if (si + 1) % 20 == 0:
            print(f'processed {si+1}/{len(idx)} vuln={n_vuln} '
                  f'bab_not={n_bab_not} bab_robust={n_bab_robust} '
                  f'bab_unknown={n_bab_unknown} catch={catch} '
                  f'undecided_witness={undecided_after_witness}', flush=True)

    n = len(idx)
    decidable = n_bab_not + n_bab_robust
    summary = {
        'seed': SEED, 'eps': EPS, 'n_samples': n, 'screen_budget': BUDGET,
        'node_budget': NODE_BUDGET,
        'screen_vulnerable_rate': round(n_vuln / n, 4),
        'screen_likely_robust_rate': round(n_likely / n, 4),
        'bab_not_robust_rate': round(n_bab_not / n, 4),
        'bab_robust_rate': round(n_bab_robust / n, 4),
        'bab_unknown_rate': round(n_bab_unknown / n, 4),
        'bab_decidable_rate': round(decidable / n, 4),
        # Recall of the screen against the complete solver's decisions:
        'screen_recall_on_decidable': round(
            catch / decidable, 4) if decidable else None,
        'screen_recall_on_not_robust': round(
            catch / n_bab_not, 4) if n_bab_not else None,
        # Precision: screen-vulnerable boxes confirmed by the solver:
        'screen_precision_vs_bab': round(
            confirmed / n_vuln, 4) if n_vuln else None,
        # Screen witnesses the budgeted BaB could not confirm:
        'witness_boxes_undecided_by_bab': undecided_after_witness,
        'witness_boxes_undecided_rate': round(
            undecided_after_witness / n, 4),
        # Consistency: must be 0.
        'screen_vuln_but_bab_robust': false_pos,
        'wall_time_s': round(time.perf_counter() - t0, 2),
    }
    out = {'summary': summary, 'details': rows}
    fn = os.path.join(OUT_DIR, args.out)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SUMMARY:', json.dumps(summary, indent=1), flush=True)
    print(f'saved {fn}', flush=True)


if __name__ == '__main__':
    main()
