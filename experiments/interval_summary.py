# -*- coding: utf-8 -*-
"""Certified two-sided verdict intervals (witness-complete summary).

Aggregates the *certification-stage* three-state counts (SAFE2-Krawczyk,
no screening) into a sound interval on the overall robust rate:

    L = (#robust) / n    sound lower bound (SAFE2-Krawczyk proofs)
    U = 1 - (#exist) / n sound upper bound (existence witnesses with
                         margin <= 0 guaranteed by interval analysis)

CIFAR-10 CNN: consumes results/cnn_safe2_3state_1500.json (3 seeds x 500
x 4 radii, no screening). MNIST: 600-box certification grid, 102 robust /
0 existence witnesses -> [0.170, 1.000]. Writes results/interval_summary.json.
"""
import os
import json

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS = os.path.join(BASE, 'results')
OUT_PATH = os.path.join(RESULTS, 'interval_summary.json')

EPS_LABEL = {0.0039: '1/255', 0.0078: '2/255',
             0.0157: '4/255', 0.0314: '8/255'}


def main():
    three = json.load(open(os.path.join(RESULTS,
                                        'cnn_safe2_3state_1500.json'),
                           encoding='utf-8'))
    per_eps = {}
    for key, cnt in three['per_seed_eps'].items():
        _, eps = key.split('_')
        eps = float(eps)
        rec = per_eps.setdefault(eps, {'robust': 0, 'exist': 0,
                                       'undecided': 0, 'n': 0})
        rec['robust'] += cnt['robust']
        rec['exist'] += cnt['exist']
        rec['undecided'] += cnt['undecided']
        rec['n'] += cnt['robust'] + cnt['exist'] + cnt['undecided']
    cifar = {}
    for eps in sorted(per_eps):
        r = per_eps[eps]
        L = r['robust'] / r['n']
        U = 1.0 - r['exist'] / r['n']
        cifar[EPS_LABEL[eps]] = {'n': r['n'], 'robust': r['robust'],
                                 'exist': r['exist'],
                                 'undecided': r['undecided'],
                                 'L': L, 'U': U, 'width': U - L}
        print(f'CIFAR {EPS_LABEL[eps]:>5}: n={r["n"]} robust={r["robust"]} '
              f'exist={r["exist"]} -> [{L:.4f}, {U:.4f}] '
              f'(width {U-L:.4f})', flush=True)
    mnist = {'n': 600, 'robust': 102, 'exist': 0, 'undecided': 498,
             'L': 102 / 600, 'U': 1.0, 'width': 1.0 - 102 / 600}
    print(f'MNIST      : n=600 robust=102 exist=0 -> '
          f'[{mnist["L"]:.4f}, {mnist["U"]:.4f}]', flush=True)
    out = {'cifar_cnn': cifar, 'mnist_mlp': mnist,
           'note': 'L from SAFE2-Krawczyk certified-robust proofs; U from '
                   'existence witnesses (margin<=0, interval-guaranteed). '
                   'Certification stage only, no screening.'}
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
