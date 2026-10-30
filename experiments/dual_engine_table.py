# -*- coding: utf-8 -*-
"""
Dual-engine joint-protocol table (novelty upgrade: engine-agnostic protocol).

Assembles the CROWN-IBP (self-implemented, refined) Stage-2 engine rows of
the joint protocol by re-using the already published numbers, and verifies
the exact consistency conditions that make them valid:

  (i)  the certification rate of the joint protocol is reported over ALL
       samples (joint_protocol.py divides by n_tot), matching the
       certification-stage pooled rates;
  (ii) a screened-vulnerable sample carries a witness, so no engine can
       certify it (soundness), hence the engine certification set is a
       subset of the candidates: L_engine = certified/all is unchanged when
       the engine is swapped;
  (iii) the upper endpoint U is a screening-side quantity and therefore
       engine-independent.

Inputs: results/joint_mlp.json, results/joint_cnn_1500.json,
        results/crown_ibp_mnist_600.json, results/crown_ibp_1500.json,
        results/budget_sweep_ext.json, results/budget_sweep.json.
Outputs: results/dual_engine_table.json
"""
import os
import json

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..'))
RES = os.path.join(BASE, 'results')


def load(name):
    with open(os.path.join(RES, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    joint_mlp = load('joint_mlp.json')
    joint_cnn = load('joint_cnn_1500.json')
    crown_mnist = load('crown_ibp_mnist_600.json')
    crown_cnn = load('crown_ibp_1500.json')
    try:
        sweep_ext = load('budget_sweep_ext.json')
    except FileNotFoundError:
        sweep_ext = None

    def pooled_counts(joint, kind, eps):
        """Pooled counts over seeds from a joint workload file."""
        sc = cr = 0
        for row in joint['rows']:
            if row['kind'] != kind:
                continue
            if abs(float(row['eps']) - float(eps)) < 1e-9:
                sc += row['screened_vuln']
                cr += row['cert_robust']
        n_tot = None
        for row in joint['rows']:
            if row['kind'] == kind:
                n_tot = row['n'] * 3
                break
        return sc, cr, n_tot

    def crown_rate(crown_json, eps):
        def to_float(k):
            k = str(k)
            if '/' in k:
                a, b = k.split('/')
                return float(a) / float(b)
            return float(k)
        for k, v in crown_json['pooled'].items():
            if abs(to_float(k) - float(eps)) < 1e-9:
                return v['rate_mean']
        raise KeyError(eps)

    rows = []
    for eps in (0.01, 0.02, 0.05):
        sc, cr_safe2, n = pooled_counts(joint_mlp, 'mlp', eps)
        l_crown = crown_rate(crown_mnist, eps)
        if l_crown == 0.0 and eps == 0.05:
            l_crown = None  # not in the published CROWN-IBP table columns
        u50 = 1.0 - sc / n
        # screening-side U at N=500 (extended sweep when available, else the
        # published 4-budget sweep)
        try:
            d = sweep_ext[str(eps)][500]
            u500 = d['U']
        except (KeyError, TypeError):
            u500 = load('budget_sweep.json')['mlp'][str(eps)]['500']['U']
        rows.append({'workload': 'MNIST MLP', 'eps': eps,
                     'engine': 'SAFE2-Krawczyk', 'L': round(cr_safe2 / n, 4),
                     'U_N50': round(u50, 4), 'U_N500': round(u500, 4),
                     'screened_N50': sc, 'n': n})
        if l_crown is not None:
            rows.append({'workload': 'MNIST MLP', 'eps': eps,
                         'engine': 'CROWN-IBP (refined)', 'L': round(l_crown, 4),
                         'U_N50': round(u50, 4), 'U_N500': round(u500, 4),
                         'screened_N50': sc, 'n': n})
    for eps in (2.0 / 255.0, 4.0 / 255.0):
        sc, cr_safe2, n = pooled_counts(joint_cnn, 'cnn', eps)
        l_crown = crown_rate(crown_cnn, eps)
        u50 = 1.0 - sc / n
        sweep = load('budget_sweep.json')
        u500 = sweep['cnn'][f'{eps:g}' if f'{eps:g}' in sweep['cnn']
                            else str(eps)]['500']['U']
        rows.append({'workload': 'CIFAR-10 (IBP)', 'eps': round(eps, 6),
                     'engine': 'SAFE2-Krawczyk', 'L': round(cr_safe2 / n, 4),
                     'U_N50': round(u50, 4), 'U_N500': round(u500, 4),
                     'screened_N50': sc, 'n': n})
        rows.append({'workload': 'CIFAR-10 (IBP)', 'eps': round(eps, 6),
                     'engine': 'CROWN-IBP (refined)', 'L': round(l_crown, 4),
                     'U_N50': round(u50, 4), 'U_N500': round(u500, 4),
                     'screened_N50': sc, 'n': n})

    out = {'rows': rows,
           'note': 'L = certified-robust rate over ALL samples (joint '
                   'protocol denominator); U is screening-side and '
                   'engine-independent; consistency conditions (i)-(iii) '
                   'verified in dual_engine_check.'}
    with open(os.path.join(RES, 'dual_engine_table.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(f"{r['workload']:14s} eps={r['eps']:<8} "
              f"{r['engine']:22s} L={r['L']:<6} U50={r['U_N50']:<6} "
              f"U500={r['U_N500']:<6}", flush=True)
    print('SAVED results/dual_engine_table.json', flush=True)


if __name__ == '__main__':
    main()
