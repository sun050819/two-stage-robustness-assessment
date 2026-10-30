# -*- coding: utf-8 -*-
"""Re-measure per-sample verification time on the IBP-trained MNIST MLPs
(3 seeds) for Table tab:runtime, matching the CIFAR measurement protocol:
time IBP / classical Krawczyk / SAFE2-Krawczyk over eps {0.01..0.2},
report mean + population std across seeds (one CPU core).
"""
import os, sys, json, time
import torch
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from verifier_lib import verify_ibp, verify_krawczyk_classical, verify_safe2  # noqa
from train import MLP, load_mnist  # noqa

SEEDS = [0, 1, 2]
EPS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
N_PER_SEED = 50


def main():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    out = {'n_per_seed': N_PER_SEED, 'eps_grid': EPS, 'per_seed': {}}
    for s in SEEDS:
        mlp = MLP([784, 64, 64, 10])
        mlp.load_state_dict(torch.load(os.path.join(CKPT, f'mlp_seed{s}.pt'),
                                       map_location='cpu'))
        mlp.eval()
        with torch.no_grad():
            preds = mlp(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:N_PER_SEED]
        t_ibp, t_kr, t_s2 = [], [], []
        for si in idx:
            x0 = x_te[si].flatten()
            c = int(y_te[si].item())
            for eps in EPS:
                t0 = time.perf_counter(); verify_ibp(mlp, x0, c, eps); t_ibp.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); verify_krawczyk_classical(mlp, x0, c, eps); t_kr.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); verify_safe2(mlp, x0, c, eps); t_s2.append(time.perf_counter() - t0)
        out['per_seed'][str(s)] = {'ibp_s': t_ibp, 'krawczyk_s': t_kr, 'safe2_s': t_s2}
        print(f'[seed {s}] n={len(idx)} '
              f'ibp {np.mean(t_ibp):.4f} kraw {np.mean(t_kr):.4f} safe2 {np.mean(t_s2):.4f}', flush=True)
    agg = {}
    for eng in ['ibp', 'krawczyk', 'safe2']:
        means = [np.mean(out['per_seed'][str(s)][eng + '_s']) for s in SEEDS]
        m = float(np.mean(means))
        pop = float(np.sqrt(np.mean([(x - m) ** 2 for x in means])))
        agg[eng] = {'per_seed_mean_s': [float(x) for x in means], 'mean_s': m, 'std_pop_s': pop}
        print(f'{eng}: per-seed {[round(x,4) for x in means]} mean {m:.4f} std {pop:.4f}', flush=True)
    out['aggregate'] = agg
    with open(os.path.join(BASE, 'results', 'mlp_runtime_ibp.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED results/mlp_runtime_ibp.json', flush=True)


if __name__ == '__main__':
    main()
