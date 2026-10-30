# -*- coding: utf-8 -*-
"""Re-measure per-sample verification time on the IBP-trained CIFAR-10 CNNs
(3 seeds) for Table tab:runtime, matching the MNIST runtime_seed*.csv protocol:
time IBP / classical Krawczyk / SAFE2-Krawczyk over the eps grid 1/255..8/255
and report mean + std across seeds (population std, as used in the tables).
"""
import os, sys, json, time
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT = os.path.join(BASE, 'data', 'checkpoints')
sys.path.insert(0, VENDOR)

from verifier_cnn import verify_ibp, verify_krawczyk_classical, verify_safe2, _modules, forward_logits  # noqa
from train_cnn import make_cnn, load_cifar10  # noqa

SEEDS = [0, 1, 2]
EPS = [1/255, 2/255, 4/255, 8/255]
N_PER_SEED = 50
INPUT_SHAPE = (3, 32, 32)

def main():
    _, _, x_te, y_te = load_cifar10()
    out = {'n_per_seed': N_PER_SEED, 'eps_grid': EPS, 'per_seed': {}}
    for s in SEEDS:
        cnn = make_cnn()
        cnn.load_state_dict(torch.load(os.path.join(CKPT, f'cnn_ibp_seed{s}.pt'),
                                       map_location='cpu'))
        cnn.eval()
        with torch.no_grad():
            preds = cnn(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:N_PER_SEED]
        t_ibp, t_kr, t_s2 = [], [], []
        for si in idx:
            x0 = x_te[si].flatten()
            c = int(y_te[si].item())
            for eps in EPS:
                t0 = time.perf_counter(); verify_ibp(cnn, x0, c, eps, INPUT_SHAPE, clamp01=True); t_ibp.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); verify_krawczyk_classical(cnn, x0, c, eps, INPUT_SHAPE, clamp01=True); t_kr.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); verify_safe2(cnn, x0, c, eps, INPUT_SHAPE, 0.2, 0.8, 12, 12, clamp01=True); t_s2.append(time.perf_counter() - t0)
            print(f'[seed {s}] n={len(idx)}', flush=True)
        out['per_seed'][str(s)] = {'ibp_s': t_ibp, 'krawczyk_s': t_kr, 'safe2_s': t_s2}
        print(f'  ibp {sum(t_ibp)/len(t_ibp):.4f}  kraw {sum(t_kr)/len(t_kr):.4f}  safe2 {sum(t_s2)/len(t_s2):.4f}', flush=True)
    # aggregate per engine: per-seed means -> mean, population std
    agg = {}
    for eng in ['ibp', 'krawczyk', 'safe2']:
        means = [sum(out['per_seed'][str(s)][eng + '_s']) / len(out['per_seed'][str(s)][eng + '_s']) for s in SEEDS]
        m = sum(means) / len(means)
        pop = (sum((x - m) ** 2 for x in means) / len(means)) ** 0.5
        agg[eng] = {'per_seed_mean_s': means, 'mean_s': m, 'std_pop_s': pop}
        print(f'{eng}: per-seed {[round(x,4) for x in means]} mean {m:.4f} std {pop:.4f}', flush=True)
    out['aggregate'] = agg
    with open(os.path.join(BASE, 'results', 'cnn_runtime_ibp.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED results/cnn_runtime_ibp.json', flush=True)

if __name__ == '__main__':
    main()
