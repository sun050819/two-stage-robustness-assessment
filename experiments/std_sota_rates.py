# -*- coding: utf-8 -*-
"""Certified robust rates on standard-trained CNNs: SAFE2 vs auto-LiRPA.

Additional analysis (Table 4 of the manuscript): train standard
(natural) CNNs on MNIST and CIFAR-10 (full sets, 3 seeds), then certify the
SAME samples with the SAME weights using:
  - IBP, classical Krawczyk, SAFE2-Krawczyk (self-implemented vendor verifier)
  - CROWN-IBP from the OFFICIAL auto-LiRPA library (same weights, same boxes)

Outputs results/std_rates_mnist.json and results/std_rates_cifar.json with
per-seed/eps/method robust counts, pooled rates, and McNemar SAFE2 vs
auto-LiRPA.

Run with the GPU-capable Python (3.11 env) so both the verifier and
auto-LiRPA use the GPU.
"""
import os
import sys
import json
import time
import copy
import argparse

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy import stats

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
DATA_DIR = os.path.join(BASE, 'data', 'datasets')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc                       # noqa: E402
import train_mnist_cnn as tmn                   # noqa: E402
from train_cnn import make_cnn, load_cifar10    # noqa: E402

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm  # noqa: E402

SEEDS = [0, 1, 2]
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12
TOL = 1e-6


def load_mnist_test():
    tf = transforms.Compose([transforms.ToTensor()])
    test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    x_te = torch.stack([test[i][0].squeeze(0) for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    return x_te, y_te


def build_model(kind):
    if kind == 'mnist_std':
        return tmn.make_mnist_cnn, (1, 28, 28)
    if kind == 'cifar_std':
        return make_cnn, (3, 32, 32)
    raise ValueError(kind)


def ckpt_path(kind, seed):
    prefix = 'mnist_cnn_std' if kind == 'mnist_std' else 'cnn_std'
    return os.path.join(CKPT_DIR, f'{prefix}_seed{seed}.pt')


def load_xte(kind):
    if kind == 'mnist_std':
        return load_mnist_test()
    return load_cifar10()[2:]


def make_auto_model(model, dummy, device):
    return BoundedModule(model, dummy, device=device)


def auto_robust_flags(model, x0_all, eps, c_all, shape, device, B=64):
    """Official auto-LiRPA CROWN-IBP robust flags over an (N, *) sample set.

    BoundedModule moves parameters to `device`, so it runs on a deep copy
    and leaves the caller's CPU model untouched (the vendor verifiers use
    the original CPU model afterwards).
    """
    dummy = torch.zeros(1, *shape)
    bm = BoundedModule(copy.deepcopy(model), dummy, device=device)
    flags = np.zeros(len(x0_all), dtype=bool)
    x0_all = x0_all.view(-1, *shape).to(device)
    c_all = c_all.to(device)
    for i in range(0, len(x0_all), B):
        xb = x0_all[i:i + B]
        cb = c_all[i:i + B]
        x_L = torch.clamp(xb - eps, 0.0, 1.0)
        x_U = torch.clamp(xb + eps, 0.0, 1.0)
        ptb = PerturbationLpNorm(norm=float('inf'), eps=eps, x_L=x_L, x_U=x_U)
        bt = BoundedTensor(xb, ptb)
        lb, ub = bm.compute_bounds(x=(bt,), method='crown-ibp')
        lb = lb.detach()
        ub = ub.detach()
        margin = lb.gather(1, cb.view(-1, 1)).squeeze(1)
        others = ub.clone()
        others.scatter_(1, cb.view(-1, 1), -float('inf'))
        best_other = others.max(dim=1).values
        flags[i:i + B] = (margin - best_other > TOL).cpu().numpy()
    return flags


def certify_workload(kind, out_path, n_samples, eps_grid, device):
    model_fn, shape = build_model(kind)
    x_te, y_te = load_xte(kind)
    x_te = x_te.to(device)
    y_te = y_te.to(device)
    out = {'workload': kind, 'n_samples_per_seed': n_samples,
           'eps_grid': [float(e) for e in eps_grid],
           'per_seed_eps': {}, 'pooled': {},
           'mcnemar_safe2_vs_crownibp': {}}
    for s in SEEDS:
        model = model_fn().to(device)
        model.load_state_dict(torch.load(ckpt_path(kind, s), map_location=device,
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *shape)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
        print(f'[{kind} seed {s}] n_samples={len(idx)}', flush=True)
        x_all = x_te[idx].contiguous()
        y_all = y_te[idx]
        for eps in eps_grid:
            cnt = {'ibp': 0, 'krawczyk': 0, 'safe2': 0, 'crown_ibp': 0}
            rt = {'ibp': [], 'krawczyk': [], 'safe2': [], 'crown_ibp': []}
            t0 = time.perf_counter()
            flags_al = auto_robust_flags(model, x_all, eps, y_all, shape, device)
            cnt['crown_ibp'] = int(flags_al.sum())
            rt['crown_ibp'] = [(time.perf_counter() - t0) / len(idx)]
            flags_s2 = np.zeros(len(idx), dtype=bool)
            for pos, si in enumerate(idx):
                x0 = x_te[si].flatten()
                c = int(y_te[si].item())
                r_ibp = vc.verify_ibp(model, x0, c, eps, shape, clamp01=True)
                r_kr = vc.verify_krawczyk_classical(model, x0, c, eps, shape, clamp01=True)
                r_s2 = vc.verify_safe2(model, x0, c, eps, shape, ALPHA_MIN,
                                       ALPHA_FACTOR, MAX_ITER_R, MAX_ITER_E,
                                       clamp01=True)[0]
                flags_s2[pos] = (r_s2 == 'robust')
                for m, r in (('ibp', r_ibp), ('krawczyk', r_kr)):
                    if r == 'robust':
                        cnt[m] += 1
                if (pos + 1) % 100 == 0:
                    print(f'[{kind} seed {s} eps {eps:.4f}] self {pos+1}/{len(idx)}',
                          flush=True)
            cnt['safe2'] = int(flags_s2.sum())
            rec = {m: {'robust': cnt[m], 'n': len(idx),
                       'rate': cnt[m] / len(idx),
                       't_mean_s': float(np.mean(rt[m]))} for m in cnt}
            out['per_seed_eps'][f'{s}_{eps}'] = rec
            with open(out_path, 'w', encoding='utf-8') as _f:
                json.dump(out, _f, indent=1)
            print(f'[{kind} seed {s} eps {eps:.4f}] ' +
                  ' '.join(f'{m}={cnt[m]}/{len(idx)}' for m in cnt), flush=True)
            key = str(float(eps))
            mc = out['mcnemar_safe2_vs_crownibp'].setdefault(key, {'b': 0, 'c': 0})
            mc['b'] += int((flags_s2 & ~flags_al).sum())
            mc['c'] += int((flags_al & ~flags_s2).sum())
    for eps in eps_grid:
        key = str(float(eps))
        pooled = {}
        for m in ('ibp', 'krawczyk', 'safe2', 'crown_ibp'):
            rates = [out['per_seed_eps'][f'{s}_{eps}'][m]['rate'] for s in SEEDS]
            pooled[m] = {'rate_mean': float(np.mean(rates)),
                         'rate_std': float(np.std(rates)),
                         'n_models': len(rates)}
        out['pooled'][key] = pooled
        mc = out['mcnemar_safe2_vs_crownibp'][key]
        b, c = mc['b'], mc['c']
        if b + c > 0:
            p = stats.binomtest(min(b, c), n=b + c, p=0.5).pvalue
            mc['p_exact'] = float(min(p, 1.0))
            mc['discordant_pairs'] = b + c
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', out_path, flush=True)
    print('POOLED', json.dumps(out['pooled'], default=str), flush=True)
    print('MCNEMAR', json.dumps(out['mcnemar_safe2_vs_crownibp']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['mnist_std', 'cifar_std'], required=True)
    ap.add_argument('--samples', type=int, default=600)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    if args.kind == 'mnist_std':
        out_path = os.path.join(BASE, 'results', 'std_rates_mnist.json')
        eps_grid = [0.01, 0.05, 0.1, 0.2]
    else:
        out_path = os.path.join(BASE, 'results', 'std_rates_cifar.json')
        eps_grid = [1.0 / 255, 2.0 / 255, 4.0 / 255, 8.0 / 255]
    certify_workload(args.kind, out_path, args.samples, eps_grid, device)
