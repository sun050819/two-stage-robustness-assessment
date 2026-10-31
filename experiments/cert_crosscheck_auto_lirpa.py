# -*- coding: utf-8 -*-
"""Official auto-LiRPA cross-check of the self-implemented CROWN-IBP column.

Additional analysis (Table 7 of the paper): run the OFFICIAL
auto-LiRPA CROWN-IBP bound on the SAME weights (existing IBP-trained
checkpoints) and the SAME correctly-classified sample sets used by the
self-implemented CROWN-IBP runs, then compare the two engines side by side.

Workloads:
  - mlp_ibp: MNIST MLP [784,64,64,10], mlp_seed{0,1,2}.pt, 200/seed,
    eps {0.01,0.02,0.05,0.1,0.15,0.2}   (vs results/crown_ibp_mnist_600.json)
  - cnn_ibp: CIFAR-10 CNN, cnn_ibp_seed{0,1,2}.pt, 500/seed,
    eps {1/255,2/255,4/255,8/255}       (vs results/crown_ibp_1500.json)

Outputs results/crosscheck_auto_lirpa.json with official rates and the
per-radius pooled comparison against the self-implemented column.
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

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
DATA_DIR = os.path.join(BASE, 'data', 'datasets')
sys.path.insert(0, VENDOR)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm  # noqa: E402

SEEDS = [0, 1, 2]
TOL = 1e-6


def load_mnist_test(flatten):
    tf = transforms.Compose([transforms.ToTensor()] +
                            ([transforms.Lambda(lambda t: t.view(-1))] if flatten else []))
    test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    x_te = torch.stack([test[i][0].squeeze(0) for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    return x_te, y_te


def build_mlp(dims=(784, 64, 64, 10)):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def build_model(kind):
    if kind == 'mlp_ibp':
        from train import MLP
        return lambda: MLP([784, 64, 64, 10]), (784,)
    if kind == 'cnn_ibp':
        from train_cnn import make_cnn, load_cifar10
        return make_cnn, (3, 32, 32)
    raise ValueError(kind)


def ckpt_path(kind, seed):
    prefix = 'mlp' if kind == 'mlp_ibp' else 'cnn_ibp'
    return os.path.join(CKPT_DIR, f'{prefix}_seed{seed}.pt')


def load_xte(kind):
    if kind == 'mlp_ibp':
        # IMPORTANT: replicate the EXACT test subset used by the
        # self-implemented CROWN-IBP column (crown_ibp_rerun_mnist600.py):
        # load_mnist(n_train=8000, n_test=300) with the fixed generator,
        # so both engines certify the SAME samples. Using the full 10000
        # test set here would compare different samples.
        from train import load_mnist
        return load_mnist(n_train=8000, n_test=300)[2:]
    from train_cnn import load_cifar10
    return load_cifar10()[2:]


def official_flags(model, x0_all, eps, c_all, shape, device, B=64, clamp=True,
                   bound='crown-ibp'):
    bm = BoundedModule(copy.deepcopy(model), torch.zeros(1, *shape), device=device)
    flags = np.zeros(len(x0_all), dtype=bool)
    x0_all = x0_all.to(device)
    c_all = c_all.to(device)
    for i in range(0, len(x0_all), B):
        xb = x0_all[i:i + B]
        cb = c_all[i:i + B]
        if clamp:
            x_L = torch.clamp(xb - eps, 0.0, 1.0)
            x_U = torch.clamp(xb + eps, 0.0, 1.0)
        else:
            x_L = xb - eps
            x_U = xb + eps
        ptb = PerturbationLpNorm(norm=float('inf'), eps=eps, x_L=x_L, x_U=x_U)
        lb, ub = bm.compute_bounds(x=(BoundedTensor(xb, ptb),), method=bound)
        lb = lb.detach()
        ub = ub.detach()
        margin = lb.gather(1, cb.view(-1, 1)).squeeze(1)
        others = ub.clone()
        others.scatter_(1, cb.view(-1, 1), -float('inf'))
        flags[i:i + B] = (margin - others.max(dim=1).values > TOL).cpu().numpy()
    return flags


def main(kind, n_samples, eps_grid, eps_labels, clamp=True, bound='crown-ibp'):
    suffix = '' if bound == 'crown-ibp' else f'_{bound}'
    out_path = os.path.join(BASE, 'results',
                            f'crosscheck_{kind}{"" if clamp else "_unclamped"}{suffix}.json')
    model_fn, shape = build_model(kind)
    x_te, y_te = load_xte(kind)
    out = {'workload': kind, 'n_per_seed': n_samples, 'clamp_box': clamp,
           'bound': bound, 'eps_grid': eps_labels, 'per_seed_eps': {}, 'pooled': {}}
    for s in SEEDS:
        model = model_fn()
        model.load_state_dict(torch.load(ckpt_path(kind, s), map_location='cpu',
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
        print(f'[{kind} seed {s}] n_samples={len(idx)}', flush=True)
        x_all = x_te[idx].contiguous()
        y_all = y_te[idx]
        for eps, label in zip(eps_grid, eps_labels):
            t0 = time.perf_counter()
            flags = official_flags(model, x_all, eps, y_all, shape, 'cuda', clamp=clamp,
                                   bound=bound)
            robust = int(flags.sum())
            el = time.perf_counter() - t0
            out['per_seed_eps'][f'{s}_{label}'] = {'robust': robust, 'n': len(idx),
                                                   'rate': robust / len(idx),
                                                   't_s': float(el)}
            with open(out_path, 'w', encoding='utf-8') as _f:
                json.dump(out, _f, indent=1)
            print(f'[{kind} seed {s} eps {label}] official robust={robust}/{len(idx)} '
                  f'({el:.0f}s)', flush=True)
    for eps, label in zip(eps_grid, eps_labels):
        rates = [out['per_seed_eps'][f'{s}_{label}']['rate'] for s in SEEDS]
        out['pooled'][label] = {'rate_mean': float(np.mean(rates)),
                                'rate_std': float(np.std(rates)),
                                'n_models': len(rates)}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', out_path, flush=True)
    print('POOLED', json.dumps(out['pooled']), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['mlp_ibp', 'cnn_ibp'], required=True)
    ap.add_argument('--bound', choices=['crown-ibp', 'crown', 'ibp'],
                    default='crown-ibp', help='official auto-LiRPA bound type')
    args = ap.parse_args()
    if args.kind == 'mlp_ibp':
        # MNIST MLP: use the same UN-CLAMPED box as the self-implemented
        # column of Table 5 (supplementary sup:crown), so the comparison is
        # apples-to-apples on identical boxes.
        main(args.kind, 200, [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
             ['0.01', '0.02', '0.05', '0.1', '0.15', '0.2'], clamp=False,
             bound=args.bound)
    else:
        # CIFAR CNN: self-implemented column uses the clamped box; keep clamp.
        main(args.kind, 500, [1.0 / 255, 2.0 / 255, 4.0 / 255, 8.0 / 255],
             ['1/255', '2/255', '4/255', '8/255'], clamp=True,
             bound=args.bound)
