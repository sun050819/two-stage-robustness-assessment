# -*- coding: utf-8 -*-
"""Train a medium-scale ReLU MLP on MNIST with the IBP robust loss
(interval bound propagation regularizer), 3 seeds, for the medium-network
certification-rate experiment (Fig./table: medium network).

Loss: L = CE(nominal logits) + beta * CE(IBP lower-bound logits) on the
eps_train box, the standard IBP training objective. The interval forward
pass reuses the vendored verifier_lib.forward_bounds, which is
differentiable (clamp-based), so the regularizer is trainable by
backpropagation.

Checkpoints are written to data/checkpoints/mlp_medium_seed{seed}.pt.
"""
import os
import sys

import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
sys.path.insert(0, VENDOR)

from train import MLP, load_mnist             # noqa: E402
from verifier_lib import forward_bounds       # noqa: E402

DIMS = [784, 256, 256, 10]
SEEDS = [0, 1, 2]
EPOCHS = 20
BATCH = 128
LR = 1e-2
EPS_TRAIN = 0.1
BETA = 1.0
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')


def ibp_lower_logits(model, x, eps):
    """IBP lower-bound logits of the box [x-eps, x+eps] (batch)."""
    lo = torch.clamp(x - eps, 0.0, 1.0)
    hi = torch.clamp(x + eps, 0.0, 1.0)
    xs_lo, xs_hi = [], []
    for i in range(x.shape[0]):
        xl, xh, _ = forward_bounds(model, lo[i], hi[i])
        xs_lo.append(xl)
        xs_hi.append(xh)
    return torch.stack(xs_lo), torch.stack(xs_hi)


def train_one(seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_mnist()
    model = MLP(DIMS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss_ce = ce(out, yb)
            lo, _ = ibp_lower_logits(model, xb, EPS_TRAIN)
            loss = loss_ce + BETA * ce(lo, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        print(f'[medium seed {seed}] epoch {ep + 1}/{epochs} '
              f'loss {tot / n:.4f} test_acc {acc:.4f}', flush=True)
    path = os.path.join(CKPT_DIR, f'mlp_medium_seed{seed}.pt')
    torch.save(model.state_dict(), path)
    print(f'[medium seed {seed}] saved -> {path} (acc {acc:.4f})', flush=True)
    return acc


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    stats_path = os.path.join(BASE, 'data', 'model_stats_medium_mlp.csv')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('seed,test_acc\n')
        for s in args.seeds:
            acc = train_one(s, args.epochs)
            f.write(f'{s},{acc:.6f}\n')
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
