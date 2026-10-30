# -*- coding: utf-8 -*-
"""Train small convolutional ReLU networks on CIFAR-10 (pixels in [0,1]).

Two training regimes:
  - standard (natural) training: seeds {0,1,2};
  - IBP robust training: interval bound propagation training with a fixed
    training radius eps_train = 2/255 and loss
        L = CE(f(x), y) + beta * CE(worst_logits(x), y),
    where worst_logits replaces the true-class logit by its lower interval
    bound and the other logits by their upper bounds (standard IBP loss).

Downsampling is done by stride-2 convolutions (no pooling), so the network
is a pure ReLU + Linear/Conv2d architecture that the generic verifier
(verifier_cnn.py) supports natively.
"""
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Vendored copy of the SAFE2 training module for the merged-project joint
# protocol. Path constants are adapted to the merged layout: this file
# lives at <project>/experiments/vendor/train_cnn.py, so the project root
# is three directory levels up and datasets live under <project>/data.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, 'data', 'datasets')
RESULT_DIR = os.path.join(BASE, 'data')
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]
EPOCHS_STD = 25
EPOCHS_IBP = 30
BATCH = 128
LR = 1e-3
EPS_TRAIN = 2.0 / 255.0
EPS_WARMUP_EPOCHS = 10  # epsilon annealed 0 -> EPS_TRAIN over these epochs
BETA = 0.5


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
        nn.Linear(128, 10))


def load_cifar10():
    tf = transforms.Compose([transforms.ToTensor()])  # pixels in [0,1]
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    x_tr = torch.stack([train[i][0] for i in range(len(train))])
    y_tr = torch.tensor([train[i][1] for i in range(len(train))])
    x_te = torch.stack([test[i][0] for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    return x_tr, y_tr, x_te, y_te


def ibp_forward_batch(model, x, eps):
    """Batched sound interval forward pass over a box of radius eps (L_inf)."""
    xl = torch.clamp(x - eps, 0.0, 1.0)
    xh = torch.clamp(x + eps, 0.0, 1.0)
    for mod in model:
        if isinstance(mod, nn.Conv2d):
            Wp = torch.clamp(mod.weight, min=0.0)
            Wn = torch.clamp(mod.weight, max=0.0)
            yl = (F.conv2d(xl, Wp, mod.bias, mod.stride, mod.padding)
                  + F.conv2d(xh, Wn, None, mod.stride, mod.padding))
            yh = (F.conv2d(xh, Wp, mod.bias, mod.stride, mod.padding)
                  + F.conv2d(xl, Wn, None, mod.stride, mod.padding))
            xl, xh = yl, yh
        elif isinstance(mod, nn.ReLU):
            xl = torch.clamp(xl, min=0.0)
            xh = torch.clamp(xh, min=0.0)
        elif isinstance(mod, nn.Flatten):
            xl = xl.flatten(1)
            xh = xh.flatten(1)
        elif isinstance(mod, nn.Linear):
            Wp = torch.clamp(mod.weight, min=0.0)
            Wn = torch.clamp(mod.weight, max=0.0)
            yl = xl @ Wp.T + xh @ Wn.T + mod.bias
            yh = xh @ Wp.T + xl @ Wn.T + mod.bias
            xl, xh = yl, yh
    return xl, xh


def train_one(regime, seed):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_cifar10()
    model = make_cnn()
    model.cuda()
    x_tr, y_tr = x_tr.cuda(), y_tr.cuda()
    x_te, y_te = x_te.cuda(), y_te.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    epochs = EPOCHS_IBP if regime == 'ibp' else EPOCHS_STD
    n = x_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        eps_eff = EPS_TRAIN if regime != 'ibp' else \
            EPS_TRAIN * min(1.0, (ep + 1) / EPS_WARMUP_EPOCHS)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            if regime == 'ibp' and eps_eff > 0:
                lo, hi = ibp_forward_batch(model, xb, eps_eff)
                worst = hi.clone()
                for b in range(xb.shape[0]):
                    worst[b, yb[b]] = lo[b, yb[b]]
                loss = loss + BETA * lossf(worst, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f'[{regime} seed {seed}] epoch {ep + 1}/{epochs} '
                  f'loss {tot / n:.4f} test_acc {acc:.4f}', flush=True)
    path = os.path.join(CKPT_DIR, f'cnn_{regime}_seed{seed}.pt')
    torch.save(model.state_dict(), path)
    stats_path = os.path.join(RESULT_DIR, 'model_stats_cnn.csv')
    with open(stats_path, 'a') as f:
        f.write(f'{regime},{seed},{acc:.6f}\n')
    print(f'[{regime} seed {seed}] saved -> {path}, test_acc={acc:.4f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--regime', choices=['std', 'ibp', 'all'], default='all')
    ap.add_argument('--seed', type=int, default=-1)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if not os.path.exists(os.path.join(RESULT_DIR, 'model_stats_cnn.csv')):
        with open(os.path.join(RESULT_DIR, 'model_stats_cnn.csv'), 'w') as f:
            f.write('regime,seed,test_acc\n')
    regimes = ['std', 'ibp'] if args.regime == 'all' else [args.regime]
    seeds = SEEDS if args.seed < 0 else [args.seed]
    for reg in regimes:
        for s in seeds:
            path = os.path.join(CKPT_DIR, f'cnn_{reg}_seed{s}.pt')
            if not args.force and os.path.exists(path):
                print(f'[{reg} seed {s}] exists, skip (use --force to retrain)',
                      flush=True)
                continue
            train_one(reg, s)
