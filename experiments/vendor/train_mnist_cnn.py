# -*- coding: utf-8 -*-
"""Train a standard (natural) small CNN on MNIST, full 60k training set.

Same architecture family as the CIFAR-10 CNN in train_cnn.py (stride-2
convolutions, no pooling, pure ReLU + Conv/Linear so the generic verifier
verifier_cnn.py supports it natively), but single-channel 28x28.

Regime: standard cross-entropy training (no IBP term), seeds {0,1,2},
25 epochs, full training set. Checkpoints -> data/checkpoints/mnist_cnn_std_seed{N}.pt
"""
import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE, 'data', 'datasets')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]
EPOCHS = 25
BATCH = 128
LR = 1e-3


def make_mnist_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
        nn.Linear(128, 10))


def load_mnist_full():
    tf = transforms.Compose([transforms.ToTensor()])  # pixels in [0,1]
    train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    x_tr = torch.stack([train[i][0] for i in range(len(train))])
    y_tr = torch.tensor([train[i][1] for i in range(len(train))])
    x_te = torch.stack([test[i][0] for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    return x_tr, y_tr, x_te, y_te


def train_one(seed, device):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_mnist_full()
    model = make_mnist_cnn().to(device)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        if (ep + 1) % 5 == 0 or ep == EPOCHS:
            print(f'[mnist_std seed {seed}] epoch {ep + 1}/{EPOCHS} '
                  f'loss {tot / n:.4f} test_acc {acc:.4f}', flush=True)
    path = os.path.join(CKPT_DIR, f'mnist_cnn_std_seed{seed}.pt')
    torch.save(model.state_dict(), path)
    stats_path = os.path.join(BASE, 'data', 'model_stats_mnist_cnn.csv')
    with open(stats_path, 'a') as f:
        f.write(f'{seed},{acc:.6f}\n')
    print(f'[mnist_std seed {seed}] saved -> {path}, test_acc={acc:.4f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=-1)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    if not os.path.exists(os.path.join(BASE, 'data', 'model_stats_mnist_cnn.csv')):
        with open(os.path.join(BASE, 'data', 'model_stats_mnist_cnn.csv'), 'w') as f:
            f.write('seed,test_acc\n')
    seeds = SEEDS if args.seed < 0 else [args.seed]
    for s in seeds:
        path = os.path.join(CKPT_DIR, f'mnist_cnn_std_seed{s}.pt')
        if not args.force and os.path.exists(path):
            print(f'[mnist_std seed {s}] exists, skip (use --force to retrain)', flush=True)
            continue
        train_one(s, device)
