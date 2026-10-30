# -*- coding: utf-8 -*-
"""Quick smoke test for the joint protocol (1 sample each workload).

Path-independent: resolves the vendored modules and bundled checkpoints
relative to the project layout (see joint_protocol.py docstring).
"""
import os
import sys
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)
CKPT = os.path.join(BASE, '..', 'data', 'checkpoints')
CKPT = os.path.abspath(CKPT)

from train import MLP, load_mnist
import verifier_lib as vl
from train_cnn import make_cnn, load_cifar10
import verifier_cnn as vc

# MLP
x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
m = MLP([784, 64, 64, 10])
m.load_state_dict(torch.load(os.path.join(CKPT, 'mlp_seed0.pt'), map_location='cpu'))
m.eval()
x0, c = x_te[0], int(y_te[0].item())
print('mlp margin_at x0:', vl.margin_at(m, x0, c))
print('mlp verify_safe2(0.01):', vl.verify_safe2(m, x0, c, 0.01, 0.2, 0.8)[0])
print('mlp pgd:', vl.pgd_attack(m, x0, c, 0.01, steps=10, restarts=2))

# CNN
_, _, xte, yte = load_cifar10()
cn = make_cnn()
cn.load_state_dict(torch.load(os.path.join(CKPT, 'cnn_ibp_seed0.pt'), map_location='cpu'))
cn.eval()
preds = cn(xte.view(-1, *((3, 32, 32)))).argmax(1)
si = (preds == yte).nonzero(as_tuple=False).flatten()[0].item()
xc, cc = xte[si], int(yte[si].item())
print('cnn margin_at x0:', vc.margin_at(cn, xc.view(-1), cc, (3, 32, 32)))
print('cnn verify_safe2(2/255):', vc.verify_safe2(
    cn, xc.view(-1), cc, 2.0 / 255.0, (3, 32, 32), 0.2, 0.8, 12, 12, clamp01=True)[0])
print('SMOKE OK')
