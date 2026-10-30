# -*- coding: utf-8 -*-
"""Locate the SAFE2 existence witnesses on the 1500 CIFAR grid and record
their margins (seed 1, eps in {4/255, 8/255})."""
import os
import sys
import json

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
OUT_PATH = os.path.join(BASE, 'results', 'cnn_exist_witnesses_1500.json')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc                    # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402

INPUT_SHAPE = (3, 32, 32)
SEEDS = [1]
EPS_GRID = [4.0 / 255.0, 8.0 / 255.0]
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12
N_SAMPLES = 500


def main():
    _, _, x_te, y_te = load_cifar10()
    witnesses = []
    for s in SEEDS:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        for eps in EPS_GRID:
            for si in idx:
                x0 = x_te[si].flatten()
                cc = int(y_te[si].item())
                res = vc.verify_safe2(
                    model, x0, cc, eps, INPUT_SHAPE, ALPHA_MIN, ALPHA_FACTOR,
                    MAX_ITER_R, MAX_ITER_E, clamp01=True)
                if res[0] == 'exist':
                    # res = (verdict, it, width_hist, m, g_m) or
                    #       (verdict, it, width_hist, root, root_margin)
                    margin = float(res[4])
                    witnesses.append({'seed': s, 'eps': eps,
                                      'sample': int(si), 'class': cc,
                                      'margin': margin})
                    print(f'exist seed {s} eps {eps:.4f} sample {int(si)} '
                          f'margin {margin:.4f}', flush=True)
    print('N', len(witnesses), flush=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(witnesses, f, indent=1)
    print('SAVED', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
