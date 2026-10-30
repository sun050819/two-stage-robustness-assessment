# -*- coding: utf-8 -*-
"""Compute SAFE2-vs-Krawczyk and SAFE2-vs-IBP discordant pairs at 2/255
on the 1500 CIFAR samples, and merge into results/cnn_rates_1500.json."""
import os
import sys
import json

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
OUT_PATH = os.path.join(BASE, 'results', 'cnn_rates_1500.json')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc                    # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402

INPUT_SHAPE = (3, 32, 32)
EPS = 2.0 / 255.0
SEEDS = [0, 1, 2]
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12
N_SAMPLES = 500


def main():
    _, _, x_te, y_te = load_cifar10()
    acc = {'safe2_vs_kraw': {'b': 0, 'c': 0},
           'safe2_vs_ibp': {'b': 0, 'c': 0}}
    for s in SEEDS:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        print(f'[seed {s}] n_samples={len(idx)}', flush=True)
        for si in idx:
            x0 = x_te[si].flatten()
            cc = int(y_te[si].item())
            r_ibp = vc.verify_ibp(model, x0, cc, EPS, INPUT_SHAPE, clamp01=True)
            r_kr = vc.verify_krawczyk_classical(
                model, x0, cc, EPS, INPUT_SHAPE, clamp01=True)
            r_s2 = vc.verify_safe2(
                model, x0, cc, EPS, INPUT_SHAPE, ALPHA_MIN, ALPHA_FACTOR,
                MAX_ITER_R, MAX_ITER_E, clamp01=True)[0]
            if r_s2 == 'robust' and r_kr != 'robust':
                acc['safe2_vs_kraw']['b'] += 1
            if r_kr == 'robust' and r_s2 != 'robust':
                acc['safe2_vs_kraw']['c'] += 1
            if r_s2 == 'robust' and r_ibp != 'robust':
                acc['safe2_vs_ibp']['b'] += 1
            if r_ibp == 'robust' and r_s2 != 'robust':
                acc['safe2_vs_ibp']['c'] += 1
    print('RESULT', json.dumps(acc), flush=True)
    with open(OUT_PATH, encoding='utf-8') as f:
        out = json.load(f)
    out['mcnemar_eps2_1500'] = acc
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('MERGED', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
