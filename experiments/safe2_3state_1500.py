# -*- coding: utf-8 -*-
"""Three-state SAFE2 statistics (robust / cert_vuln / undecided) on the
1500-sample CIFAR grid (3 seeds x 500 x 4 eps), to refresh the certification
cross-validation totals in Section 7."""
import os
import sys
import json

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
CKPT_DIR = os.path.join(BASE, 'data', 'checkpoints')
OUT_PATH = os.path.join(BASE, 'results', 'cnn_safe2_3state_1500.json')
sys.path.insert(0, VENDOR)

import verifier_cnn as vc                    # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402

INPUT_SHAPE = (3, 32, 32)
EPS_GRID = [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0, 8.0 / 255.0]
SEEDS = [0, 1, 2]
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8
MAX_ITER_R = 12
MAX_ITER_E = 12
N_SAMPLES = 500


def main():
    _, _, x_te, y_te = load_cifar10()
    out = {'per_seed_eps': {}, 'totals': {'robust': 0, 'exist': 0,
                                          'undecided': 0}}
    for s in SEEDS:
        model = make_cnn()
        model.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f'cnn_ibp_seed{s}.pt'), map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        print(f'[seed {s}] n_samples={len(idx)}', flush=True)
        for eps in EPS_GRID:
            cnt = {'robust': 0, 'exist': 0, 'undecided': 0}
            for si in idx:
                x0 = x_te[si].flatten()
                cc = int(y_te[si].item())
                r = vc.verify_safe2(
                    model, x0, cc, eps, INPUT_SHAPE, ALPHA_MIN, ALPHA_FACTOR,
                    MAX_ITER_R, MAX_ITER_E, clamp01=True)[0]
                cnt[r] = cnt.get(r, 0) + 1
            out['per_seed_eps'][f'{s}_{eps:.4f}'] = cnt
            for k in cnt:
                out['totals'][k] += cnt[k]
            print(f'[seed {s} eps {eps:.4f}] {cnt}', flush=True)
    print('TOTALS', json.dumps(out['totals']), flush=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('SAVED', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
