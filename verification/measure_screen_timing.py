# -*- coding: utf-8 -*-
"""Measure the end-to-end cost of one 50-forward screen on one input
(MLP on MNIST at eps=0.01, CNN on CIFAR-10 at eps=2/255), median over
REPS repetitions. This is the "measured end-to-end 50-forward screen"
instantiation of c_s(50) in the cost model (Table cost), complementing
the pure-forward numbers in forward_timing.json.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "experiments" / "vendor"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "experiments"))

from train import MLP, load_mnist            # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402
from joint_protocol import screen_rs         # noqa: E402

REPS = 30
OUT = ROOT / "verification" / "screen_timing.json"


def median_screen_time(model, x0, c, eps, reps=REPS, **kw):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        screen_rs(model, x0, c, eps, **kw)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    torch.set_num_threads(4)
    mlp = MLP([784, 64, 64, 10])
    mlp.load_state_dict(torch.load(ROOT / "data" / "checkpoints" / "mlp_seed0.pt",
                                   map_location="cpu"))
    mlp.eval()
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    mlp_c = int(y_te[0].item())
    mlp_t = median_screen_time(mlp, x_te[0].flatten(), mlp_c, 0.01)

    cnn = make_cnn()
    cnn.load_state_dict(torch.load(ROOT / "data" / "checkpoints" / "cnn_ibp_seed0.pt",
                                   map_location="cpu"))
    cnn.eval()
    _, _, x_c, y_c = load_cifar10()
    cnn_c = int(y_c[0].item())
    cnn_t = median_screen_time(cnn, x_c[0], cnn_c, 2.0 / 255.0,
                               input_shape=(3, 32, 32), clamp=True)

    res = {
        "mlp_c_s(50)_end_to_end_s": mlp_t,
        "cnn_c_s(50)_end_to_end_s": cnn_t,
        "reps": REPS,
        "budget_per_screen": 50,
        "note": "median wall time of one full screen_rs call "
                "(50 forward passes + sampling + margin checks) on one input",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("mlp end-to-end c_s(50) =", round(mlp_t * 1000.0, 3), "ms")
    print("cnn end-to-end c_s(50) =", round(cnn_t * 1000.0, 3), "ms")
    print("SAVED", OUT)


if __name__ == "__main__":
    main()
