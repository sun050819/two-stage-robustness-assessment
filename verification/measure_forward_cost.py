# -*- coding: utf-8 -*-
"""Measure per-forward-pass wall time (CPU, batch 1) for the two certification
workloads, to instantiate c_s(N) = N x forward-time in the cost model."""
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "experiments" / "vendor"
sys.path.insert(0, str(VENDOR))
from train import MLP  # noqa: E402
from train_cnn import make_cnn  # noqa: E402

REPS = 200
OUT = ROOT / "verification" / "forward_timing.json"


def bench(model, x, reps=REPS):
    with torch.no_grad():
        for _ in range(10):  # warmup
            model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / reps * 1000.0  # ms


def main():
    torch.set_num_threads(4)
    mlp = MLP([784, 64, 64, 10])
    mlp.load_state_dict(torch.load(ROOT / "data" / "checkpoints" / "mlp_seed0.pt", map_location="cpu"))
    cnn = make_cnn()
    cnn.load_state_dict(torch.load(ROOT / "data" / "checkpoints" / "cnn_ibp_seed0.pt", map_location="cpu"))
    mlp.eval()
    cnn.eval()
    x_mlp = torch.randn(1, 784)
    x_cnn = torch.randn(1, 3, 32, 32)
    res = {
        "mlp_784_64_64_10_ms_per_forward": bench(mlp, x_mlp),
        "cnn_cifar_ms_per_forward": bench(cnn, x_cnn),
        "mlp_c_s(50)_s": bench(mlp, x_mlp) * 50 / 1000.0,
        "cnn_c_s(50)_s": bench(cnn, x_cnn) * 50 / 1000.0,
        "reps": REPS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        import json
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(res)


if __name__ == "__main__":
    main()
