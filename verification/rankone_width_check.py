# -*- coding: utf-8 -*-
"""Numeric verification of the rank-one projection claim (Section 5.2).

Claim: with C the d x 1 Moore-Penrose preconditioner (C = g/(||g||^2)), the
interval-Newton image K1(X) satisfies P_{C_perp} K1(X) = P_{C_perp} X, so no
direction orthogonal to C can shrink: width(v.K1)/width(v.X) ~= 1 for every
v perp C, while along C a shrink is possible.

We measure, on the certification-stage MNIST MLP (seed 0, eps=0.01, 30
correctly classified samples x 50 random orthonormal directions perp C):
  - mean / min / max width-retention ratio in C_perp,
  - fraction of C_perp directions with ratio < 1 (outward-rounding excluded),
  - the ratio along C itself.
Uses the exact operators of verifier_lib (same box, hull, and K1 as the paper).
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "experiments" / "vendor"
sys.path.insert(0, str(VENDOR))

from train import MLP, load_mnist  # noqa: E402
import verifier_lib as vl  # noqa: E402

torch.manual_seed(0)
CKPT = ROOT / "data" / "checkpoints" / "mlp_seed0.pt"
EPS = 0.01
N_SAMPLES = 30
N_DIRS = 50


def interval_dot_width(v, lo, hi):
    """Width of the interval dot product v . [lo, hi] (sign-split)."""
    pos = v >= 0
    neg = ~pos
    lo_v = torch.zeros(lo.shape, dtype=lo.dtype)
    hi_v = torch.zeros(hi.shape, dtype=hi.dtype)
    lo_v[pos] = lo[pos]
    lo_v[neg] = hi[neg]
    hi_v[pos] = hi[pos]
    hi_v[neg] = lo[neg]
    return float((hi_v * v).sum() - (lo_v * v).sum())


def main():
    model = MLP([784, 64, 64, 10])
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]

    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero().flatten()[:N_SAMPLES].tolist()

    mean_ret = []
    min_ret = []
    max_ret = []
    below_one = 0
    total = 0
    c_ratios = []
    c_shrink = 0
    rng = np.random.default_rng(42)

    for si, i in enumerate(idx):
        x0 = x_te[i]
        c = int(y_te[i].item())
        X_lo, X_hi = vl._outward(x0 - EPS, x0 + EPS)
        m = (X_lo + X_hi) / 2.0
        ops = vl.forward_bounds(model, X_lo, X_hi)[2]
        g_m = vl.margin_at(model, m, c)
        if g_m <= 0:
            continue  # midpoint already a witness; no K1 to measure
        J_lo, J_hi = vl.gradient_margin_hull(model, X_lo, X_hi, c, ops)
        J_mid = (J_lo + J_hi) / 2.0
        C = vl.moore_penrose(J_mid)
        if C.norm().item() < 1e-9:
            continue
        K_lo, K_hi = vl.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)

        Cn = C.detach().numpy()
        cnorm2 = float(Cn @ Cn)
        # orthonormal random directions perp C
        vs = []
        for _ in range(N_DIRS):
            v = rng.standard_normal(784)
            v = v - Cn * (v @ Cn) / cnorm2
            n = np.linalg.norm(v)
            if n < 1e-8:
                continue
            vs.append(torch.tensor(v / n, dtype=torch.float64))
        wX = [interval_dot_width(v, X_lo.double(), X_hi.double()) for v in vs]
        wK = [interval_dot_width(v, K_lo.double(), K_hi.double()) for v in vs]
        for wx, wk in zip(wX, wK):
            r = wk / wx if wx > 0 else float("nan")
            mean_ret.append(r)
            if r < 1.0:
                below_one += 1
            total += 1
        min_ret.append(min(wk / wx for wx, wk in zip(wX, wK) if wx > 0))
        max_ret.append(max(wk / wx for wx, wk in zip(wX, wK) if wx > 0))
        # along C
        w = C.detach() / C.detach().norm()
        wxc = interval_dot_width(w.double(), X_lo.double(), X_hi.double())
        wkc = interval_dot_width(w.double(), K_lo.double(), K_hi.double())
        rc = wkc / wxc if wxc > 0 else float("nan")
        c_ratios.append(rc)
        if rc < 1.0:
            c_shrink += 1

    res = {
        "model": "MLP 784-64-64-10 seed0",
        "eps": EPS,
        "boxes": total // N_DIRS,
        "directions_per_box": N_DIRS,
        "C_perp_retention_mean": float(np.mean(mean_ret)),
        "C_perp_retention_min": float(np.min(mean_ret)),
        "C_perp_retention_max": float(np.max(mean_ret)),
        "per_box_min": [float(x) for x in min_ret],
        "per_box_max": [float(x) for x in max_ret],
        "frac_below_one": below_one / total,
        "C_parallel_ratio_mean": float(np.mean(c_ratios)),
        "C_parallel_shrink_fraction": c_shrink / len(c_ratios),
    }
    out = ROOT / "verification" / "rankone_width_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
