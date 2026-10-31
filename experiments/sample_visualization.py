# -*- coding: utf-8 -*-
"""Qualitative sample visualisation (Figure 11 of the paper).

Panel A  Two-moons: a nontrivial verification box (midpoint robust, corner
          violating) with the adversarial witness corner marked.
Panel B  MNIST: three clean samples and the adversarial examples found by
          the joint protocol's multi-restart PGD (10 steps x 2 restarts,
          eps = 0.01) on the seed-0 IBP-trained MLP.
Panel C  CIFAR-10: the six existence witnesses reported in
          results/cnn_exist_witnesses_1500.json, reproduced with
          verifier_cnn.verify_safe2 (seed-1 CNN), clean image vs. witness.

Writes paper/figures/fig11_samples.pdf.
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
SRC_CODE = os.path.join(BASE, "sources", "SAFE2_NN_Verification", "code")
CKPT_DIR = os.path.join(BASE, "data", "checkpoints")
FIG_DIR = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, VENDOR)
sys.path.insert(0, SRC_CODE)

LOG = os.path.join(BASE, "results", "fig11_samples.log")


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def main():
    if os.path.exists(LOG):
        os.remove(LOG)

    # ---------- Panel A: two-moons nontrivial box with witness corner ----------
    import existence_experiment as ee
    Xtr, ytr, Xte, yte = ee.make_data(ee.SEED)
    model = ee.TwoMoonNet()
    model.load_state_dict(torch.load(
        os.path.join(BASE, "sources", "SAFE2_NN_Verification", "results",
                     "existence_model.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    log(f"[A] two-moons test acc {acc:.4f}")

    rows = list(__import__("csv").DictReader(open(
        os.path.join(BASE, "sources", "SAFE2_NN_Verification", "results",
                     "existence_results.csv"), encoding="utf-8")))
    pick = None
    for r in rows:
        if r["eps"] == "0.5" and r["nontrivial"] == "1" and r["safe2"] == "exist":
            pick = r
            break
    assert pick is not None, "no nontrivial exist sample at eps=0.5"
    s_idx = int(pick["sample"])
    eps_a = float(pick["eps"])
    x0 = Xte[s_idx]
    c0 = int(yte[s_idx].item())
    log(f"[A] sample {s_idx} class {c0} eps {eps_a} "
        f"mid_margin {pick['mid_margin']} corner_min {pick['corner_min']}")

    corner_m = {}
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            pt = torch.tensor([x0[0] + sx * eps_a, x0[1] + sy * eps_a])
            corner_m[(sx, sy)] = ee.margin_at(model, pt, c0)
    worst = min(corner_m, key=corner_m.get)
    log(f"[A] corner margins {corner_m} worst {worst} "
        f"margin {corner_m[worst]:.4f}")

    # ---------- Panel B: MNIST clean vs. PGD ----------
    from train import MLP
    x_te, y_te = load_mnist_offline()
    mlp = MLP([784, 64, 64, 10])
    mlp.load_state_dict(torch.load(os.path.join(CKPT_DIR, "mlp_seed0.pt"),
                                   map_location="cpu"))
    mlp.eval()
    with torch.no_grad():
        preds = mlp(x_te).argmax(1)
    mn_idx = (preds == y_te).nonzero(as_tuple=False).flatten().tolist()[:3]
    mn_pairs = []
    for si in mn_idx:
        xm = x_te[si]
        cc = int(y_te[si].item())
        found, adv = pgd_mlp(mlp, xm, cc, 0.01)
        mn_pairs.append((xm, adv, cc, found))
        log(f"[B] mnist sample {si} class {cc} pgd_broken {found}")

    # ---------- Panel C: CIFAR existence witnesses ----------
    import verifier_cnn as vc
    from train_cnn import make_cnn
    INPUT_SHAPE = (3, 32, 32)
    _, _, cx_te, cy_te = load_cifar_offline()
    cnn = make_cnn()
    cnn.load_state_dict(torch.load(os.path.join(CKPT_DIR, "cnn_ibp_seed1.pt"),
                                   map_location="cpu"))
    cnn.eval()
    wit = json.load(open(os.path.join(BASE, "results",
                                      "cnn_exist_witnesses_1500.json"),
                         encoding="utf-8"))
    cf_pairs = []
    for w in wit:
        x0c = cx_te[w["sample"]].flatten()
        res = vc.verify_safe2(cnn, x0c, w["class"], w["eps"], INPUT_SHAPE,
                              0.2, 0.8, 12, 12, clamp01=True)
        verdict = res[0]
        root = res[3] if len(res) > 3 else None
        margin = float(res[4]) if len(res) > 4 else None
        cf_pairs.append((w, x0c, root, verdict, margin))
        log(f"[C] sample {w['sample']} class {w['class']} eps {w['eps']:.4f} "
            f"verdict {verdict} margin {margin:.6f}")

    # ---------- Render ----------
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 8,
                         "axes.titlesize": 8, "figure.constrained_layout.use": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.9, 8.0))

    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.85, 1.4], hspace=0.32)

    # Panel A
    axA = fig.add_subplot(gs[0])
    axA.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=3, c="#4c72b0",
                alpha=0.5, label="class 0", rasterized=True)
    axA.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=3, c="#2f6f4f",
                alpha=0.5, label="class 1", rasterized=True)
    rect = plt.Rectangle((x0[0] - eps_a, x0[1] - eps_a), 2 * eps_a, 2 * eps_a,
                         fill=False, edgecolor="#c44e52", lw=1.4,
                         label=f"box $\\varepsilon={eps_a:g}$")
    axA.add_patch(rect)
    axA.scatter([x0[0]], [x0[1]], s=28, c="k", marker="+", zorder=5,
                label=f"midpoint $g={float(pick['mid_margin']):+.3f}$")
    wx = x0[0] + worst[0] * eps_a
    wy = x0[1] + worst[1] * eps_a
    axA.scatter([wx], [wy], s=42, c="#c44e52", marker="*", zorder=6,
                label=f"witness $g={corner_m[worst]:.3f}\\leq 0$")
    axA.set_xlabel("$x_1$")
    axA.set_ylabel("$x_2$")
    axA.set_title(f"(a) two-moons: nontrivial box, sample {s_idx}, "
                  f"class {c0} (adversarial witness corner)", loc="left")
    # Legend outside the data region (right of the panel) so it never
    # covers scatter points.
    axA.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6,
               framealpha=0.9)

    # Panel B
    n = len(mn_pairs)
    gsB = gs[1].subgridspec(2, n, hspace=0.25, wspace=0.12)
    for j, (xm, adv, cc, found) in enumerate(mn_pairs):
        for k, (img, tag) in enumerate([(xm, "clean"),
                                        (adv if found else xm, "adversarial")]):
            axi = fig.add_subplot(gsB[k, j])
            axi.imshow(img.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axi.set_xticks([]); axi.set_yticks([])
            axi.set_title(tag if j == 0 else "", fontsize=6)
            if k == 1:
                axi.set_xlabel(f"class {cc}\n"
                               + ("PGD: none found" if not found else
                                  "adversarial"), fontsize=5.5)
    broken = sum(1 for _, _, _, f in mn_pairs if f)
    if broken == 0:
        cap_b = ("(b) MNIST MLP, $\\varepsilon=0.01$, multi-restart PGD "
                 "(10 steps $\\times$ 2 restarts): PGD breaks none of the "
                 "three shown samples, consistent with the 0/102 result in "
                 "Table 13.")
    else:
        cap_b = ("(b) MNIST MLP, $\\varepsilon=0.01$, multi-restart PGD "
                 "(10 steps $\\times$ 2 restarts): adversarial examples "
                 "found on {broken} of the three shown samples."
                 ).replace("{broken}", str(broken))
    fig.text(0.02, 0.545, cap_b, fontsize=6.5, va="bottom")

    # Panel C
    gsC = gs[2].subgridspec(2, 6, hspace=0.28, wspace=0.10)
    for j, (w, x0c, root, verdict, margin) in enumerate(cf_pairs):
        for k, (img, tag) in enumerate([(x0c, "clean"), (root, "witness")]):
            axi = fig.add_subplot(gsC[k, j])
            imgp = img.detach().cpu().numpy().reshape(*INPUT_SHAPE).transpose(1, 2, 0)
            axi.imshow(imgp, vmin=0, vmax=1)
            axi.set_xticks([]); axi.set_yticks([])
            axi.set_title(tag if j == 0 else "", fontsize=6)
            if k == 1:
                axi.set_xlabel(f"class {w['class']}\n"
                               f"$\\varepsilon={w['eps']*255:.0f}/255$",
                               fontsize=5.5)
    fig.text(0.02, 0.235, "(c) CIFAR-10 CNN (seed 1): the six existence "
             "witnesses of Table 13 (all with margin $\\leq 0$), clean image "
             "and the extracted witness point.", fontsize=6.5, va="bottom")

    out = os.path.join(FIG_DIR, "fig11_samples.pdf")
    fig.savefig(out, bbox_inches="tight")
    log(f"SAVED {out}")


def load_mnist_offline(n_test=300):
    """Load MNIST test images/labels directly from the raw idx files
    (offline, same file order torchvision uses; generator seed 0 matches
    the vendored load_mnist subsample)."""
    import struct
    base = os.path.join(BASE, "data", "datasets", "MNIST", "raw")
    with open(os.path.join(base, "t10k-images-idx3-ubyte"), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    with open(os.path.join(base, "t10k-labels-idx1-ubyte"), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(x), generator=g)[:n_test]
    x_t = torch.from_numpy(x[idx].astype(np.float32) / 255.0)
    y_t = torch.from_numpy(y[idx])
    return x_t, y_t


def load_cifar_offline():
    """Load CIFAR-10 test images directly from the pickle batches
    (offline; same file order as torchvision's CIFAR10 test set)."""
    import pickle
    base = os.path.join(BASE, "data", "datasets", "cifar-10-batches-py")
    with open(os.path.join(base, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].astype(np.float32) / 255.0
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return (None, None,
            torch.from_numpy(x), torch.from_numpy(y))


def pgd_mlp(model, x, y, eps, steps=10, restarts=2, step_size=0.05):
    """Multi-restart PGD in the L-inf ball, same configuration as Table 11."""
    x0 = x.clone()
    lo = torch.clamp(x0 - eps, 0, 1)
    hi = torch.clamp(x0 + eps, 0, 1)
    for _ in range(restarts):
        delta = torch.empty_like(x0).uniform_(-eps, eps)
        adv = torch.clamp(x0 + delta, 0, 1).requires_grad_(True)
        for _ in range(steps):
            out = model(adv.unsqueeze(0))
            loss = -out[0, y] + out[0].masked_fill(
                torch.arange(out.size(1), device=out.device) == y,
                -1e18).max()
            model.zero_grad()
            loss.backward()
            g = adv.grad.data
            adv = adv + step_size * g.sign()
            adv = torch.clamp(adv, lo, hi)
            adv = adv.detach().requires_grad_(True)
        with torch.no_grad():
            if model(adv.unsqueeze(0)).argmax(1).item() != y:
                return True, adv.detach()
    return False, None


if __name__ == "__main__":
    main()
