# -*- coding: utf-8 -*-
"""
Equal-budget comparison, extended from 5 to 10 independent runs
(Section 7.5 of the merged manuscript).

Faithfully mirrors the released run_experiment4_multi.py: FC-256 MNIST,
SGD 0.01/0.9, 20 epochs, batch 128; 200 fixed test samples; gradient-guided
sampling (GGS) vs. pure random sampling (RS) at budgets of 50 forward
evaluations; eps in {0.05, 0.10, 0.15}; mean +/- std over runs and paired
t-test per epsilon.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json
import time
import os
import pathlib
from scipy import stats

BASE = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / "data" / "datasets"
OUT_DIR = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMNISTNet(nn.Module):
    def __init__(self, hidden_sizes=[256]):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(-1, 28 * 28))


def train_model(seed, epochs=20, hidden_sizes=[256]):
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True,
                                   transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128,
                                               shuffle=True)
    model = SimpleMNISTNet(hidden_sizes=hidden_sizes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def compute_g_and_grad(model, x_np, true_label):
    x_tensor = torch.tensor(x_np, dtype=torch.float32,
                            device=DEVICE).view(1, 1, 28, 28)
    x_tensor.requires_grad_(True)
    out = model(x_tensor)
    true_out = out[0, true_label]
    other_out = -1e10
    for i in range(10):
        if i != true_label and out[0, i].item() > other_out:
            other_out = out[0, i].item()
    boundary = true_out - torch.tensor(other_out, requires_grad=False)
    boundary.backward()
    g_m = boundary.item()
    grad = x_tensor.grad.detach().cpu().numpy().flatten().copy()
    return g_m, grad


def eval_along_direction(model, x_center_np, d, t, true_label):
    x_np = x_center_np + t * d
    x_tensor = torch.tensor(x_np, dtype=torch.float32,
                            device=DEVICE).view(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x_tensor)
        true_out = out[0, true_label].item()
        other_out = -1e10
        for i in range(10):
            if i != true_label and out[0, i].item() > other_out:
                other_out = out[0, i].item()
        return true_out - other_out


def verify_gradient_guided(model, x_center_np, true_label, epsilon, n_evals=50):
    g_m, grad = compute_g_and_grad(model, x_center_np, true_label)
    if g_m <= 0:
        return False
    grad_n = np.linalg.norm(grad)
    if grad_n < 1e-10:
        return True
    d_grad = grad / grad_n
    grad_samples = min(10, n_evals // 5)
    for t in np.linspace(-epsilon, epsilon, grad_samples):
        if eval_along_direction(model, x_center_np, d_grad, t, true_label) <= 0:
            return False
    remaining = n_evals - grad_samples
    n_random_dirs = max(1, remaining // 5)
    for _ in range(n_random_dirs):
        d_rand = np.random.randn(*x_center_np.shape)
        d_rand = d_rand / (np.linalg.norm(d_rand) + 1e-10)
        for t in np.linspace(-epsilon, epsilon, 5):
            if eval_along_direction(model, x_center_np, d_rand, t,
                                    true_label) <= 0:
                return False
    return True


def verify_random_sampling(model, x_center_np, true_label, epsilon, n_evals=50):
    g_m = eval_along_direction(model, x_center_np, np.zeros_like(x_center_np),
                               0.0, true_label)
    if g_m <= 0:
        return False
    for _ in range(n_evals):
        perturbation = np.random.uniform(-epsilon, epsilon, x_center_np.shape)
        if eval_along_direction(model, x_center_np, perturbation, 1.0,
                                true_label) <= 0:
            return False
    return True


def main():
    torch.backends.cudnn.benchmark = True
    print("device:", DEVICE, flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(DATA_ROOT, train=False, download=True,
                                  transform=transform)
    test_samples = [test_dataset[i] for i in range(200)]
    epsilons = [0.05, 0.10, 0.15]
    n_runs = 10

    results = {"epsilons": epsilons, "n_runs": n_runs, "n_samples": 200,
               "gg_rates": {}, "rs_rates": {}, "gg_time": {}, "rs_time": {}}
    for eps in epsilons:
        results["gg_rates"][eps] = []
        results["rs_rates"][eps] = []
        results["gg_time"][eps] = []
        results["rs_time"][eps] = []

    for run in range(n_runs):
        print(f"\n=== Run {run+1}/{n_runs} (seed {run}) ===", flush=True)
        model = train_model(run, epochs=20, hidden_sizes=[256])
        model.eval()
        for eps in epsilons:
            gg_c = rs_c = 0
            t_gg = t_rs = 0.0
            for i, (x, y) in enumerate(test_samples):
                x_np = x.view(-1).numpy()
                tl = y
                t0 = time.perf_counter()
                r1 = verify_gradient_guided(model, x_np, tl, eps, 50)
                t1 = time.perf_counter()
                r2 = verify_random_sampling(model, x_np, tl, eps, 50)
                t2 = time.perf_counter()
                gg_c += int(r1)
                rs_c += int(r2)
                t_gg += (t1 - t0)
                t_rs += (t2 - t1)
            results["gg_rates"][eps].append(gg_c / 200 * 100)
            results["rs_rates"][eps].append(rs_c / 200 * 100)
            results["gg_time"][eps].append(t_gg / 200 * 1000)
            results["rs_time"][eps].append(t_rs / 200 * 1000)
            print(f"  eps={eps}: GG={gg_c/2:.1f}% RS={rs_c/2:.1f}% | "
                  f"GG {t_gg/200*1000:.2f}ms RS {t_rs/200*1000:.2f}ms",
                  flush=True)

    for eps in epsilons:
        gg = np.array(results["gg_rates"][eps])
        rs = np.array(results["rs_rates"][eps])
        t_stat, p = stats.ttest_rel(gg, rs)
        w, wp = stats.wilcoxon(gg - rs)
        print(f"\neps={eps}: GG {gg.mean():.2f}±{gg.std():.2f}  "
              f"RS {rs.mean():.2f}±{rs.std():.2f}  t={t_stat:.3f} "
              f"p={p:.4f} wilcoxon={wp:.4f}", flush=True)
        results.setdefault("stats", {})[str(eps)] = {
            "gg_mean": float(gg.mean()), "gg_std": float(gg.std()),
            "rs_mean": float(rs.mean()), "rs_std": float(rs.std()),
            "t": float(t_stat), "p": float(p),
            "wilcoxon_p": float(wp),
        }
    results["gg_time_summary"] = {str(e): {"mean_ms": float(np.mean(v)),
                                           "std_ms": float(np.std(v))}
                                  for e, v in results["gg_time"].items()}
    results["rs_time_summary"] = {str(e): {"mean_ms": float(np.mean(v)),
                                           "std_ms": float(np.std(v))}
                                  for e, v in results["rs_time"].items()}

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_DIR / "gds_vs_rs_runs10.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nsaved gds_vs_rs_runs10.json", flush=True)


if __name__ == "__main__":
    main()
