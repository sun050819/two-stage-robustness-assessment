# -*- coding: utf-8 -*-
"""
Screening statistics, extended from 5 to 10 independent runs
(Section 7.5 of the merged manuscript).

Reproduces the two design comparisons of the screening study with 10
independently trained FC-128 MNIST models (seeds 0..9) instead of 5:

  * adaptive vs. fixed budget (Panel B of Table 6);
  * uniform vs. gradient-guided importance sampling vs. adaptive
    importance sampling (Panel C of Table 6).

The algorithms faithfully mirror the released screening scripts
(run_experiment9_multi.py / run_experiment10_multi.py): FC-128, SGD
0.01/0.9, 10 epochs, batch 128; 1000 fixed MNIST test samples in batches
of 100; budget 50; eps in {0.05, 0.10, 0.15}; per-sample detection rate
with paired t-tests per epsilon.

Outputs mean +/- std over the 10 runs and paired t-test p-values, so the
paper's tables can be updated to the 10-run statistics.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json
import os
import pathlib
from scipy import stats

BASE = pathlib.Path(__file__).resolve().parents[1]
DATA_ROOT = BASE / "data" / "datasets"
OUT_DIR = BASE / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FCNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x.view(-1, 784))))


def train_model(seed, epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True,
                                   transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = FCNet().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    return model


def fixed_budget_sampling(model, x, y, epsilon, n_samples):
    model.eval()
    with torch.no_grad():
        batch_size = x.size(0)
        detected = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        for _ in range(n_samples):
            delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
            output = model(x + delta)
            detected = detected | (output.argmax(dim=1) != y)
    return detected.float().mean().item() * 100


def adaptive_budget_sampling(model, x, y, epsilon, avg_n_samples):
    model.eval()
    with torch.no_grad():
        batch_size = x.size(0)
        output0 = model(x)
        margin = (output0.gather(1, y.view(-1, 1)).squeeze()
                  - output0.scatter(1, y.view(-1, 1), -1e10).max(dim=1)[0])
        margin_min = margin.min().item()
        margin_max = margin.max().item()
        margin_range = margin_max - margin_min + 1e-8
        detected = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        total_used = 0
        for i in range(batch_size):
            margin_ratio = (margin[i].item() - margin_min) / margin_range
            n_adaptive = max(1, int(avg_n_samples * (1.5 - margin_ratio)))
            xi = x[i:i+1]
            yi = y[i:i+1]
            for _ in range(n_adaptive):
                delta = torch.empty_like(xi).uniform_(-epsilon, epsilon)
                out = model(xi + delta)
                if out.argmax(dim=1).item() != yi.item():
                    detected[i] = True
                    break
            total_used += n_adaptive
        return detected.float().mean().item() * 100, total_used / batch_size


def compute_margin_and_gradient(model, x, y):
    x_var = x.clone().detach().requires_grad_(True)
    output = model(x_var)
    true_score = output.gather(1, y.view(-1, 1)).squeeze()
    output_masked = output.clone()
    output_masked.scatter_(1, y.view(-1, 1), -1e10)
    margin = true_score - output_masked.max(dim=1)[0]
    margin.sum().backward()
    gradient = x_var.grad.clone()
    return margin.detach(), gradient.detach()


def uniform_sampling(model, x, y, epsilon, n_samples):
    model.eval()
    with torch.no_grad():
        batch_size = x.size(0)
        detected = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
        for _ in range(n_samples):
            delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
            output = model(x + delta)
            detected = detected | (output.argmax(dim=1) != y)
    return detected.float().mean().item() * 100


def importance_sampling(model, x, y, epsilon, n_samples, sigma=0.3, alpha=0.5):
    model.eval()
    batch_size = x.size(0)
    detected = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    margin, gradient = compute_margin_and_gradient(model, x, y)
    grad_norm = gradient.view(batch_size, -1).norm(dim=1, keepdim=True).view(
        batch_size, 1, 1, 1)
    grad_norm = torch.clamp(grad_norm, min=1e-8)
    grad_direction = gradient / grad_norm
    for _ in range(n_samples):
        delta_uniform = torch.empty_like(x).uniform_(-epsilon, epsilon)
        delta_grad = -torch.randn_like(x) * sigma * epsilon
        delta_grad = delta_grad * grad_direction.expand_as(x)
        delta = (1 - alpha) * delta_uniform + alpha * delta_grad
        delta = torch.clamp(delta, -epsilon, epsilon)
        output = model(x + delta)
        detected = detected | (output.argmax(dim=1) != y)
    return detected.float().mean().item() * 100


def adaptive_importance_sampling(model, x, y, epsilon, n_samples, sigma=0.3,
                                 alpha=0.5):
    model.eval()
    batch_size = x.size(0)
    margin, gradient = compute_margin_and_gradient(model, x, y)
    grad_norm = gradient.view(batch_size, -1).norm(dim=1, keepdim=True).view(
        batch_size, 1, 1, 1)
    grad_norm = torch.clamp(grad_norm, min=1e-8)
    grad_direction = gradient / grad_norm
    margin_min = margin.min().item()
    margin_max = margin.max().item()
    margin_range = margin_max - margin_min + 1e-8
    detected = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)
    for i in range(batch_size):
        margin_ratio = (margin_max - margin[i].item()) / margin_range
        n_adaptive = int(n_samples * (0.5 + margin_ratio))
        xi = x[i:i+1]
        yi = y[i:i+1]
        grad_i = grad_direction[i:i+1]
        for _ in range(n_adaptive):
            delta_uniform = torch.empty_like(xi).uniform_(-epsilon, epsilon)
            delta_grad = torch.randn_like(xi) * sigma * epsilon * grad_i
            delta = (1 - alpha) * delta_uniform + alpha * delta_grad
            delta = torch.clamp(delta, -epsilon, epsilon)
            out = model(xi + delta)
            if out.argmax(dim=1).item() != yi.item():
                detected[i] = True
                break
    return detected.float().mean().item() * 100


def main():
    torch.backends.cudnn.benchmark = True
    print("device:", DEVICE, flush=True)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(DATA_ROOT, train=False, download=True,
                                  transform=transform)
    test_subset = Subset(test_dataset, range(1000))
    test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)
    epsilons = [0.05, 0.10, 0.15]
    avg_budget = 50
    n_runs = 10
    seeds = list(range(n_runs))

    res_adapt = {"fixed": {}, "adaptive": {}, "avg_actual": {}, "stats": {},
                 "epsilons": epsilons, "n_runs": n_runs}
    res_imp = {"uniform": {}, "importance": {}, "adaptive_importance": {},
               "stats": {}, "epsilons": epsilons, "n_runs": n_runs}

    for run, seed in enumerate(seeds):
        print(f"\n=== Run {run+1}/{n_runs} (seed {seed}) ===", flush=True)
        model = train_model(seed, epochs=10)
        model.eval()
        for eps in epsilons:
            fixed_r = []
            adapt_r = []
            avg_used = []
            u_r = []
            i_r = []
            a_r = []
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                fixed_r.append(fixed_budget_sampling(model, data, target, eps,
                                                     avg_budget))
                rr, au = adaptive_budget_sampling(model, data, target, eps,
                                                  avg_budget)
                adapt_r.append(rr)
                avg_used.append(au)
                u_r.append(uniform_sampling(model, data, target, eps,
                                            avg_budget))
                i_r.append(importance_sampling(model, data, target, eps,
                                               avg_budget, sigma=0.3,
                                               alpha=0.5))
                a_r.append(adaptive_importance_sampling(model, data, target,
                                                        eps, avg_budget,
                                                        sigma=0.3, alpha=0.5))
            res_adapt["fixed"].setdefault(str(eps), []).append(
                float(np.mean(fixed_r)))
            res_adapt["adaptive"].setdefault(str(eps), []).append(
                float(np.mean(adapt_r)))
            res_adapt["avg_actual"].setdefault(str(eps), []).append(
                float(np.mean(avg_used)))
            res_imp["uniform"].setdefault(str(eps), []).append(
                float(np.mean(u_r)))
            res_imp["importance"].setdefault(str(eps), []).append(
                float(np.mean(i_r)))
            res_imp["adaptive_importance"].setdefault(str(eps), []).append(
                float(np.mean(a_r)))
            print(f"  eps={eps}: fixed {np.mean(fixed_r):.2f}  "
                  f"adaptive {np.mean(adapt_r):.2f}  "
                  f"U {np.mean(u_r):.2f}  IS {np.mean(i_r):.2f}  "
                  f"AIS {np.mean(a_r):.2f}", flush=True)

    for eps in epsilons:
        f = np.array(res_adapt["fixed"][str(eps)])
        a = np.array(res_adapt["adaptive"][str(eps)])
        t_af, p_af = stats.ttest_rel(a, f)
        res_adapt["stats"][str(eps)] = {
            "fixed_mean": float(f.mean()), "fixed_std": float(f.std()),
            "adaptive_mean": float(a.mean()), "adaptive_std": float(a.std()),
            "t": float(t_af), "p": float(p_af),
            "avg_budget_actual": float(
                np.mean(res_adapt["avg_actual"][str(eps)])),
        }
        u = np.array(res_imp["uniform"][str(eps)])
        i = np.array(res_imp["importance"][str(eps)])
        a2 = np.array(res_imp["adaptive_importance"][str(eps)])
        t_ui, p_ui = stats.ttest_rel(u, i)
        t_ua, p_ua = stats.ttest_rel(u, a2)
        res_imp["stats"][str(eps)] = {
            "uniform_mean": float(u.mean()), "uniform_std": float(u.std()),
            "is_mean": float(i.mean()), "is_std": float(i.std()),
            "ais_mean": float(a2.mean()), "ais_std": float(a2.std()),
            "t_uniform_vs_is": float(t_ui), "p_uniform_vs_is": float(p_ui),
            "t_uniform_vs_ais": float(t_ua), "p_uniform_vs_ais": float(p_ua),
        }
        print(f"STATS eps={eps}: fixed {f.mean():.2f}±{f.std():.2f} vs "
              f"adaptive {a.mean():.2f}±{a.std():.2f} p={p_af:.4f} | "
              f"U {u.mean():.2f}±{u.std():.2f} IS {i.mean():.2f}±{i.std():.2f} "
              f"AIS {a2.mean():.2f}±{a2.std():.2f} | "
              f"U-vs-IS p={p_ui:.4f} U-vs-AIS p={p_ua:.4f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_DIR / "screening_runs10_adaptive.json", "w") as f:
        json.dump(res_adapt, f, indent=2, default=str)
    with open(OUT_DIR / "screening_runs10_importance.json", "w") as f:
        json.dump(res_imp, f, indent=2, default=str)
    print("\nsaved screening_runs10_adaptive.json / "
          "screening_runs10_importance.json", flush=True)


if __name__ == "__main__":
    main()
