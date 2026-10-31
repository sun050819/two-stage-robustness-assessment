# Two-Stage Robustness Assessment Protocol

Companion code for a two-stage robustness assessment protocol: fast
sampling-based screening followed by sound certification. It integrates
two research lines into one protocol:

- `FastRobustnessEstimation` — sampling-based fast robustness estimation
  (screening stage; fixed-budget sampling strategies).
- `SAFE2_NN_Verification` — SAFE2-Krawczyk gated relaxed interval-Newton
  two-directional certification (certification stage).

The paper's headline experiment — the *joint protocol evaluation*
(Section 7.6, Table 14) — is fully self-contained: it runs from this
folder with no hardcoded absolute paths, using vendored verification code
and bundled checkpoints; the MNIST and CIFAR-10 datasets are downloaded by
torchvision on first run into `data/datasets/` (or can be pre-seeded
there).

## Directory layout

```
two-stage-robustness-assessment/
├── README.md                  # this file — project navigation
├── verification/             # reproduction files for the analysis sections (rank-one check, forward-cost timing)
├── experiments/               # experiment scripts (all results reproducible)
│   └── vendor/                   # vendored SAFE2 modules
├── data/                      # bundled checkpoints + datasets
└── results/                   # result JSONs (authoritative numbers used in the tables)
```

## Running the joint protocol evaluation

Requirements: Python 3.x with `torch`, `torchvision`, `numpy`
(`verifier_cnn.py` also uses `scipy` for the Moore-Penrose preconditioner).

```powershell
cd experiments
python smoke_joint.py            # ~1 sample per workload, verifies the setup
python joint_protocol.py         # full run: 600 MNIST MLP + 900 CIFAR-10 CNN samples
python joint_protocol.py --workload cnn --cnn-samples 500 --out-suffix _1500  # 1500 CNN samples
python joint_protocol.py --workload mlp   # MNIST MLPs only (eps 0.01/0.02/0.05)
python joint_protocol.py --workload cnn   # IBP-trained CNNs only (eps 2/255, 4/255), 300/seed default
python cnn_rates_1500.py         # Section 7.4 certified rates: 3 seeds x 4 eps x 500 (~90 min CPU)
python crown_ibp_align_900.py    # historical alignment vs auto-LiRPA (superseded by refined runs)
python crown_ibp_rerun_1500.py   # refined CROWN-IBP on the same 1500 CIFAR set (Table 7, ~30 min CPU)
python crown_ibp_rerun_mnist600.py # refined CROWN-IBP on the same 600 MNIST set (Table 4, ~15 min)
python crown_refine_soundcheck.py # refined-CROWN soundness, MNIST MLP: 90 samples x 1500 draws, viol = 0
python crown_cnn_refine_soundcheck.py # refined-CROWN soundness, CIFAR CNN: 36 samples x 1000 draws, viol = 0
python crown_mnist_soundcheck.py # historical fixed-relaxation soundness check (kept for provenance)
python budget_sweep.py           # monotone tightening U(N), budgets 50..500 (Prop. 3, Table 16)
python budget_sweep_ext.py       # extended curve 50..2000 + exponential fit (Prop. 4, ~20 min CPU)
python dual_engine_table.py      # engine-agnosticism table (reuses published JSONs, no heavy compute)
python bab_safe2_probe.py        # Interval-Newton BaB probe (contraction + CROWN-BaB, ~40 min CPU)
python interval_summary.py       # certified two-sided interval [L,U] from the 1500-grid totals
python scaling_r2.py             # Section 7.5: scaling slopes + fit R^2 (few minutes)
python highdim_bab_completion.py # Section 7.6: BaB on 30 undecided MNIST boxes (~30 min)
python complete_solver_compare.py# Section 7.4: complete-solver comparison, 100 MNIST boxes
python screen_recall.py --eps 0.01 --out screen_recall.json        # Section 7.4
python screen_recall.py --eps 0.05 --out screen_recall_eps005.json # Section 7.4
python screening_runs10.py       # Section 7.5: adaptive/importance, 10 runs
python gds_vs_rs_runs10.py       # Section 7.5: equal-budget GGS vs RS, 10 runs
python lemma3_demo.py            # Section 5: strict-inclusion branch demo
```

The run is deterministic (per-sample screening streams seeded from the
4242 family; certification follows the fixed-seed protocol of the
certification stage). Results are written to `results/` and reproduce the
verdict distribution reported in Table 14 of the manuscript:

| Workload            | eps    | Screened vuln. | Cert. robust | Cert. vuln. | Undecided | Certified interval [L,U] | U(100) | U(500) |
|---------------------|--------|---------------:|-------------:|------------:|----------:|--------------------------|-------:|-------:|
| MNIST MLP           | 0.01   | 0.8%           | 17.0%        | 0.0%        | 82.2%     | [0.170, 0.992]           | 0.988  | 0.987  |
| MNIST MLP           | 0.02   | 1.5%           | 0.0%         | 0.0%        | 98.5%     | [0.000, 0.985]           | 0.982  | 0.977  |
| MNIST MLP           | 0.05   | 2.8%           | 0.0%         | 0.0%        | 97.2%     | [0.000, 0.972]           | 0.970  | 0.955  |
| CIFAR-10 CNN (IBP)  | 2/255  | 2.1%           | 55.0%        | 0.0%        | 42.9%     | [0.550, 0.979]           | 0.977  | 0.973  |
| CIFAR-10 CNN (IBP)  | 4/255  | 4.1%           | 25.2%        | 0.0%        | 70.7%     | [0.252, 0.959]           | 0.956  | 0.952  |

U = 1 − (screened + certified vulnerable)/n (Definition 7: every screened
witness counts, so both endpoints are backed by guarantees). The
monotone tightening curve U(N) at budgets 50/100/200/500 (Proposition 3;
measured on Table 16 of the manuscript) is produced by `budget_sweep.py` →
`results/budget_sweep.json`; the columns U(100)/U(500) show the curve at
larger budgets (U at N=50 is the interval's upper endpoint), and the
intermediate N=200 values are 0.987/0.978/0.963/0.975/0.952 in row order.

All 1305 robust certificates (102 MNIST, 825 and 378 CIFAR-10) survive
multi-restart PGD (steps=10,
restarts=2). The `results/*.json` files contain the full per-sample
records (`screen_min_margin`, `cert_iters`) needed to recompute the
statistics. The CIFAR rows above are the 1500-sample rerun
(`joint_cnn_1500.json`, `--cnn-samples 500`).

## Medium-scale network extension

Beyond the manuscript tables, `results/medium_mlp_rates.json` reports the
same two-stage protocol on a wider IBP-trained MLP
(`[784,256,256,10]`, FC-256 family, 3 seeds, 200 samples/seed, pooled
n=600), produced by `train_mlp_ibp_medium.py` → `medium_network_rates.py`
(checkpoints: `data/checkpoints/mlp_medium_seed{0,1,2}.pt`):

| Workload        | eps  | Screened vuln. | Cert. robust | Cert. vuln. | Undecided |
|-----------------|------|---------------:|-------------:|------------:|----------:|
| Medium MLP 256x2 | 0.01 | 0.5%           | 43.8%        | 0.0%        | 55.7%     |
| Medium MLP 256x2 | 0.02 | 0.7%           | 8.7%         | 0.0%        | 90.7%     |
| Medium MLP 256x2 | 0.05 | 2.0%           | 0.0%         | 0.0%        | 98.0%     |

## CROWN-IBP column (self-implemented, refined)

The CROWN-IBP entries in Tables 6 and 9 are our self-implemented
CROWN-IBP with per-layer intermediate-bound refinement
(RefineIntermediateBounds: every ReLU pre-activation bound is tightened by
a CROWN backward pass from the source affine output and merged with the
IBP bound by max/min), implemented in `experiments/vendor/baseline_crown.py`
(`crown_bounds_logits(..., refine=True)`). Current numbers on the paper's
sample sets:

| Set | eps | Certified rate (refined) | Historical auto-LiRPA reference |
|-----|-----|-------------------------:|-------------------------------:|
| MNIST 600 | 0.01 | 78.2% ± 0.029 | 57.2% |
| MNIST 600 | 0.02 | 25.8% ± 0.025 | 4.0% |
| MNIST 600 | 0.05+ | 0.0% | 0.0% |
| CIFAR 1500 | 1/255 | 77.3% ± 0.070 | -- |
| CIFAR 1500 | 2/255 | 60.4% ± 0.058 | 57.4% |
| CIFAR 1500 | 4/255 | 33.7% ± 0.037 | 28.6% |
| CIFAR 1500 | 8/255 | 5.4% ± 0.010 | -- |

The auto-LiRPA reference values were produced by the source repository's
baseline script on its own checkpoints (different weight set); on the
weight set used throughout this manuscript the refined self-implementation
matches or exceeds them. Soundness of the refined bound is verified by
sampling on both architectures (90 MNIST samples x 1500 draws, 36 CIFAR
samples x 1000 draws, per logit): worst violation 0.0 everywhere.

## Engine-agnosticism of the protocol (dual-engine table)

The Stage-2 engine is a replaceable component of the protocol. Running the
same two-stage protocol with the refined CROWN-IBP bound instead of
SAFE2-Krawczyk leaves the verdict semantics and the certified-interval
output form unchanged, and the screening-side upper endpoint U is exactly
invariant; only the certified lower endpoint moves with the engine
(Table 14 of the manuscript, `results/dual_engine_table.json`):

| Workload   | eps   | Engine              | L (cert. robust) | U (N=50; N=500) |
|------------|-------|---------------------|-----------------:|----------------:|
| MNIST MLP  | 0.01  | SAFE2-Krawczyk      | 0.170            | 0.992; 0.987    |
| MNIST MLP  | 0.01  | CROWN-IBP (refined) | 0.782            | 0.992; 0.987    |
| MNIST MLP  | 0.02  | SAFE2-Krawczyk      | 0.000            | 0.985; 0.977    |
| MNIST MLP  | 0.02  | CROWN-IBP (refined) | 0.258            | 0.985; 0.977    |
| CIFAR-10   | 2/255 | SAFE2-Krawczyk      | 0.550            | 0.979; 0.973    |
| CIFAR-10   | 2/255 | CROWN-IBP (refined) | 0.604            | 0.979; 0.973    |
| CIFAR-10   | 4/255 | SAFE2-Krawczyk      | 0.252            | 0.959; 0.952    |
| CIFAR-10   | 4/255 | CROWN-IBP (refined) | 0.337            | 0.959; 0.952    |

## Standard-model SOTA comparison and engine cross-check (2026-10-28)

Two additional measurement blocks were run on the local GPU
(Python 3.11, torch 2.5.1+cu121, RTX 3060 Laptop; official auto-LiRPA
0.8.0). All numbers below come from the JSON files in `results/`.

### 1. Same-weight comparison on standard (non-IBP) models — Table 4 / Fig. 3 of the manuscript

Scripts: `experiments/vendor/train_mnist_cnn.py` (standard MNIST CNN
training), `experiments/train_cnn.py --regime std` (standard CIFAR-10 CNN),
`experiments/std_sota_rates.py` (four-engine certification on the same
weights), `experiments/gen_fig12_std.py` (Fig. 3).

Checkpoints: `data/checkpoints/mnist_cnn_std_seed{0,1,2}.pt`,
`data/checkpoints/cnn_std_seed{0,1,2}.pt`. Stats:
`data/model_stats_mnist_cnn.csv`, `data/model_stats_cnn.csv`.
Results: `results/std_rates_mnist.json`, `results/std_rates_cifar.json`.

| Workload | eps | IBP | Krawczyk | SAFE2 | official auto-LiRPA |
|---|---|---|---|---|---|
| MNIST std CNN (acc 0.986) | 0.01 | 0.06% | 1.3%±0.008 | 1.6%±0.006 | 75.6%±0.059 |
| MNIST std CNN (acc 0.986) | ≥0.05 | 0% | 0% | 0% | 0% |
| CIFAR-10 std CNN (acc 0.616) | 1/255–8/255 | 0% | 0% | 0% | 0% |

McNemar (SAFE2 vs auto-LiRPA, MNIST, eps=0.01): b=0, c=1333, p≈0 —
every sample certified by any engine is certified by auto-LiRPA. The
paper reports this as the quantified boundary of the interval-method
family and scopes the certification stage to IBP-trained models.

### 2. Same-weight engine cross-check: official auto-LiRPA vs self-implemented CROWN-IBP — Table 7 of the manuscript

Script: `experiments/cert_crosscheck_auto_lirpa.py` (official auto-LiRPA
CROWN-IBP on the *identical* checkpoints, samples, and boxes as the
self-implemented column; `--kind mlp_ibp` uses the same 300-sample MNIST
test subset and un-clamped box as `crown_ibp_rerun_mnist600.py`;
`--kind cnn_ibp` uses the same 1500-sample CIFAR set and clamped box as
`crown_ibp_rerun_1500.py`).
Results: `results/crosscheck_mlp_ibp_unclamped.json`,
`results/crosscheck_cnn_ibp.json`.

| Workload | eps | official auto-LiRPA | self-implemented | Δ (pp) |
|---|---|---|---|---|
| MNIST MLP (IBP) | 0.01 | 57.2%±0.021 | 78.2%±0.029 | −21.0 |
| MNIST MLP (IBP) | 0.02 | 4.0%±0.007 | 25.8%±0.025 | −21.8 |
| CIFAR-10 CNN (IBP) | 1/255 | 76.5%±0.058 | 77.3%±0.070 | −0.8 |
| CIFAR-10 CNN (IBP) | 2/255 | 57.9%±0.052 | 60.4%±0.058 | −2.5 |
| CIFAR-10 CNN (IBP) | 4/255 | 29.0%±0.021 | 33.7%±0.037 | −4.7 |
| CIFAR-10 CNN (IBP) | 8/255 | 3.4%±0.007 | 5.4%±0.010 | −2.0 |

The official engine certifies less on both workloads; on MNIST its 57.2%
coincides with the historical reference value on the source repository's
weight set, which anchors the comparison. Both engines are sound by
sampling (Supplementary Material); the MNIST gap is the
RefineIntermediateBounds step of the self-implementation, which
auto-LiRPA's default CROWN-IBP configuration does not reproduce for that
architecture. Caveat: earlier exploratory runs that reported 86.5% for
the official engine used the full 10000-sample MNIST test set (different
samples than the manuscript's 300-sample subset) and were discarded;
`results/crosscheck_mlp_ibp_unclamped.json` is the authoritative number.

## Exponential convergence of the tightening (Proposition 4)

`budget_sweep_ext.py` extends the tightening curve to budgets
50/100/200/500/1000/2000 (600 MNIST samples pooled over three seeds,
same nested-draw seeds). Proposition 4 predicts
E[S(N)] ≈ (sum_i p_i) N whenever p_max N << 1 (the linear asymptote of
the exponential law E[S(N)] = n(1-(1-p)^N)), followed by an exponential
saturation beyond that range. In the measurable budget regime the data
verify the monotone and linear-asymptotic content of the law: a
least-squares fit of S(N) = a N + b gives per-sample slopes 0.0013 /
0.0029 / 0.0069 at eps = 0.01 / 0.02 / 0.05 (R^2 = 0.50 / 0.68 / 0.74,
Table 16), and the 0.05 slope is ~5x the 0.01 slope, matching the
fivefold radius ratio. The exponential tail is not directly observable in
this range: the fitted per-sample rates (~1e-6) place saturation around
N >= 1e5 draws, so the tail shape is carried by the theoretical bound
(`results/budget_sweep_ext.json`). This turns the monotone tightening
(Proposition 3) into a quantified budget-to-width exchange with an honest
identifiability statement: the witness side of the interval is reducible
at a measured rate and is guaranteed to keep reducing, while the absolute
rates on these workloads are small.

## Interval-Newton BaB probe (honest negative result)

`bab_safe2_probe.py` tests whether feeding the SAFE2 Phase-R interval-Newton
contraction as a preprocessing step unlocks the completion of undecided
MNIST boxes (the interval-Newton branch-and-bound route of the Future-work
section). On the same 30 undecided MNIST boxes at eps=0.02, the
contraction leaves every box unshrunken (mean and median width reduction
0.0; the first Krawczyk image is non-strict in all 30 cases) and the
CROWN-BaB decided fraction is unchanged (1/30, the same box the plain
CROWN-BaB decides). The bottleneck is the rank-one preconditioner, not the
search structure: with a rank-deficient Jacobian the Krawczyk image
collapses toward the midpoint and never strictly shrinks the box. The
paper reports this as a quantified bottleneck and records the open problem
(an interval preconditioner that handles rank-deficient Jacobians)
rather than claiming the route solved (`results/bab_safe2_probe.json`).

## Self-containment notes

- **No hardcoded paths.** All scripts resolve every path relative to
  their own location (`experiments/` -> `vendor/`, `data/checkpoints/`,
  `results/`); the project folder can be moved or renamed as a whole.
- **Datasets** are bundled under `data/datasets/`. If they are missing,
  the vendored loaders (`train.load_mnist`, `train_cnn.load_cifar10`)
  fall back to torchvision auto-download into the same directory.
- **Checkpoints** (MNIST MLPs and IBP-trained CNNs, 3 seeds each) are
  bundled under `data/checkpoints/`; training scripts are available in
  `experiments/vendor/` if you need to regenerate them.

## Reproducing the stage experiments (reproduction scripts)

The screening-stage and certification-stage experiments reported in the
manuscript are reproduced by the scripts in this folder (self-contained;
the vendored verification engine lives under `experiments/vendor/`):

- Screening experiments: `experiments/screening_runs10.py`,
  `experiments/screen_recall*.py`
- Certification experiments (training, certified rates, two-directional
  certificates, scaling, BaB cross-validation):
  `experiments/std_sota_rates.py`, `experiments/joint_protocol.py`,
  `experiments/crown_ibp_rerun_*.py`, `experiments/scaling_r2.py`,
  `experiments/highdim_bab_completion.py`,
  `experiments/complete_solver_compare.py`
- Joint-protocol extension on the medium-scale IBP-trained MLP:
  `experiments/train_mlp_ibp_medium.py` +
  `experiments/medium_network_rates.py`
- Timing / rank-one checks: `verification/*.py`

All scripts resolve their data/checkpoint paths relative to their own
location (no hardcoded absolute paths); datasets are auto-downloaded into
`data/datasets/` on first use if not pre-seeded.

## Data coverage notes

Reproducibility status of every number in the manuscript:

- **Fully reproducible from this repo:** the joint-protocol tables
  (Table 14, Table 15, Table 8 cross-checks), the standard-model SOTA
  tables (Table 5, S.1, S.5), the runtime tables (Table 13,
  `results/mlp_runtime_ibp.json` + `results/cnn_runtime_ibp.json`),
  the scaling/Prop.-4 fits (Table 12, Table 16), and Figs. 10, 11, 12.
  Fig. 6 (certified rate vs. radius) is reproducible from
  `results/aggregated_by_eps.csv` + `results/crown_ibp_mnist_600.json`.
- `results/aggregated_by_eps.csv` is the Fig. 6 data file. The SAFE2
  column is computed from `results/joint_mlp.json`; the IBP and Krawczyk
  columns reproduce the manuscript Table 4 values (the original CSV of the
  SAFE2 project is not part of this repository).
- `verification/measure_screen_timing.py` measures the end-to-end
  50-forward screen cost `c_s(50)` (median over 30 runs on one input,
  written to `verification/screen_timing.json`).
- **Timing semantics (Table 17):** the manuscript's 7 ms / 14 ms figures
  are the end-to-end single-input cost of one 50-forward screen
  (50 forward passes plus the random-direction sampling and margin
  checks), as measured by `measure_screen_timing.py`. By contrast,
  `verification/forward_timing.json` (produced by
  `verification/measure_forward_cost.py`) reports only the pure forward
  cost: its `mlp_c_s(50)_s` = 1.4 ms and `cnn_c_s(50)_s` = 4.2 ms are
  50 x single-forward timings without the screening overhead, so the two
  files intentionally measure different quantities and are not in
  conflict.
- **Not reproducible from this repo (original SAFE2 two-moons / early
  screening experiments):** Fig. 8 (witnessed-vulnerability rate on
  two-moons boxes), Fig. 9 (contraction trajectory), Table 10 (two-moons
  existence witnesses), Table 3 Panel A (MDS vs. RS screening rates),
  Table 6 / Tables S.6-S.7 (max certified radius h_max), Tables S.2-S.4
  (GDS/MDS screening and attack-detection rates). These numbers were
  produced by the original SAFE2_NN_Verification project and early-stage
  scripts that are not archived here; `generate_figures.py` skips Figs. 8-9
  with a notice when the source files are absent. The values are available
  from the authors on request.
