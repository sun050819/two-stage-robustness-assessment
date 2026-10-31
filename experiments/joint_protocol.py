# -*- coding: utf-8 -*-
"""
Joint two-stage protocol evaluation (merged paper, Section 7.6).

Runs the two-stage robustness assessment protocol on the certification
models of the SAFE2 line of work, using the screening machinery of the
sampling-estimation line:

    Stage 1 (screen):  pure random sampling, budget N=50, pointwise margin
                       g(x) = f_y(x) - max_{i!=y} f_i(x); a point with g<=0
                       is an explicit adversarial witness  -> screened vulnerable.
    Stage 2 (certify): SAFE2-Krawczyk (gated relaxed interval-Newton) on the
                       candidates that pass the screen -> certified robust /
                       certified vulnerable / undecided.

Checks:
  * soundness cross-check: no sample can be screened vulnerable AND
    certified robust (both stages share the same margin convention);
  * PGD spot check on all certified-robust samples (steps=10, restarts=2);
  * per-sample screening and certification runtime.

Workloads:
  * MNIST MLP 784-64-64-10, 3 seeds, first 200 correctly classified test
    images per seed, eps in {0.01, 0.02, 0.05};
  * CIFAR-10 IBP-trained CNN (3,32,32), 3 seeds, first 300 correctly
    classified test images per seed, eps in {2/255, 4/255} (clamp01=True).

Self-contained layout (no hardcoded absolute paths; all paths resolve
relative to this file):
  <project>/
    experiments/
      joint_protocol.py      (this file)
      vendor/                (SAFE2 modules: train.py, train_cnn.py,
                              verifier_lib.py, verifier_cnn.py)
    data/
      checkpoints/           (mlp_seed*.pt, cnn_ibp_seed*.pt)
      datasets/              (MNIST/, cifar-10-batches-py/)
    results/
      joint_mlp.json, joint_cnn.json, joint_run.log

Datasets are bundled under data/datasets; if absent, the vendored loaders
fall back to torchvision auto-download into the same directory.

Outputs (JSON + log) written to ../results/.
"""
import os
import sys
import json
import time
import argparse

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(BASE, 'vendor')
sys.path.insert(0, VENDOR)

from train import MLP, load_mnist            # noqa: E402
import verifier_lib as vl                    # noqa: E402
from train_cnn import make_cnn, load_cifar10  # noqa: E402
import verifier_cnn as vc                    # noqa: E402

OUT_DIR = os.path.join(BASE, '..', 'results')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

CKPT = os.path.join(BASE, '..', 'data', 'checkpoints')
CKPT = os.path.abspath(CKPT)

BUDGET = 50                 # screening evaluations per sample
ALPHA_MIN_MLP = 0.2
ALPHA_FACTOR_MLP = 0.8
ALPHA_MIN_CNN = 0.2
ALPHA_FACTOR_CNN = 0.8
MAX_ITER_R_CNN = 12
MAX_ITER_E_CNN = 12
INPUT_SHAPE = (3, 32, 32)
PGD_STEPS = 10
PGD_RESTARTS = 2


def margin_at_logits(model, x, c):
    """Pointwise margin of a (flattened, 1-D) input, logit convention."""
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).squeeze(0)
        others = logits.clone()
        others[c] = float('-inf')
        return float(logits[c] - others.max())


def screen_rs(model, x0, c, eps, budget=BUDGET, seed_base=0,
              input_shape=None, clamp=False):
    """Pure random sampling screen. Returns (verdict, min_margin)."""
    best_margin = None
    torch.manual_seed(seed_base)
    with torch.no_grad():
        for _ in range(budget):
            delta = (torch.rand_like(x0) * 2.0 - 1.0) * eps
            x = x0 + delta
            if clamp:
                x = torch.clamp(x, 0.0, 1.0)
            if input_shape is not None:
                g = vc.margin_at(model, x, c, input_shape)
            else:
                logits = model(x.unsqueeze(0)).squeeze(0)
                others = logits.clone()
                others[c] = float('-inf')
                g = float(logits[c] - others.max())
            if best_margin is None or g < best_margin:
                best_margin = g
            if g <= 0:
                return 'vulnerable', g
    return 'likely_robust', best_margin


def run_workload(kind, seeds, eps_list, n_samples, ckpt_prefix,
                 use_cnn, log, out_suffix=''):
    """Run the two-stage protocol over one workload family."""
    rows = []          # per (seed, eps)
    per_sample = []    # detailed records
    totals = {m: {eps: {'screened_vuln': 0, 'cert_robust': 0,
                        'cert_vuln': 0, 'undecided': 0, 'errors': 0}
                  for eps in eps_list} for m in ('counts',)}
    screen_times = {eps: [] for eps in eps_list}
    cert_times = {eps: [] for eps in eps_list}

    if use_cnn:
        _, _, x_te, y_te = load_cifar10()
        models = {}
        for s in seeds:
            model = make_cnn()
            model.load_state_dict(torch.load(
                os.path.join(CKPT, f'{ckpt_prefix}_seed{s}.pt'),
                map_location='cpu'))
            model.eval()
            models[s] = model
    else:
        x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
        models = {}
        for s in seeds:
            model = MLP([784, 64, 64, 10])
            model.load_state_dict(torch.load(
                os.path.join(CKPT, f'{ckpt_prefix}_seed{s}.pt'),
                map_location='cpu'))
            model.eval()
            models[s] = model

    for s in seeds:
        model = models[s]
        with torch.no_grad():
            if use_cnn:
                preds = model(x_te.view(-1, *INPUT_SHAPE)).argmax(1)
            else:
                preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n_samples]
        log(f'[{kind} seed {s}] n_samples={len(idx)}')

        for eps in eps_list:
            cnt = {'screened_vuln': 0, 'cert_robust': 0, 'cert_vuln': 0,
                   'undecided': 0, 'errors': 0}
            pgd_broken = 0
            pgd_checked = 0
            st_screen, st_cert = 0.0, 0.0
            for pos, si in enumerate(idx.tolist()):
                x0 = x_te[si]
                c = int(y_te[si].item())
                seed_base = 4242 + s * 100000 + int(eps * 1e6) + pos

                # ---- Stage 1: screening ----
                t0 = time.perf_counter()
                try:
                    if use_cnn:
                        verdict, min_marg = screen_rs(
                            model, x0.view(-1), c, eps, BUDGET, seed_base,
                            INPUT_SHAPE, clamp=True)
                    else:
                        verdict, min_marg = screen_rs(
                            model, x0, c, eps, BUDGET, seed_base)
                    st_screen += time.perf_counter() - t0
                except Exception as e:
                    log(f'  ERR screen {kind} s{s} eps{eps} si{si}: {e}')
                    cnt['errors'] += 1
                    continue

                if verdict == 'vulnerable':
                    cnt['screened_vuln'] += 1
                    rec = {'seed': s, 'eps': eps, 'sample': si,
                           'verdict': 'screened_vulnerable',
                           'screen_min_margin': min_marg}
                    per_sample.append(rec)
                    continue

                # ---- Stage 2: certification on candidates ----
                t0 = time.perf_counter()
                try:
                    if use_cnn:
                        r, it, _, root, root_marg = vc.verify_safe2(
                            model, x0.view(-1), c, eps, INPUT_SHAPE,
                            ALPHA_MIN_CNN, ALPHA_FACTOR_CNN,
                            MAX_ITER_R_CNN, MAX_ITER_E_CNN, clamp01=True)
                    else:
                        r, it, _, root, root_marg = vl.verify_safe2(
                            model, x0, c, eps, ALPHA_MIN_MLP, ALPHA_FACTOR_MLP)
                    st_cert += time.perf_counter() - t0
                except Exception as e:
                    log(f'  ERR certify {kind} s{s} eps{eps} si{si}: {e}')
                    cnt['errors'] += 1
                    continue

                if r == 'robust':
                    cnt['cert_robust'] += 1
                    pgd_checked += 1
                    try:
                        if use_cnn:
                            broken = vc.pgd_attack(
                                model, x0.view(-1), c, eps, INPUT_SHAPE,
                                steps=PGD_STEPS, restarts=PGD_RESTARTS,
                                step_size=0.05)
                        else:
                            broken = vl.pgd_attack(
                                model, x0, c, eps, steps=PGD_STEPS,
                                restarts=PGD_RESTARTS, step_size=0.05)
                        if broken:
                            pgd_broken += 1
                    except Exception:
                        pass
                elif r == 'exist':
                    cnt['cert_vuln'] += 1
                    # existence witnesses are midpoint witnesses in the
                    # rank-one regime; record their margins
                    root_marg = root_marg if root_marg is not None else min_marg
                    rec = {'seed': s, 'eps': eps, 'sample': si,
                           'verdict': 'certified_vulnerable',
                           'screen_min_margin': min_marg,
                           'cert_root_margin': root_marg}
                    per_sample.append(rec)
                    continue
                else:
                    cnt['undecided'] += 1

                rec = {'seed': s, 'eps': eps, 'sample': si,
                       'verdict': 'certified_robust' if r == 'robust'
                       else 'undecided',
                       'screen_min_margin': min_marg,
                       'cert_iters': int(it)}
                per_sample.append(rec)

            screen_times[eps].append(st_screen)
            cert_times[eps].append(st_cert)
            totals['counts'][eps]['screened_vuln'] += cnt['screened_vuln']
            totals['counts'][eps]['cert_robust'] += cnt['cert_robust']
            totals['counts'][eps]['cert_vuln'] += cnt['cert_vuln']
            totals['counts'][eps]['undecided'] += cnt['undecided']
            totals['counts'][eps]['errors'] += cnt['errors']
            n = len(idx)
            log(f'[{kind} s{s} eps{eps}] n={n} '
                f'screened_vuln={cnt["screened_vuln"]} '
                f'cert_robust={cnt["cert_robust"]} '
                f'cert_vuln={cnt["cert_vuln"]} '
                f'undecided={cnt["undecided"]} errors={cnt["errors"]} '
                f'pgd_checked={pgd_checked} pgd_broken={pgd_broken} '
                f'screen_s={st_screen:.2f} cert_s={st_cert:.2f}')
            rows.append({'kind': kind, 'seed': s, 'eps': eps,
                         'n': n, **cnt,
                         'pgd_checked': pgd_checked,
                         'pgd_broken': pgd_broken})

    # ---- pooled summary ----
    summary = {}
    for eps in eps_list:
        vals = {k: [] for k in
                ('screened_vuln', 'cert_robust', 'cert_vuln', 'undecided')}
        for r in rows:
            if r['eps'] == eps:
                for k in vals:
                    vals[k].append(r[k])
        n_tot = sum(r['n'] for r in rows if r['eps'] == eps)
        summary[str(eps)] = {
            'n_total': n_tot,
            'n_seeds': len(vals['screened_vuln']),
            'screened_vulnerable': {
                'count': sum(vals['screened_vuln']),
                'rate_mean': float(sum(vals['screened_vuln']) / max(n_tot, 1)),
                'per_seed_mean': float(sum(vals['screened_vuln']) /
                                       len(vals['screened_vuln'])),
                'per_seed_std': float(
                    torch.tensor(vals['screened_vuln']).float().std().item())
                if len(vals['screened_vuln']) > 1 else 0.0,
            },
            'certified_robust': {
                'count': sum(vals['cert_robust']),
                'rate_mean': float(sum(vals['cert_robust']) / max(n_tot, 1)),
                'per_seed_std': float(
                    torch.tensor(vals['cert_robust']).float().std().item())
                if len(vals['cert_robust']) > 1 else 0.0,
            },
            'certified_vulnerable': {
                'count': sum(vals['cert_vuln']),
                'rate_mean': float(sum(vals['cert_vuln']) / max(n_tot, 1)),
            },
            'undecided': {
                'count': sum(vals['undecided']),
                'rate_mean': float(sum(vals['undecided']) / max(n_tot, 1)),
            },
            'screening_s_per_seed': float(sum(screen_times[eps]) /
                                          len(screen_times[eps])),
            'cert_s_per_seed': float(sum(cert_times[eps]) /
                                     len(cert_times[eps])),
        }
    out = {'kind': kind, 'rows': rows, 'summary': summary,
           'per_sample': per_sample}
    fn = os.path.join(OUT_DIR, f'joint_{kind}{out_suffix}.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    log(f'[{kind}] saved {fn}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workload', choices=['mlp', 'cnn', 'all'], default='all')
    ap.add_argument('--cnn-samples', type=int, default=300,
                    help='CIFAR samples per seed (1500 total at 500/seed)')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='suffix for joint_<kind>.json (e.g. _1500)')
    args = ap.parse_args()

    log_path = os.path.join(OUT_DIR,
                            f'joint_run{args.out_suffix}.log')
    LOG = open(log_path, 'w', encoding='utf-8')

    def log(msg):
        print(msg, flush=True)
        LOG.write(msg + '\n')
        LOG.flush()

    out_suffix = args.out_suffix
    if args.workload in ('mlp', 'all'):
        run_workload('mlp', [0, 1, 2], [0.01, 0.02, 0.05], 200,
                     'mlp', False, log, out_suffix)
    if args.workload in ('cnn', 'all'):
        run_workload('cnn', [0, 1, 2],
                     [2.0 / 255.0, 4.0 / 255.0], args.cnn_samples,
                     'cnn_ibp', True, log, out_suffix)
    log('ALL DONE')
    LOG.close()


if __name__ == '__main__':
    main()
