# -*- coding: utf-8 -*-
"""Generate fig12_std_cert_rates.pdf: standard-trained MNIST CNN certified
rates, four engines on the same weights (data: results/std_rates_mnist.json).

PR figure standards: no in-figure title (all context lives in the caption),
no redundant in-figure annotation, serif/Times fonts matching the rest of
the figure set, grayscale-distinguishable bar colors.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "pdf.fonttype": 42,   # embed TrueType fonts (Elsevier production requirement)
    "ps.fonttype": 42,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(BASE, 'results', 'std_rates_mnist.json'), encoding='utf-8'))
p = d['pooled']
eps01 = p['0.01']
methods = ['ibp', 'krawczyk', 'safe2', 'crown_ibp']
labels = {'ibp': 'IBP', 'krawczyk': 'Krawczyk', 'safe2': 'SAFE2-Krawczyk',
          'crown_ibp': 'auto-LiRPA (CROWN-IBP)'}
colors = {'ibp': '#8c8c8c', 'krawczyk': '#4c72b0', 'safe2': '#dd8452',
          'crown_ibp': '#55a868'}
rates = [eps01[m]['rate_mean'] * 100 for m in methods]
stds = [eps01[m]['rate_std'] * 100 for m in methods]

fig, ax = plt.subplots(figsize=(7.0, 3.1), dpi=200)
x = range(len(methods))
bars = ax.bar(x, rates, yerr=stds, capsize=4, width=0.58,
              color=[colors[m] for m in methods], edgecolor='black', linewidth=0.5)
for xi, (r, s) in enumerate(zip(rates, stds)):
    ax.text(xi, r + s + 1.2, ('%.2f%%' % r),
            ha='center', va='bottom', fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([labels[m] for m in methods], fontsize=8.5)
ax.set_ylabel('certified robust rate (%)', fontsize=9)
ax.set_ylim(0, 88)
ax.grid(axis='y', alpha=0.3, linewidth=0.5)
fig.tight_layout()
out = os.path.join(BASE, 'paper', 'figures', 'fig12_std_cert_rates.pdf')
fig.savefig(out, bbox_inches='tight')
print('SAVED', out)
