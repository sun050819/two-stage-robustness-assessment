# -*- coding: utf-8 -*-
"""Regenerate manuscript figures (Figs. 2-5) to PR figure standards.

PR-compliant conventions applied:
  * vector PDF output (no in-figure titles, no color-dependent text)
  * error bars (mean +/- std) where the caption promises them
  * triple distinction (color + line style + marker) so plots stay
    readable in grayscale; no red/green as the only contrast
  * legends outside the data area
  * terminology unified to the manuscript text ("witnessed-vulnerability
    rate")
  * data sources are the checked-in result files; every number matches
    the corresponding table in the manuscript.

Run:  python generate_figures.py
Outputs are written to ../paper/figures/ (fig6_cert_rates.pdf,
fig8_existence.pdf, fig9_contraction.pdf, fig10_scaling.pdf).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
SRC_RESULTS = os.path.join(BASE, "sources", "SAFE2_NN_Verification", "results")
FIG_DIR = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

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

# Low-saturation, grayscale-distinguishable styles: color + ls + marker.
STYLE = {
    "IBP":            dict(color="#7f8c8d", ls="-",  marker="o", lw=1.5, ms=4),
    "Krawczyk":       dict(color="#4c72b0", ls="--", marker="s", lw=1.5, ms=4),
    "SAFE2-Krawczyk": dict(color="#2f6f4f", ls="-.", marker="^", lw=1.8, ms=5),
    "CROWN-IBP (self-implemented)": dict(color="#c44e52", ls=":", marker="D", lw=1.5, ms=4),
}


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(FIG_DIR, name))


def fig2_cert_rates():
    """MNIST certified robust rate vs. radius (data from tab:rates)."""
    rows = {}
    with open(os.path.join(SRC_RESULTS, "aggregated_by_eps.csv")) as f:
        for line in f.read().strip().splitlines()[1:]:
            eps, method, mean, std = line.split(",")
            rows.setdefault(method, {})[float(eps)] = (float(mean), float(std))
    with open(os.path.join(RESULTS, "crown_ibp_mnist_600.json")) as f:
        crown = json.load(f)["pooled"]
    eps_list = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    fig, ax = plt.subplots(figsize=(4.6, 3.45))
    series = {
        "IBP":            [(rows["ibp"][e][0], rows["ibp"][e][1]) for e in eps_list],
        "Krawczyk":       [(rows["krawczyk"][e][0], rows["krawczyk"][e][1]) for e in eps_list],
        "SAFE2-Krawczyk": [(rows["safe2"][e][0], rows["safe2"][e][1]) for e in eps_list],
        "CROWN-IBP (self-implemented)": [(crown[str(e)]["rate_mean"], crown[str(e)]["rate_std"])
                             for e in eps_list],
    }
    for name, pts in series.items():
        ys = np.array([p[0] for p in pts])
        errs = np.array([p[1] for p in pts])
        ax.errorbar(eps_list, ys, yerr=errs, capsize=3, elinewidth=0.8,
                    label=name, **STYLE[name])
    ax.set_xlabel(r"perturbation radius $\varepsilon$ ($L_\infty$)")
    ax.set_ylabel("certified robust rate (fraction)")
    # Log radius axis: the decisive divergence (eps = 0.01-0.05) gets the
    # visual width it deserves; the zero-line plateau at eps >= 0.05 is
    # compressed to the right instead of squeezing the informative range.
    ax.set_xscale("log")
    ax.set_xlim(0.008, 0.24)
    ax.set_ylim(-0.03, 1.03)
    # Ticks aligned to the measured radii so data points sit on gridlines.
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2])
    ax.set_xticklabels([r"0.01", r"0.02", r"0.05", r"0.1", r"0.2"])
    ax.minorticks_off()
    # Legend outside the plot area (top, horizontal) so it never covers data.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4,
              fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig6_cert_rates.pdf")


def fig3_existence():
    """Two-moons witnessed-vulnerability rate (data from Sec. 7.4.2)."""
    with open(os.path.join(SRC_RESULTS, "existence_aggregated.json")) as f:
        d = json.load(f)
    eps_list = sorted([float(k) for k in d])
    cl = [d[str(e)]["classical_success_on_nontrivial"] * 100 for e in eps_list]
    s2 = [d[str(e)]["safe2_success_on_nontrivial"] * 100 for e in eps_list]
    nbox = [d[str(e)]["nontrivial_boxes"] for e in eps_list]
    x = np.arange(len(eps_list))
    width = 0.32
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    b1 = ax.bar(x - width / 2, cl, width, color="#4c72b0", edgecolor="black",
                linewidth=0.5, label="classical iteration")
    b2 = ax.bar(x + width / 2, s2, width, color="#2f6f4f", edgecolor="black",
                linewidth=0.5, hatch="//", label="SAFE2 gated relaxation")
    for xi, (v1, v2) in enumerate(zip(cl, s2)):
        ax.text(xi - width / 2, v1 + 1.2, f"{v1:.1f}", ha="center", fontsize=7)
        ax.text(xi + width / 2, v2 + 1.2, f"{v2:.1f}", ha="center", fontsize=7)
    # n annotations folded into the radius tick labels (two lines), so the
    # annotation cannot collide with the legend or the significance bracket.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{e:g}\n($n$={n})" for e, n in zip(eps_list, nbox)],
                       fontsize=7)
    # Significance bracket on the last radius (eps = 0.5): exact McNemar
    # on discordant pairs (b = 14, c = 0) over the 200 nontrivial boxes.
    i_sig = len(eps_list) - 1
    y_top = max(cl[i_sig], s2[i_sig])
    ax.plot([x[i_sig] - width, x[i_sig] + width],
            [y_top + 3.5, y_top + 3.5], color="0.3", lw=0.8)
    ax.plot([x[i_sig] - width, x[i_sig] - width],
            [y_top + 2.5, y_top + 3.5], color="0.3", lw=0.8)
    ax.plot([x[i_sig] + width, x[i_sig] + width],
            [y_top + 2.5, y_top + 3.5], color="0.3", lw=0.8)
    # Significance annotation: two short stacked lines per note, centered on
    # the bar pair; the x-limit leaves >= 1 unit of axes room to the right of
    # the pair (x=1.5) so even the widest line (half-width ~28 pt) stays
    # inside the axes boundary (reported "text beyond the plot
    # border"). Headroom below ylim 64.
    ax.set_xlim(-0.55, 2.55)
    ax.text(x[i_sig], y_top + 5.7,
            r"$^{*}p = 1.22\times10^{-4}$", ha="center", fontsize=7)
    ax.text(x[i_sig], y_top + 7.0, "(exact McNemar)", ha="center",
            fontsize=7)
    ax.text(x[i_sig], y_top + 10.0, r"$+14$ extra boxes", ha="center",
            fontsize=7, color="0.25")
    ax.text(x[i_sig], y_top + 11.3, r"($n=200$)", ha="center", fontsize=7,
            color="0.25")
    ax.set_xlabel(r"perturbation radius $\varepsilon$ (input units)")
    ax.set_ylabel("witnessed-vulnerability rate (%)")
    ax.set_ylim(0, 64)
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig8_existence.pdf")


def fig4_contraction():
    """Contraction trajectory on a representative two-moons box (linear scale)."""
    with open(os.path.join(SRC_RESULTS, "protocol_demo.json")) as f:
        d = json.load(f)
    w0 = 2.0 * float(d["eps"])
    w_cl = [w0] + d["classical_widths"]
    w_s2 = [w0] + d["safe2_widths"]
    xs_s2 = range(len(w_s2))
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    ax.plot(list(xs_s2), w_s2, marker="^", ms=4, lw=1.8, ls="-.",
            color="#2f6f4f", label="SAFE2 gated relaxation")
    ax.plot(range(len(w_cl)), w_cl, marker="s", ms=5, lw=1.5, ls="--",
            color="#4c72b0", label="classical iteration")
    stall_x, stall_y = len(w_cl) - 1, w_cl[-1]
    ax.annotate("classical stalls", xy=(stall_x, stall_y),
                xytext=(stall_x + 1.1, stall_y + 0.004),
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
                color="0.2", fontsize=7)
    ax.set_xlabel("contraction iteration")
    ax.set_ylabel("box width")
    ax.set_xlim(-0.3, len(w_s2) - 0.3)
    ax.set_ylim(0.65, 0.74)
    # Legend below the plot area so it cannot overlap the curves.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2,
              fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig9_contraction.pdf")


def fig5_scaling():
    """Interval-image evaluation time vs. dimension (log-log).

    Uses results/scaling_r2.json (the same dataset as Table tab:scale:
    pure interval-Krawczyk image step, single evaluation, mean over
    REPEATS repetitions; no forward pass / gradient hull included).
    """
    import json as _json
    with open(os.path.join(RESULTS, "scaling_r2.json")) as f:
        data = _json.load(f)
    d_vals = [r["d"] for r in data["rows"]]
    t_struct = [r["t_struct_s"] for r in data["rows"]]
    t_dense = [r["t_dense_s"] for r in data["rows"]]
    fig, ax = plt.subplots(figsize=(4.4, 3.35))
    ax.loglog(d_vals, t_struct, marker="o", ms=4, lw=1.5, ls="-",
              color="#2f6f4f", label="structured image, $O(d)$")
    ax.loglog(d_vals, t_dense, marker="s", ms=4, lw=1.5, ls="--",
              color="#c44e52", label="dense image, $O(d^2)$")
    ax.set_xlabel("input dimension $d$")
    ax.set_ylabel("evaluation time (s)")
    # Ticks aligned to the measured powers of two (d in {2^6..2^14}).
    ax.set_xticks([2 ** 6, 2 ** 8, 2 ** 10, 2 ** 12, 2 ** 14])
    ax.set_xticklabels([r"$2^{6}$", r"$2^{8}$", r"$2^{10}$", r"$2^{12}$",
                        r"$2^{14}$"])
    # y-limit above the dense endpoint at d=2^14 (2.55 s) so the quadratic
    # curve is not clipped.
    ax.set_ylim(1e-4, 4.0)
    # Legend above the plot area, out of the data region.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2,
              fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig10_scaling.pdf")


if __name__ == "__main__":
    fig2_cert_rates()
    fig3_existence()
    fig4_contraction()
    fig5_scaling()
    print("all figures written to", FIG_DIR)
