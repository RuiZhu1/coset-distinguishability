"""Figures for papers/exchange-rates (run from anywhere):

    ~/miniconda3/envs/lcd/bin/python papers/exchange-rates/figures/make_figures.py

Reads results/p1_exchange_rate.json (read-only) and writes, next to this script,
  fig_surface_rates.pdf   R^(d)/c vs d at the six work points, and R_alpha/c (fit window d = 5-11)
  fig_free_order.pdf      the free order of N_cc(p, e) seen from one point (two scalars p and mu)
Also prints the numbers used in Table 3 of the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "results" / "p1_exchange_rate.json"

# categorical slots 1-3 of the reference palette (blue, orange, aqua), fixed order by e0
COL = {0.0: "#2a78d6", 0.02: "#eb6834", 0.05: "#1baf7a"}
MARK = {0.0: "o", 0.02: "s", 0.05: "D"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def fit(point, window=(5, 11)):
    for f in point["fits"]:
        if list(f["window"]) == list(window) and f["weighted"]:
            return f
    raise KeyError(window)


def fig_surface_rates(data):
    pts = data["points"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, p0 in zip(axes, (0.04, 0.06)):
        for k, e0 in enumerate((0.0, 0.02, 0.05)):
            P = next(q for q in pts if q["p0"] == p0 and q["e0"] == e0)
            c = P["c"]
            rows = [r for r in P["per_d"] if r["d"] <= 11]
            d = np.array([r["d"] for r in rows]) + (k - 1) * 0.18
            y = np.array([r["R"] for r in rows]) / c
            s = np.array([r["R_se"] for r in rows]) / c
            ax.errorbar(d, y, yerr=s, color=COL[e0], marker=MARK[e0], ms=4, lw=1.4, capsize=2,
                        label=fr"$e_0={e0:g}$")
            f = fit(P)
            ax.errorbar([13.2 + (k - 1) * 0.3], [f["R_alpha"] / c], yerr=[f["R_alpha_se"] / c], color=COL[e0],
                        marker=MARK[e0], ms=4, mfc="white", capsize=2, lw=1.4)
        ax.axhline(1.0, color=INK, lw=1.0, ls="--")
        ax.text(5.0, 1.03, r"bound $R=c$", color=INK, fontsize=7.5, va="bottom")
        ax.axvline(12.2, color=MUTED, lw=0.6, ls=":")
        ax.set_xticks([5, 7, 9, 11, 13.2])
        ax.set_xticklabels(["5", "7", "9", "11", r"$R_\alpha$"])
        ax.set_xlabel(r"distance $d$")
        ax.set_title(fr"$p_0={p0:g}$", fontsize=9, color=INK)
        ax.set_ylim(-0.05, 1.25)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
    axes[0].set_ylabel(r"marginal rate $/\,c$")
    axes[1].legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    out = HERE / "fig_surface_rates.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_free_order():
    p0, e0 = 0.06, 0.25
    lam = lambda p: 1 - 4 * p / 3  # noqa: E731
    mu0 = (1 - e0) * lam(p0)
    c = (0.75 - p0) / (1 - e0)
    e = np.linspace(0, 0.75, 400)
    # boundary of the reachable set: p' = p0 for e' >= e0 ; mu' = mu0 for e' < e0
    p_mu = 0.75 * (1 - mu0 / (1 - e))  # mu(p, e) = mu0
    fig, ax = plt.subplots(figsize=(3.3, 2.8))
    upper = 0.75
    left = e <= e0
    ax.fill_between(e, np.where(left, p_mu, p0), upper, color="#2a78d6", alpha=0.18, lw=0)
    ax.plot(e[left], p_mu[left], color="#2a78d6", lw=1.8)
    ax.plot([e0, 0.75], [p0, p0], color="#2a78d6", lw=1.8)
    ax.plot(e[~left], p_mu[~left], color="#2a78d6", lw=1.0, ls=":")
    # trade region: e' >= e0, p' < p0, mu' <= mu0 (unreachable; single-loss status open)
    tr = ~left & (p_mu < p0)
    ax.fill_between(e[tr], np.maximum(p_mu[tr], 0), p0, color="#eb6834", alpha=0.25, lw=0)
    # tangent of slope -c at x0
    t = np.linspace(-0.12, 0.12, 2)
    ax.plot(e0 + t, p0 - c * t, color=INK, lw=0.9, ls="--")
    ax.plot([e0], [p0], "o", color=INK, ms=5, zorder=5)
    ax.annotate(r"$x_0=(p_0,e_0)$", (e0, p0), xytext=(0.33, 0.17), fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.text(0.40, 0.45, "reachable from $x_0$\n($p'\\geq p_0$, $\\mu'\\leq\\mu_0$)", fontsize=8, color=INK, ha="center")
    ax.text(0.58, 0.10, "trade\nregion", fontsize=7.5, color=INK, ha="center")
    ax.text(0.02, 0.70, r"slope $-c$ at $x_0$", fontsize=7.5, color=INK)
    ax.annotate("", xy=(e0 - 0.09, p0 + c * 0.09), xytext=(0.08, 0.68),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.6))
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel(r"erasure rate $e'$")
    ax.set_ylabel(r"depolarizing rate $p'$")
    fig.tight_layout()
    out = HERE / "fig_free_order.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def print_table(data):
    print("p0    e0    c       R5            R7            R9            R11           Ra(5-11)       Ra/c   R_B")
    for P in data["points"]:
        r = {x["d"]: x for x in P["per_d"]}
        f = fit(P)
        cells = " ".join(f"{r[d]['R']:.3f}+-{r[d]['R_se']:.3f}" for d in (5, 7, 9, 11))
        print(f"{P['p0']:.2f}  {P['e0']:.2f}  {P['c']:.3f}  {cells}  {f['R_alpha']:.3f}+-{f['R_alpha_se']:.3f}  "
              f"{f['R_alpha'] / P['c']:.2f}  {P['R_B']:.3f}")
    print("\nwindow dependence of R_alpha (weighted fits):")
    for P in data["points"]:
        s = "  ".join(f"{w[0]}-{w[1]}: {fit(P, w)['R_alpha']:.3f}+-{fit(P, w)['R_alpha_se']:.3f}"
                      for w in ((5, 9), (5, 11), (7, 11), (5, 13)))
        print(f"  ({P['p0']:.2f},{P['e0']:.2f})  {s}")


if __name__ == "__main__":
    data = json.loads(DATA.read_text())
    print(fig_surface_rates(data))
    print(fig_free_order())
    print_table(data)
    print("bootstrap replicates recorded in the JSON:", data["bootstrap"], "| commit", data["git_commit"][:7],
          "| git_dirty_src", data["git_dirty_src"])
