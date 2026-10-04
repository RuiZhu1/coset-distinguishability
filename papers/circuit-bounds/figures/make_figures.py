"""Figures for papers/circuit-bounds (reads the committed results JSON; writes PDFs next to this script).

Run from anywhere:  ~/miniconda3/envs/lcd/bin/python papers/circuit-bounds/figures/make_figures.py
Also prints the summary numbers used in the tables of main.tex (margins s*, data-certificate ranges), so the
tables can be checked against the results files.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = ROOT / "results"


def load(name):
    return json.loads((RES / name).read_text())


plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "pdf.fonttype": 42,
})
# grayscale-safe: identity carried by marker and line style, gray levels only reinforce it
STYLE = {
    "4.28": dict(color="0.65", marker="s", ls=":", label="Thm. 1 (basic)"),
    "4.32": dict(color="0.40", marker="^", ls="--", label="Thm. 2 (geodesic)"),
    "4.34": dict(color="0.0", marker="o", ls="-", label="Thm. 3 (gap)"),
}


# ------------------------------------------------------------------ figure 1
def figure_bounds():
    geo = load("circuit_peierls_geodesic.json")["bounds"]
    cp = load("circuit_peierls.json")["circuit"]
    gc = load("geodesic_checks.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, p in zip(axes, (1e-3, 3e-3)):
        for key, col in (("4.28", "theorem_4_28"), ("4.32", "theorem_4_32"), ("4.34", "theorem_4_34")):
            pts = {r["d"]: r[col] for r in geo if math.isclose(r["p"], p)}
            if key == "4.28":   # Theorem 4.28 is tabulated to d = 15 in circuit_peierls.json
                pts.update({r["d"]: r["bound"] for r in cp["table"] if math.isclose(r["p"], p)})
            ds = sorted(d for d, v in pts.items() if v is not None and np.isfinite(v))
            ax.plot(ds, [pts[d] for d in ds], markersize=5, markerfacecolor="white", **STYLE[key])
        # simulated pymatching rates with 99% Clopper-Pearson intervals
        sim = {}
        for r in cp["simulation"]:
            if math.isclose(r["p"], p):
                sim[r["d"]] = (r["rate"], r["cp99"][0], r["cp99"][1])
        for r in gc["G3"]:
            if math.isclose(r["p"], p) and r["d"] not in sim:
                sim[r["d"]] = (r["rate"], r["low"], r["high"])
        ds = sorted(sim)
        y = np.array([sim[d][0] for d in ds])
        lo = y - np.array([sim[d][1] for d in ds])
        hi = np.array([sim[d][2] for d in ds]) - y
        ax.errorbar(ds, y, yerr=[lo, hi], fmt="D", color="0.0", markerfacecolor="0.0", markersize=4, capsize=2,
                    ls="none", label="pymatching, sampled")
        ax.set_yscale("log")
        ax.set_xlabel("distance $d$ ($d$ rounds)")
        ax.set_title(f"$p = {p * 1e3:.0f}\\times10^{{-3}}$", fontsize=9)
        ax.set_xticks([3, 5, 7, 9, 11, 13, 15] if p == 1e-3 else [3, 5, 7, 9])
        ax.axhline(1.0, color="0.8", lw=0.8, zorder=0)
        ax.grid(axis="y", color="0.92", lw=0.6, which="major")
    axes[0].set_ylabel("logical failure probability per shot")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(HERE / "bounds_vs_distance.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ hardware margins
def margin_groups():
    """(label, {theorem: list of s*}) for Willow (RL-optimized prior, 10 rounds) and Sycamore (p_ij models, >= 15 rounds;
    per experiment the mean over the two p_ij models, as in the summary of hardware_gap.json)."""
    E = load("hardware_gap.json")["experiments"]
    out = []
    for d in (3, 5, 7):
        rows = [x for x in E if x["device"] == "willow" and x["code"] == "surface_code" and x["d"] == d
                and x["rounds"] == 10]
        out.append((f"Willow\n$d={d}$", len(rows), {
            k: [x["models"]["rl_optimized"][f"s_star{suf}"] for x in rows]
            for k, suf in (("4.28", ""), ("4.32", "_4_32"), ("4.34", "_4_34"))}))
    for d in (3, 5):
        rows = [x for x in E if x["device"] == "sycamore" and x["code"] == "surface_code" and x["d"] == d
                and x["rounds"] >= 15]
        out.append((f"Sycamore\n$d={d}$", len(rows), {
            k: [x["s_star"][f"pij{suf}"] for x in rows]
            for k, suf in (("4.28", ""), ("4.32", "_4_32"), ("4.34", "_4_34"))}))
    return out


def figure_margins(groups):
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    off = {"4.28": -0.22, "4.32": 0.0, "4.34": 0.22}
    for i, (_, _, g) in enumerate(groups):
        for k, v in g.items():
            v = np.array(v, float)
            m = np.median(v)
            ax.errorbar([i + off[k]], [m], yerr=[[m - v.min()], [v.max() - m]], fmt=STYLE[k]["marker"],
                        color=STYLE[k]["color"], markerfacecolor="white" if k != "4.34" else "0.0",
                        markersize=6, capsize=2.5, lw=1.2, label=STYLE[k]["label"] if i == 0 else None)
    ax.axhline(1.0, color="0.0", lw=0.8, ls="-")
    ax.text(len(groups) - 0.55, 1.04, "fitted noise ($s=1$)", fontsize=7.5, ha="right", va="bottom")
    ax.axhline(4.0, color="0.75", lw=0.6, ls="--")
    ax.text(len(groups) - 0.55, 3.85, "search cap ($s=4$)", fontsize=7, ha="right", va="top", color="0.35")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("margin $s^\\ast$")
    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "4"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.15, 5)
    ax.legend(frameon=False, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.16))
    fig.tight_layout()
    fig.savefig(HERE / "hardware_margins.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ printed numbers
def print_margins(groups):
    print("== hardware margins s* (median [min, max], n, capped at 4)")
    for lab, n, g in groups:
        s = "  ".join(f"{k}: {np.median(v):.3f} [{min(v):.3f}, {max(v):.3f}] cap={sum(x >= 4 for x in v)}"
                      for k, v in g.items())
        print(lab.replace("\n", " "), f"n={n}", s)


def print_data_certificate():
    d = load("data_certificate.json")
    g = defaultdict(list)
    for r in d["willow"] + d["sycamore"]:
        g[(r["device"], r["d"], r["rounds"])].append(r)
    thr_drift = None
    print("== data certificate (paired intervals, Theorem 4.32; ranges over experiments)")
    for k, rs in sorted(g.items()):
        def rng(f):
            v = [f(r) for r in rs]
            v = [x for x in v if x is not None]
            return f"[{min(v):.3g}, {max(v):.3g}]" if v else "-"
        c = lambda r: r["certificates"]["paired"]["bound_4_32"]  # noqa: E731
        ratio = lambda r: c(r) / r["mwpm_observed"]["rate"] if np.isfinite(c(r)) else None  # noqa: E731
        ne = sum(r["diagnostics"]["nonedge"]["exceed"] > 0 for r in rs)
        # drift threshold: Bonferroni over the chunks at level delta (two-sided normal)
        from scipy.stats import norm
        dr = [r["diagnostics"]["stationarity"] for r in rs]
        thr = [float(norm.isf(d["meta"]["delta"] / (2 * x["chunks"]))) for x in dr]
        drift = sum(x["max_abs_z"] > t for x, t in zip(dr, thr))
        both = [r for r, x, t in zip(rs, dr, thr) if r["diagnostics"]["nonedge"]["exceed"] == 0 and x["max_abs_z"] <= t]
        thr_drift = thr[0]
        print(k, f"n={len(rs)} cert {rng(c)} point {rng(lambda r: r['certificates']['paired']['bound_4_32_point'])}"
                 f" obs {rng(lambda r: r['mwpm_observed']['rate'])} ratio {rng(ratio)}"
                 f" (T) {rng(lambda r: r['certificates']['paired+T']['bound_4_32'])}"
                 f" Thm4.28 {rng(lambda r: r['certificates']['paired']['bound_4_28'])}"
                 f" margin {rng(lambda r: r['certificates']['paired'].get('margin_4_32'))}"
                 f" nonedge>thr {ne} drift>thr {drift} pass-both {len(both)}"
                 f" cert(pass) {'[%.3g, %.3g]' % (min(c(r) for r in both), max(c(r) for r in both)) if both else '-'}"
                 f" obs(pass) {'[%.3g, %.3g]' % (min(r['mwpm_observed']['rate'] for r in both), max(r['mwpm_observed']['rate'] for r in both)) if both else '-'}")
    print("drift threshold (50 chunks):", thr_drift)


def print_tightness():
    gc = load("geodesic_checks.json")
    cp = load("circuit_peierls.json")["circuit"]
    print("== bound / sampled rate (G3, G6 rows)")
    for r, r6 in zip(gc["G3"], gc["G6"]):
        print(r["d"], r["p"], "rate", r["rate"], "4.28", r["theorem_4_28"] / r["rate"], "4.32", r["geodesic"] / r["rate"],
              "4.34", r6["gap"] / r6["rate"])
    print("== Theorem 4.28 / sampled (circuit_peierls.json simulation)")
    for r in cp["simulation"]:
        print(r["d"], r["p"], r["rate"], r["bound"] / r["rate"])


if __name__ == "__main__":
    figure_bounds()
    groups = margin_groups()
    figure_margins(groups)
    print_margins(groups)
    print_data_certificate()
    print_tightness()
