"""Finite-size-scaling fit of the bit-flip MWPM threshold (README sec. 2, sanity check "pure bit-flip noise, MWPM ~ 10.3%").

Data: results/threshold_check_bitflip.csv (d = 5..15, 1e5 shots, wide grid) and results/threshold_check_bitflip_large.csv
(d = 11..41, 4e5 shots, p in [0.094, 0.106]).  Ansatz (Wang-Harrington-Preskill):

    p_L = A + B x + C x^2,   x = (p - p_th) d^(1/nu)                                   ("plain")
    p_L = A + B x + C x^2 + D d^(-1/mu)                                                ("boundary correction")

fitted by weighted least squares for several minimum distances d_min (only points within the narrow window |p - 0.1| <= 0.006),
with parameter errors from a parametric bootstrap (400 replicates).  Also prints the consecutive-pair crossings.

    python experiments/p1_erasure_pauli/threshold_fss.py
Writes results/threshold_fss_bitflip.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[2]
FILES = ["threshold_check_bitflip.csv", "threshold_check_bitflip_large.csv"]
WINDOW = (0.094 - 1e-9, 0.106 + 1e-9)


def load():
    rows = {}
    for name in FILES:
        with open(REPO / "results" / name) as fh:
            for r in csv.DictReader(fh):
                d, x = int(r["d"]), float(r["x"])
                if not WINDOW[0] <= x <= WINDOW[1]:
                    continue
                key = (d, round(x, 6))
                # where both files have a point, keep the one with more shots
                if key not in rows or int(r["shots"]) > rows[key][3]:
                    rows[key] = (d, x, float(r["p_L"]), int(r["shots"]), float(r["se"]))
    a = np.array(sorted(rows.values()))
    return a[:, 0], a[:, 1], a[:, 2], a[:, 4]


def model(t, d, p, corr):
    pth, nu, A, B, C = t[:5]
    x = (p - pth) * d ** (1 / nu)
    m = A + B * x + C * x * x
    return m + t[5] * d ** (-1 / t[6]) if corr else m


def fit(d, p, y, s, corr):
    x0 = [0.10, 1.5, 0.12, 1.0, 1.0] + ([0.0, 1.0] if corr else [])
    lb = [0.08, 0.5, -1, -10, -100] + ([-5, 0.1] if corr else [])
    ub = [0.13, 4.0, 1, 10, 100] + ([5, 10] if corr else [])
    r = least_squares(lambda t: (model(t, d, p, corr) - y) / s, x0, bounds=(lb, ub))
    return r.x, float(np.sum(r.fun ** 2)), int(y.size - len(x0))


def main() -> None:
    d, p, y, s = load()
    rng = np.random.default_rng(1)
    out = dict(script="experiments/p1_erasure_pauli/threshold_fss.py", files=FILES, window=list(WINDOW), fits=[], crossings={})
    for dmin in (5, 9, 11, 15, 21):
        for corr in (False, True):
            k = d >= dmin
            if len(np.unique(d[k])) < (4 if corr else 3):
                continue
            t, chi2, dof = fit(d[k], p[k], y[k], s[k], corr)
            boots = []
            for _ in range(400):
                yb = y[k] + s[k] * rng.standard_normal(k.sum())
                boots.append(fit(d[k], p[k], yb, s[k], corr)[0][:2])
            boots = np.array(boots)
            rec = dict(d_min=dmin, d_values=sorted(int(v) for v in np.unique(d[k])), boundary_correction=corr,
                       p_th=float(t[0]), p_th_se=float(boots[:, 0].std()), nu=float(t[1]), nu_se=float(boots[:, 1].std()),
                       chi2_dof=chi2 / dof, extra=[float(v) for v in t[5:]], at_bound=bool(corr and (t[6] >= 9.99 or t[6] <= 0.11)))
            out["fits"].append(rec)
            print(f"d >= {dmin:2d} {'+corr' if corr else 'plain'}: p_th = {rec['p_th'] * 100:.3f} +- {rec['p_th_se'] * 100:.3f} %, "
                  f"nu = {rec['nu']:.2f} +- {rec['nu_se']:.2f}, chi2/dof = {rec['chi2_dof']:.2f}"
                  + ("  (mu at its bound: correction not determined)" if rec["at_bound"] else ""))
    ds = sorted(set(int(v) for v in d))
    from lcd.analysis import crossing
    for d1, d2 in zip(ds[:-1], ds[1:]):
        xs = sorted(set(p[d == d1]) & set(p[d == d2]))
        if len(xs) < 3:
            continue
        get = lambda dd, arr: [arr[(d == dd) & (p == x)][0] for x in xs]
        c, e = crossing(xs, get(d1, y), get(d2, y), get(d1, s), get(d2, s))
        out["crossings"][f"d{d1}_d{d2}"] = dict(crossing=c, se=e)
        print(f"crossing d = {d1} vs {d2}: {c * 100:.3f} +- {e * 100:.3f} %")
    (REPO / "results" / "threshold_fss_bitflip.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
