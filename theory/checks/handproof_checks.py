"""Checks of the pen-and-paper pass (theory sec 3.9, 4.5-4.6, 5.4).  Light: numpy only, a few seconds, < 300 MB.

  H1  Proposition 3.36: the class polynomials of [[5,1,3]] (unflagged, flagged at the error qubit, flagged elsewhere)
      equal the table in the notes for every syndrome, and the stated factorizations hold
  H2  Proposition 4.22: self-avoiding-walk counts c_m on Z^2 for m <= 14 by enumeration, against the cited values;
      mu_K, gamma_K and the ranges of validity quoted in the notes (4.29%, 4.56%, 4.93%, 5.6%)
  H3  Propositions 4.21, 4.22 and Theorem 4.15 against data: the Peierls / SAW bounds must lie above the measured
      MWPM failure rates (results/p1_mwpm_margin.json) and above the ML failure rates (results/p1_exchange_rate.json,
      results/cc_certificate.json) wherever they are valid; violation = bound below the lower end of the
      certified interval of the estimate
  H4  Theorem 4.18 and Lemma 4.19 on a small detector error model with K = 2 locations (repetition code, two rounds,
      measurement errors) and planted bursts: per-logical BC <= tilted product bound, eps* <= 1/2 sum BC <=
      1/2 W^L(B_bar), exact enumeration; detector firing probability >= (1 - prod lambda)/2
  H5  Theorem 5.9: D_pattern = D_count + Lambda_1 D(pi_1 || pi_0) on random marked Poisson models, and the
      weak-signal limit 1 + chi^2

Run:  python theory/checks/handproof_checks.py        Exit status non-zero if a check fails.
"""
from __future__ import annotations

import itertools
import json
import sys
from math import exp, log, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from completion_checks import class_polys, code_structure  # noqa: E402
from exact_small_codes import five_qubit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def poly(*c):
    return np.array(c + (0,) * (6 - len(c)), dtype=object)


# ----------------------------------------------------------------------------------------------------------------
def h1_five_qubit_table() -> None:
    print("\n=== H1 class polynomials of [[5,1,3]] (Prop. 3.36) ===")
    st = code_structure(five_qubit())
    n = st.n
    W0 = class_polys(st, ())
    Wf = [class_polys(st, (j,)) for j in range(n)]
    syn_of = np.full(st.ncls, -1)
    syn_of[st.cls] = st.syn
    wt = st.nz.sum(axis=1)
    ok = True
    # syndrome 0: the class of the identity (index 0)
    c0 = int(st.cls[0])
    others = [int(c) for c in np.flatnonzero(syn_of == syn_of[c0]) if c != c0]
    ok &= list(W0[c0]) == list(poly(1, 0, 0, 0, 15, 0))
    ok &= all(list(W0[c]) == list(poly(0, 0, 0, 10, 0, 6)) for c in others)
    for j in range(n):
        ok &= list(Wf[j][c0]) == list(poly(1, 0, 0, 12, 3, 0))
        ok &= all(list(Wf[j][c]) == list(poly(0, 0, 6, 4, 6, 0)) for c in others)
    n_syn = 1
    # single-error syndromes
    for e in np.flatnonzero(wt == 1):
        k = int(np.flatnonzero(st.nz[e])[0])
        cs = int(st.cls[e])
        oth = [int(c) for c in np.flatnonzero(syn_of == syn_of[cs]) if c != cs]
        n_syn += 1
        ok &= list(W0[cs]) == list(poly(0, 1, 0, 4, 8, 3))
        ok &= all(list(W0[c]) == list(poly(0, 0, 2, 4, 6, 4)) for c in oth)
        ok &= list(Wf[k][cs]) == list(poly(1, 0, 0, 12, 3, 0))
        ok &= all(list(Wf[k][c]) == list(poly(0, 0, 6, 4, 6, 0)) for c in oth)
        for j in range(n):
            if j != k:
                ok &= list(Wf[j][cs]) == list(poly(0, 1, 3, 7, 5, 0))
                ok &= all(list(Wf[j][c]) == list(poly(0, 1, 3, 7, 5, 0)) for c in oth)
    check("H1 table of Prop. 3.36 for all 16 syndromes and all single flags", ok and n_syn == 16, f"({n_syn} syndromes)")
    P = np.polynomial.polynomial
    one_m = [1, -1]
    cube = P.polypow(one_m, 3)
    facts = [
        (P.polysub([1, 0, 0, 0, 15], [0, 0, 0, 10, 0, 6]), P.polymul(cube, [1, 3, 6])),
        (P.polysub([0, 1, 0, 4, 8, 3], [0, 0, 2, 4, 6, 4]), P.polymul(P.polymul([0, 1], cube), [1, 1])),
        (P.polysub([1, 0, 0, 12, 3], [0, 0, 6, 4, 6]), P.polymul(cube, [1, 3])),
    ]
    check("H1 factorizations (1-r)^3 (1+3r+6r^2), r(1-r)^3(1+r), (1-r)^3(1+3r)",
          all(np.allclose(np.trim_zeros(a, 'b'), np.trim_zeros(b, 'b')) for a, b in facts))


# ----------------------------------------------------------------------------------------------------------------
CITED = [1, 4, 12, 36, 100, 284, 780, 2172, 5916, 16268, 44100, 120292, 324932, 881500, 2374444]
C20_CITED = 897697164


def saw_counts(mmax: int) -> list[int]:
    """c_m for m <= mmax: walks whose first step is +x, times 4."""
    counts = [0] * (mmax + 1)
    counts[0] = 1
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    visited = {(0, 0), (1, 0)}
    stack = [((1, 0), 1, iter(steps))]
    counts[1] += 1
    while stack:
        pos, m, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop()
            visited.discard(pos)
            continue
        q = (pos[0] + nxt[0], pos[1] + nxt[1])
        if q in visited or m == mmax:
            continue
        visited.add(q)
        counts[m + 1] += 1
        stack.append((q, m + 1, iter(steps)))
    return [counts[0]] + [4 * c for c in counts[1:]]


def p_threshold(mu: float) -> float:
    """Largest p with mu * B_M(p, 0) < 1, B_M = 2 sqrt(q(1-q)), q = 2p/3."""
    x = 1 / mu / 2
    q = (1 - sqrt(1 - 4 * x * x)) / 2
    return 1.5 * q


def h2_saw() -> tuple[float, float]:
    print("\n=== H2 self-avoiding walks (Prop. 4.22) ===")
    c = saw_counts(14)
    check("H2 enumerated c_m (m <= 14) equal the cited values", c == CITED, f"(c_14 = {c[14]})")
    mu10 = c[10] ** 0.1
    g10 = max(c[r] * mu10 ** -r for r in range(10))
    mu20 = C20_CITED ** 0.05
    print(f"    mu_10 = {mu10:.4f}, gamma_10 = {g10:.4f}, mu_20 (cited c_20) = {mu20:.4f}")
    th = {"3": p_threshold(3.0), "mu10": p_threshold(mu10), "mu20": p_threshold(mu20), "mu": p_threshold(2.638)}
    print("    ranges at e = 0: " + ", ".join(f"{k}: p < {100 * v:.2f}%" for k, v in th.items()))
    ok = abs(100 * th["3"] - 4.29) < 0.01 and abs(100 * th["mu10"] - 4.56) < 0.01 and abs(100 * th["mu20"] - 4.93) < 0.01 \
        and abs(100 * th["mu"] - 5.6) < 0.05 and abs(g10 - 1.46) < 0.01
    check("H2 numbers quoted in the notes (4.29, 4.56, 4.93, 5.6 %, gamma_10 = 1.46)", ok)
    return mu10, g10


# ----------------------------------------------------------------------------------------------------------------
def b_m(p, e):
    q = 2 * p / 3
    return e + (1 - e) * 2 * sqrt(q * (1 - q))


def peierls(d, p, e):
    B = b_m(p, e)
    return 2 * d * B * (3 * B) ** (d - 1) / (1 - 3 * B) if 3 * B < 1 else np.inf


def saw_bound(d, p, e, mu, g):
    B = b_m(p, e)
    return 4 * d * g * B * B * (mu * B) ** (d - 2) / (1 - mu * B) if mu * B < 1 else np.inf


def h3_data(mu10, g10) -> None:
    print("\n=== H3 Peierls / SAW bounds against MWPM and ML data (Thm 4.15, Props 4.21, 4.22) ===")
    rows = []
    for f, kind in (("p1_mwpm_margin.json", "MWPM"), ("p1_exchange_rate.json", "ML")):
        for pt in json.load(open(REPO / "results" / f))["points"]:
            for r in pt["per_d"]:
                lo = r.get("p_L_cert", [r["p_L"] - 3 * r["p_L_se"]])[0]
                rows.append((kind, pt["p0"], pt["e0"], r["d"], r["p_L"], lo))
    for pt in json.load(open(REPO / "results" / "cc_certificate.json"))["points"]:
        for d, v in pt["eps"].items():
            lo = v.get("cert_interval", [v["eps"] - 3 * v["se"]])[0]
            rows.append(("ML", pt["p"], pt["e"], int(d), v["eps"], lo))
    tested, worst = 0, np.inf
    for kind, p, e, d, pl, lo in rows:
        b = min(peierls(d, p, e), saw_bound(d, p, e, mu10, g10))
        if np.isfinite(b):
            tested += 1
            worst = min(worst, b / max(lo, 1e-300))
            if b < lo:
                print(f"    violation: {kind} p={p} e={e} d={d}: bound {b:.3e} < {lo:.3e}")
    check("H3 bounds above every valid data point", worst >= 1, f"({tested} points tested; smallest bound/estimate ratio {worst:.3g})")


# ----------------------------------------------------------------------------------------------------------------
def h4_small_dem() -> None:
    print("\n=== H4 space-time burst bound on a small DEM (Thm 4.18, Lemma 4.19) ===")
    # locations: data flips before round 1 (x1_0..2), before round 2 (x2_0..2), measurement flips in round 1 (m_0, m_1)
    L = 8
    H = np.array([[1, 1, 0], [0, 1, 1]], np.uint8)
    faults = np.array(list(itertools.product([0, 1], repeat=L)), np.uint8)
    x1, x2, m = faults[:, 0:3], faults[:, 3:6], faults[:, 6:8]
    det = np.concatenate([(x1 @ H.T + m) % 2, (x2 @ H.T + m) % 2], axis=1)
    obs = (x1[:, 0] + x2[:, 0]) % 2
    det_id = det @ (1 << np.arange(4))
    G_idx = [i for i in range(len(faults)) if not det[i].any() and obs[i]]
    regions = [(0, 1, 6), (4, 5, 7), (2, 3)]
    worst_G, worst_eps, worst_det, ncase = -np.inf, -np.inf, -np.inf, 0
    for p in (0.01, 0.05):
        for rho in (0.05, 0.3):
            for theta in (0.3, 1.0):
                P = np.zeros(len(faults))
                for act in itertools.product([0, 1], repeat=len(regions)):
                    w = float(np.prod([rho if a else 1 - rho for a in act]))
                    s = np.full(L, p)
                    for a, R in zip(act, regions):
                        if a:
                            for l in R:
                                s[l] = (1 - theta) * s[l] + theta * 0.5
                    P += w * np.prod(np.where(faults == 1, s, 1 - s), axis=1)
                # exact ML
                tab = np.zeros((16, 2))
                np.add.at(tab, (det_id, obs), P)
                eps = 1 - tab.max(axis=1).sum()
                idx = {tuple(f): i for i, f in enumerate(faults)}
                bcs = []
                for gi in G_idx:
                    shifted = np.array([idx[tuple((f + faults[gi]) % 2)] for f in faults])
                    bcs.append(np.sqrt(P * P[shifted]).sum())
                for t in (p, 0.2):
                    tau = t / (1 - t)
                    msp = lambda s_: (1 - s_) * sqrt(tau) + s_ / sqrt(tau)
                    m0, kap = msp(p), msp(0.5) / msp(p)
                    rbar = max(sum(rho for R in regions if l in R) for l in range(L))
                    V = max(len(R) for R in regions)
                    lBbar = log(m0) + rbar * (kap ** V - 1) / V          # logs: B_bar can exceed the float range
                    for gi, bc in zip(G_idx, bcs):
                        supp = set(np.flatnonzero(faults[gi]))
                        ltilt = len(supp) * log(m0) + sum(rho * (kap ** len(supp & set(R)) - 1) for R in regions)
                        worst_G = max(worst_G, log(bc) - ltilt, ltilt - len(supp) * lBbar)
                    terms = np.array([int(faults[gi].sum()) * lBbar for gi in G_idx])
                    lW = terms.max() + log(np.exp(terms - terms.max()).sum())   # log W^L(B_bar)
                    worst_eps = max(worst_eps, eps / (0.5 * sum(bcs)) - 1, log(sum(bcs)) - lW)
                    ncase += 1
                lam = 1 - 2 * p
                for D in range(4):
                    sup = int(H[D % 2].sum()) + 1                  # two data locations + one measurement location
                    fire = float((P * det[:, D]).sum())
                    worst_det = max(worst_det, 0.5 * (1 - lam ** sup) - fire)
    check("H4 BC <= tilted product <= B_bar^|G| for every logical (in logs)", worst_G <= 1e-12, f"(max log excess {worst_G:.1e})")
    check("H4 eps* <= 1/2 sum BC <= 1/2 W^L(B_bar)", worst_eps <= 1e-12, f"({ncase} cases; max excess {worst_eps:.1e})")
    check("H4 detector firing >= (1 - prod lambda)/2", worst_det <= 1e-12, f"(max excess {worst_det:.1e})")


# ----------------------------------------------------------------------------------------------------------------
def h5_pattern_gain() -> None:
    print("\n=== H5 pattern gain (Thm 5.9) ===")
    rng = np.random.default_rng(1)
    pois = lambda a, b: a * np.log(a / b) - a + b
    worst = 0.0
    for _ in range(200):
        Z = int(rng.integers(2, 12))
        l0 = rng.uniform(0.01, 2, Z)
        l1 = rng.uniform(0.01, 2, Z)
        Dp, Dc = pois(l1, l0).sum(), pois(l1.sum(), l0.sum())
        p1, p0 = l1 / l1.sum(), l0 / l0.sum()
        worst = max(worst, abs(Dp - Dc - l1.sum() * (p1 * np.log(p1 / p0)).sum()))
    check("H5 D_pattern = D_count + Lambda_1 D(pi_1||pi_0)", worst < 1e-12, f"(max error {worst:.1e})")
    l0 = rng.uniform(0.1, 1, 8)
    w = rng.uniform(0, 1, 8)
    eps = 1e-5
    ratio = pois(l0 + eps * w, l0).sum() / pois(l0.sum() + eps * w.sum(), l0.sum())
    wh, p0 = w / w.sum(), l0 / l0.sum()
    check("H5 weak-signal limit 1 + chi^2", abs(ratio - (wh ** 2 / p0).sum()) < 1e-3,
          f"(ratio {ratio:.6f}, 1 + chi^2 = {(wh ** 2 / p0).sum():.6f})")


def main() -> int:
    h1_five_qubit_table()
    mu10, g10 = h2_saw()
    h3_data(mu10, g10)
    h4_small_dem()
    h5_pattern_gain()
    print("\nSUMMARY:", "ALL HANDPROOF CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
