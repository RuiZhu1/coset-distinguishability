"""Second batch of light checks for the pen-and-paper pass (theory sec 3.9, 4.4, 4.6, 5.4).  Minutes, < 1 GB.

  L1  Corollary 3.35 and Proposition 3.33 on a code library ([[5,1,3]], Steane, d = 3 surface, random [[n,1]] codes,
      n = 4..7): small-p (E1) by the lexicographic criterion, the first-order obstruction, and consistency
      (obstruction => lexicographic failure; lexicographic verdict = float verdict at p = 1e-4)
  L2  Theorem 5.10 under the marking model: -(1/T) ln beta_T -> D for pattern and count tests (importance sampling
      under H1 with the error under H1 held at a = 0.1); the count version is also computed exactly
  L3  Theorem 4.15 / Proposition 4.22 against simulation: MWPM failure rates at p < 4% (with and without erasure),
      d = 3..9, must lie below the Peierls / SAW bounds
  L4  Theorem 4.14 coverage at d = 3: simulated calibration data with Poisson bursts on one plaquette; e_bar (flags),
      p_bar (a weight-2 check off the region, Lemma 4.12(b)), r_bar (one detector inside the region with the
      efficiency eta of Lemma 4.12(c) for burst strengths >= 0.2); coverage of {p <= p_bar, e <= e_bar, r <= r_bar}
      >= 1 - delta, and certificate >= exact eps* (with bursts) on every covering run

Run:  python theory/checks/second_pass_checks.py [--quick]     Writes results/second_pass_checks.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from math import comb, exp, factorial, log, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import beta, poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_small_codes as esc  # noqa: E402
from completion_checks import b_burst, class_polys, code_structure, oplus  # noqa: E402
from handproof_checks import peierls, saw_bound, saw_counts  # noqa: E402
from sharp_codes_scan import random_isotropic  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
failures: list[str] = []
OUT: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


# ----------------------------------------------------------------------------------------------------------------
def lexmax(rows):
    """Indices of the lexicographic maxima of integer coefficient rows (lowest power first)."""
    best = max(tuple(r) for r in rows)
    return {i for i, r in enumerate(rows) if tuple(r) == best}


def analyse_e1(Sg) -> dict:
    st = code_structure(Sg)
    n = st.n
    syn_of = np.full(st.ncls, -1)
    syn_of[st.cls] = st.syn
    W0 = class_polys(st, ())
    Wf = [class_polys(st, (j,)) for j in range(n)]
    wt = st.nz.sum(axis=1)
    r0 = 1e-4 / (3 * (1 - 1e-4))
    lex_fail = obstruction = inconsistent = float_mismatch = 0
    for s in np.unique(st.syn):
        cl = [int(c) for c in np.flatnonzero(syn_of == s)]
        rows0 = [list(W0[c]) for c in cl]
        M0 = lexmax(rows0)
        # float check of the lexicographic verdict at a small rate
        vals = [sum(int(a) * r0 ** k for k, a in enumerate(r)) for r in rows0]
        fM0 = {i for i, v in enumerate(vals) if v >= max(vals) * (1 - 1e-12)}
        float_mismatch += fM0 != M0
        mins = [min(k for k, a in enumerate(r) if a) for r in rows0]
        for j in range(n):
            Mj = lexmax([list(Wf[j][c]) for c in cl])
            fails = not (M0 & Mj)
            lex_fail += fails
            # Corollary 3.35
            if len(M0) == 1:
                i0 = next(iter(M0))
                w = mins[i0]
                if mins.count(w) == 1:
                    cs = cl[i0]
                    in_cs = st.cls == cs
                    if not st.nz[in_cs & (wt == w), j].any():
                        A_star = int((in_cs & (wt == w)).sum()) + int((in_cs & (wt == w + 1) & (st.nz[:, j] == 1)).sum())
                        for i, c in enumerate(cl):
                            if c != cs and mins[i] == w + 1:
                                A_p = int(((st.cls == c) & (wt == w + 1) & (st.nz[:, j] == 1)).sum())
                                if A_p > A_star:
                                    obstruction += 1
                                    inconsistent += not fails
                                    break
    return dict(lex_fail=lex_fail, obstruction=obstruction, inconsistent=inconsistent, float_mismatch=float_mismatch)


def l1_e1_library(per_n: int) -> None:
    print("\n=== L1 small-p (E1): lexicographic criterion and first-order obstruction (Prop 3.33, Cor 3.35) ===")
    rng = np.random.default_rng(20261002)
    lib = [("[[5,1,3]]", esc.five_qubit()), ("Steane", esc.steane()), ("rotated d=3", esc.rotated_surface_d3())]
    for n in range(4, 8):
        lib += [(f"random n={n} #{i}", random_isotropic(n, n - 1, rng)) for i in range(per_n)]
    recs, incons, mism = [], 0, 0
    for name, S in lib:
        r = analyse_e1(np.array(S, np.uint8))
        r["name"] = name
        recs.append(r)
        incons += r["inconsistent"]
        mism += r["float_mismatch"]
        if not name.startswith("random"):
            print(f"    {name}: (E1) fails at small p for {r['lex_fail']} (syndrome, flag) pairs; "
                  f"first-order obstruction at {r['obstruction']}")
    rnd = [r for r in recs if r["name"].startswith("random")]
    n_fail = sum(r["lex_fail"] > 0 for r in rnd)
    n_obs = sum(r["obstruction"] > 0 for r in rnd)
    print(f"    random codes: {len(rnd)}; (E1) fails at small p for {n_fail}; the first-order obstruction detects {n_obs}")
    five = recs[0]
    check("L1 [[5,1,3]] satisfies (E1) at small p", five["lex_fail"] == 0)
    check("L1 Steane satisfies (E1) at small p, d=3 surface does not", recs[1]["lex_fail"] == 0 and recs[2]["lex_fail"] > 0)
    check("L1 obstruction => lexicographic failure (Cor 3.35 is sound)", incons == 0, f"({sum(r['obstruction'] for r in recs)} obstructions)")
    check("L1 lexicographic unflagged maxima = float maxima at p = 1e-4", mism == 0)
    OUT["L1"] = dict(codes=len(recs), random=len(rnd), random_fail_small_p=n_fail, random_detected_by_obstruction=n_obs,
                     named={r["name"]: {k: r[k] for k in ("lex_fail", "obstruction")} for r in recs[:3]})


# ----------------------------------------------------------------------------------------------------------------
def l2_stein(samples: int) -> None:
    print("\n=== L2 Stein form of the ratio of rounds (Thm 5.10), marked Poisson model ===")
    rng = np.random.default_rng(7)
    Z = 6
    lam0 = np.array([0.40, 0.30, 0.20, 0.10, 0.05, 0.05])
    lam1 = lam0 + 0.08 * np.array([0.0, 0.1, 0.1, 0.3, 1.0, 1.5])
    L0, L1 = lam0.sum(), lam1.sum()
    Dp = float((lam1 * np.log(lam1 / lam0) - lam1 + lam0).sum())
    Dc = float(L1 * log(L1 / L0) - L1 + L0)
    a = 0.1
    rows = []
    for T in (50, 200, 800, 3200, 12800):
        N = rng.poisson(T * lam1, size=(samples, Z))
        llr_p = N @ np.log(lam1 / lam0) - T * (L1 - L0)
        Ntot = N.sum(axis=1)
        llr_c = Ntot * log(L1 / L0) - T * (L1 - L0)
        out = {}
        for key, llr in (("pattern", llr_p), ("count", llr_c)):
            tau = np.quantile(llr, a)                       # error under H1 ~ a (decide 0 iff llr <= tau)
            x = -llr[llr > tau]                             # P0(llr > tau) = E_P1[1{llr > tau} e^{-llr}], in logs
            out[key] = -(x.max() + log(np.exp(x - x.max()).sum()) - log(len(llr))) / T
        # exact count test: smallest non-randomized threshold with P1(N <= k) <= a, error under H0 = P0(N > k)
        k = int(poisson.ppf(a, T * L1)) - 1
        while poisson.cdf(k + 1, T * L1) <= a:
            k += 1
        exact_c = -poisson.logsf(k, T * L0) / T
        rows.append(dict(T=T, pattern=out["pattern"], count=out["count"], count_exact=float(exact_c)))
        print(f"    T = {T:5d}: -ln(beta)/T pattern {out['pattern']:.4f} (D = {Dp:.4f}), count {out['count']:.4f} "
              f"(exact {exact_c:.4f}; D = {Dc:.4f})")
    gap_p = [abs(r["pattern"] - Dp) / Dp for r in rows]
    gap_c = [abs(r["count_exact"] - Dc) / Dc for r in rows][2:]       # T >= 800: small T is dominated by discreteness
    check("L2 pattern exponent converges to D_pattern (rel. gap decreasing, < 10% at T = 12800)",
          gap_p[-1] < 0.10 and all(x > y for x, y in zip(gap_p, gap_p[1:])), f"(rel. gaps {', '.join(f'{g:.3f}' for g in gap_p)})")
    check("L2 count exponent converges to D_count (exact, rel. gap decreasing for T >= 800, < 15% at T = 12800)",
          gap_c[-1] < 0.15 and all(x > y for x, y in zip(gap_c, gap_c[1:])), f"(T >= 800; rel. gaps {', '.join(f'{g:.3f}' for g in gap_c)})")
    check("L2 importance-sampled count exponent agrees with the exact one (3%, T >= 800)",
          all(abs(r["count"] - r["count_exact"]) / r["count_exact"] < 0.03 for r in rows if r["T"] >= 800))
    OUT["L2"] = dict(D_pattern=Dp, D_count=Dc, ratio=Dp / Dc, rows=rows)


# ----------------------------------------------------------------------------------------------------------------
def l3_mwpm(shots: int) -> None:
    print("\n=== L3 MWPM failure rate vs Peierls / SAW bounds (Thm 4.15, Prop 4.22) ===")
    from lcd.codes import RotatedSurfaceCode
    from lcd.decoders import MWPMCodeCapacity
    from lcd.noise import sample_iid_erasure
    c = saw_counts(10)
    mu, g = c[10] ** 0.1, max(c[r] * (c[10] ** 0.1) ** -r for r in range(10))
    rng = np.random.default_rng(11)
    rows, worst = [], np.inf
    for d in (3, 5, 7, 9):
        code = RotatedSurfaceCode(d)
        dec = MWPMCodeCapacity(code)
        for p, e in ((0.01, 0.0), (0.02, 0.0), (0.03, 0.0), (0.01, 0.05), (0.005, 0.1)):
            N = shots if e == 0 else shots // 5
            fails = 0
            for start in range(0, N, 20000):
                m = min(20000, N - start)
                ex, ez, er = sample_iid_erasure(code.n, p, e, m, rng)
                fails += int(dec.fail(ex, ez, er if e > 0 else None).sum())
            lo = 0.0 if fails == 0 else float(beta.ppf(0.001, fails, N - fails + 1))
            b = min(peierls(d, p, e), saw_bound(d, p, e, mu, g))
            rows.append(dict(d=d, p=p, e=e, shots=N, fails=fails, bound=b))
            worst = min(worst, b / max(lo, 1e-300))
            if b < lo:
                print(f"    violation d={d} p={p} e={e}: bound {b:.3e} < lower limit {lo:.3e}")
    for r in rows:
        if r["d"] in (3, 9):
            print(f"    d={r['d']} p={r['p']} e={r['e']}: MWPM {r['fails'] / r['shots']:.2e} ({r['shots']} shots), bound {r['bound']:.2e}")
    check("L3 simulated MWPM failure rates below the bounds (99.9% lower limits)", worst >= 1, f"({len(rows)} points)")
    OUT["L3"] = rows


# ----------------------------------------------------------------------------------------------------------------
REGION = [0, 1, 3, 4]          # the qubits of the bulk X plaquette with corner (0,0), d = 3 (qubit (r,c) -> 3r + c)
G0 = [6, 7]                    # bottom X boundary check, disjoint from the region
PB_MIN = 0.2


def lam(q):
    return 1 - 4 * q / 3


def true_eps_bursts(st, p, e, r, pb, amax=3, nmax=4) -> float:
    """Exact eps* (upper value: erased sets with |A| > amax counted as failures) for d = 3 with Poisson(r) bursts of
    strength pb on REGION."""
    n = st.n
    wN = [poisson.pmf(k, r) for k in range(nmax + 1)]
    tail_N = 1 - sum(wN)
    eps, mass = 0.0, 0.0
    for a in range(amax + 1):
        for A in itertools.combinations(range(n), a):
            pA = e ** a * (1 - e) ** (n - a)
            mass += pA
            W = np.zeros(st.ncls)
            for k, wk in enumerate(wN):
                q = np.full(n, float(p))
                qb = p
                for _ in range(k):
                    qb = oplus(qb, pb)
                q[REGION] = qb
                q[list(A)] = 0.75
                P = np.prod(np.where(st.nz == 1, q / 3, 1 - q), axis=1)
                W += wk * np.bincount(st.cls, weights=P, minlength=st.ncls)
            W = W[st.order]
            eps += pA * (1 - float(np.maximum.reduceat(W, st.starts).sum()))
    return eps + (1 - mass) + tail_N


def cp_upper(x: int, N: int, delta: float) -> float:
    return 1.0 if x >= N else float(beta.ppf(1 - delta, x + 1, N - x))


def l4_coverage(trials: int, M: int) -> None:
    print("\n=== L4 coverage of Theorem 4.14 at d = 3 (Poisson bursts on one plaquette) ===")
    st = code_structure(esc.rotated_surface_d3())
    Wt = json.load(open(REPO / "results" / "cc_certificate.json"))["logical_weight_distribution"]["3"]
    n, V, delta = 9, len(REGION), 0.05
    d1 = d2 = d3 = delta / 3
    eta = 0.5 * (1 - lam(PB_MIN) ** V)
    rng = np.random.default_rng(3)
    res = []
    for (p, e, r, pb) in ((0.005, 0.01, 0.02, 0.5), (0.01, 0.0, 0.05, 0.3), (0.003, 0.02, 0.01, 0.75)):
        eps_true = true_eps_bursts(st, p, e, r, pb)
        theta = 4 * pb / 3
        cover, cert_ok, certs = 0, 0, []
        for _ in range(trials):
            flags = rng.random((M, n)) < e
            z = rng.random((M, n)) < 2 * p / 3                      # Z component of the background Pauli
            Nb = rng.poisson(r, M)
            hit = rng.random((M, V)) < 1 - (1 - theta) ** Nb[:, None]   # each region qubit independently uniformized
            zr = rng.random((M, V)) < 0.5
            z[:, REGION] = np.where(hit, zr, z[:, REGION])
            z = np.where(flags, rng.random((M, n)) < 0.5, z)        # erased qubits uniform
            e_bar = cp_upper(int(flags.sum()), n * M, d1)
            clean = ~flags[:, G0].any(axis=1)
            bits = z[clean][:, G0].sum(axis=1) % 2
            P_bar = cp_upper(int(bits.sum()), int(clean.sum()), d2)
            p_bar = 0.75 * (1 - max(1 - 2 * P_bar, 0.0) ** (1 / len(G0)))
            T = z[:, REGION].sum(axis=1) % 2                        # the region's own X check
            PT = cp_upper(int(T.sum()), M, d3)
            r_bar = np.inf if PT >= eta else -log(1 - PT / eta)
            covered = p <= p_bar and e <= e_bar and r <= r_bar
            cover += covered
            if covered and np.isfinite(r_bar):
                rho_bar = 1 - exp(-r_bar)
                lB = min(b_burst(p_bar, e_bar, rho_bar, V, t) for t in np.linspace(0.005, 0.7, 140))
                cert = 0.5 * sum(cnt * exp(min(w * lB, 700.0)) for w, cnt in enumerate(Wt))
                certs.append(cert)
                cert_ok += cert >= eps_true
        ncov = sum(1 for _ in certs)
        res.append(dict(p=p, e=e, r=r, pb=pb, eps_true=eps_true, coverage=cover / trials, covered_finite=ncov,
                        cert_ok=cert_ok, cert_median=float(np.median(certs)) if certs else None))
        print(f"    p={p} e={e} r={r} p_b={pb}: eps*={eps_true:.3e}; coverage {cover / trials:.3f}; "
              f"certificate >= eps* on {cert_ok}/{ncov} covering runs (median certificate {res[-1]['cert_median']:.3g})")
    # coverage is >= 1 - delta in expectation; fail only if the observed count is implausible under 0.95 (p < 0.001)
    from scipy.stats import binom
    pv = [binom.cdf(round(x["coverage"] * trials), trials, 1 - delta) for x in res]
    check("L4 coverage not significantly below 1 - delta = 0.95", min(pv) > 1e-3,
          f"(coverages {', '.join(f'{x['coverage']:.3f}' for x in res)}; smallest binomial p-value {min(pv):.2g})")
    nonvac = sum(1 for x in res if x["cert_median"] is not None and x["cert_median"] < 1)
    print(f"    certificates below 1 (non-vacuous) at {nonvac} of {len(res)} points: the burst penalty kappa^V dominates at d = 3")
    check("L4 certificate >= exact eps* on every covering run", all(x["cert_ok"] == x["covered_finite"] for x in res))
    OUT["L4"] = dict(trials=trials, M=M, delta=delta, eta=eta, pb_min=PB_MIN, points=res)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = a.quick
    l1_e1_library(10 if q else 40)
    l2_stein(5000 if q else 40000)
    l3_mwpm(20000 if q else 100000)
    l4_coverage(50 if q else 300, 5000 if q else 10000)
    OUT["failures"] = failures
    (REPO / "results" / "second_pass_checks.json").write_text(json.dumps(OUT, indent=1, default=float))
    print("\nSUMMARY:", "ALL SECOND-PASS CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
