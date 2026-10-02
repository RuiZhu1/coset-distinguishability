"""Falsification test of Theorem 4.5 (finite-sample certification at code capacity), theory/sec6 correspondence table.

Theorem 4.5: from M samples (flags and syndromes) of N_cc(p, e) form
    e_bar = one-sided Clopper-Pearson upper limit of the flag rate (level 1 - delta1)                  (Lemma 4.3)
    p_bar = 3/4 (1 - lambda_low),  lambda_low = (1 - 2 P_bar)_+^(1/w0), P_bar the Clopper-Pearson upper
            limit of the flip rate of one X-type generator g0 (weight w0) on samples with no flag on its support (Lemma 4.4)
Then with probability >= 1 - delta (delta = delta1 + delta2), for all d at once,
    eps*_d(p, e) <= eps*_d(p_bar, e_bar) <= (1/2) W~_d(B(p_bar, e_bar)),   W~_d = W_{N(S)} - W_S.

Checks:
  (i)   coverage: over many simulated calibration runs, the fraction with {e <= e_bar and p <= p_bar} is >= 1 - delta;
  (ii)  the certificate (1/2) W~_d(B(p_bar, e_bar)) is >= the true eps*_d(p, e) (exact at d = 3, stratified exact-TN ML
        at d = 5); it must hold on every run in the covering event;
  (iii) the union bound itself at the true parameters, eps*_d <= (1/2) W~_d(B(p, e)) (check C5, extended from d = 3 to d = 5).
W~_d is computed exactly (integer weight distribution of S by enumeration, MacWilliams for N(S)) for d = 3, 5; d >= 7 would need
a tensor-network weight enumerator and is not attempted.  The certificate is an upper bound for ML decoding only.

    python experiments/certification/cc_certificate.py [--trials 2000]

Writes results/cc_certificate.json.  Exit status is non-zero if a check fails.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import beta

import lcd
from lcd.analysis import Strata, StratifiedEstimator
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder
from lcd.noise import sample_iid_erasure, sample_stratum

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "theory" / "checks"))
import exact_small_codes as esc  # noqa: E402

POINTS = [(0.005, 0.0), (0.01, 0.0), (0.01, 0.01), (0.02, 0.02), (0.04, 0.02), (0.06, 0.05)]
MS = [100, 1000, 10000]
DELTA1 = DELTA2 = 0.025


# ------------------------------------------------------------------ computational layer
def stabilizer_weight_distribution(code: RotatedSurfaceCode) -> list[int]:
    """A_w = number of stabilizer-group elements of Pauli weight w (exact enumeration of the 2^(n-1) elements, CSS)."""
    def span(H):
        out = np.zeros(1, np.int64)
        for row in H:
            v = int(sum(1 << int(j) for j in np.flatnonzero(row)))
            out = np.concatenate([out, out ^ v])
        return out
    sx, sz = span(code.HX), span(code.HZ)
    A = np.zeros(code.n + 1, np.int64)
    for chunk in np.array_split(sx, max(1, sx.size // 256)):
        w = np.bitwise_count(chunk[:, None] | sz[None, :]).ravel()
        A += np.bincount(w, minlength=code.n + 1)
    return [int(a) for a in A]


def logical_weight_distribution(code: RotatedSurfaceCode) -> list[int]:
    """Coefficients of W~ = W_{N(S)} - W_S (exact integers; MacWilliams: W_N(x) = |S|^-1 sum_g (1+3x)^(n-wt g) (1-x)^(wt g))."""
    n = code.n
    A = stabilizer_weight_distribution(code)
    size = sum(A)
    assert size == 2 ** (n - 1)
    WN = [0] * (n + 1)
    for w, a in enumerate(A):
        if a == 0:
            continue
        # coefficients of (1+3x)^(n-w) (1-x)^w
        for i in range(n - w + 1):
            ci = comb(n - w, i) * 3 ** i
            for j in range(w + 1):
                WN[i + j] += a * ci * comb(w, j) * (-1) ** j
    assert all(c % size == 0 for c in WN)
    WN = [c // size for c in WN]
    assert sum(WN) == 2 ** (n + 1)
    Wt = [WN[j] - A[j] for j in range(n + 1)]
    assert all(c >= 0 for c in Wt) and min(j for j, c in enumerate(Wt) if c) == code.d
    return Wt


def certificate(Wt: list[int], p: float, e: float) -> float:
    x = esc.B(p, e)
    return 0.5 * float(sum(c * x ** j for j, c in enumerate(Wt)))


# ------------------------------------------------------------------ statistical layer
def cp_upper(x: int, N: int, delta: float) -> float:
    return 1.0 if x >= N else float(beta.ppf(1 - delta, x + 1, N - x))


def calibrate(code: RotatedSurfaceCode, g0: np.ndarray, p: float, e: float, M: int, rng) -> tuple[float, float]:
    """One calibration run of M samples; returns (p_bar, e_bar)."""
    ex, ez, er = sample_iid_erasure(code.n, p, e, M, rng)
    e_bar = cp_upper(int(er.sum()), code.n * M, DELTA1)
    clean = ~er[:, g0].any(axis=1)                       # no flag on the support of g0
    bits = ez[clean][:, g0].sum(axis=1) % 2              # X-type generator flips on Z/Y components
    Mp = int(clean.sum())
    if Mp == 0:
        return 0.75, e_bar
    P_bar = cp_upper(int(bits.sum()), Mp, DELTA2)
    lam_low = max(1 - 2 * P_bar, 0.0) ** (1 / len(g0))
    return 0.75 * (1 - lam_low), e_bar


# ------------------------------------------------------------------ true eps*
def true_eps(d: int, p: float, e: float, rng) -> dict:
    if d == 3:
        code = esc.Code("rotated d=3", esc.rotated_surface_d3())
        return dict(eps=esc.Eps(code)(p, e), se=0.0, method="exact enumeration (theory/checks)")
    code = RotatedSurfaceCode(d)
    dec = ExactTNMLDecoder(code)

    def sampler(k, w, N, r):
        ex, ez, er = sample_stratum(code.n, k, w, N, r)
        return dec.fail(ex, ez, p, er, r)

    S = Strata(code.n, d, [(p, e)], cutoff=1e-12)
    est = StratifiedEstimator(S, sampler)
    est.run(200_000, rng, n0=500, rounds=3)
    res = est.estimate(p, e)
    return dict(eps=res.p_L, se=res.se, cert_interval=[res.lo, res.hi], decodes=res.decodes,
                method="stratified exact-TN ML matched at p")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    failures = []
    delta = DELTA1 + DELTA2

    # computational layer: exact logical weight distributions; cross-check d = 3 against theory/checks
    Wt = {d: logical_weight_distribution(RotatedSurfaceCode(d)) for d in (3, 5)}
    esc3 = esc.Code("rotated d=3", esc.rotated_surface_d3())
    for x in (0.05, 0.2, 0.5):
        ref = esc.weight_enum_N_minus_S(esc3, x)
        val = sum(c * x ** j for j, c in enumerate(Wt[3]))
        assert abs(val - ref) <= 1e-12 * max(1.0, ref), (x, val, ref)
    print("W~_3 coefficients:", Wt[3])
    print("W~_5 coefficients:", Wt[5])

    calib_code = RotatedSurfaceCode(5)
    g0 = np.flatnonzero(next(r for r in calib_code.HX if r.sum() == 4))
    out = dict(points=[], logical_weight_distribution={str(d): w for d, w in Wt.items()}, delta1=DELTA1, delta2=DELTA2,
               calibration=dict(code="rotated d=5", g0_support=[int(j) for j in g0]))
    for (p, e) in POINTS:
        eps = {d: true_eps(d, p, e, rng) for d in (3, 5)}
        pt = dict(p=p, e=e, B=float(esc.B(p, e)), eps=eps, union_bound={}, runs=[])
        # (iii) union bound at the true parameters
        for d in (3, 5):
            ub = certificate(Wt[d], p, e)
            slack = ub - (eps[d]["eps"] + 3 * eps[d]["se"])
            ok = ub >= eps[d]["eps"] - 3 * eps[d]["se"]
            pt["union_bound"][str(d)] = dict(bound=ub, ok=bool(ok), ratio=ub / eps[d]["eps"] if eps[d]["eps"] > 0 else float("inf"))
            if not ok:
                failures.append(f"union bound d={d} at (p, e) = ({p}, {e}): {ub:.3e} < {eps[d]['eps']:.3e}")
        print(f"(p, e) = ({p}, {e}): eps_3 = {eps[3]['eps']:.3e}, eps_5 = {eps[5]['eps']:.3e} +- {eps[5]['se']:.1e}; "
              f"union bound at truth: {pt['union_bound']['3']['bound']:.3e}, {pt['union_bound']['5']['bound']:.3e}")
        for M in MS:
            cov_e = cov_p = cov = 0
            certs = {3: [], 5: []}
            cert_ok = {3: 0, 5: 0}
            pbars, ebars = [], []
            for _ in range(a.trials):
                pb, eb = calibrate(calib_code, g0, p, e, M, rng)
                ce, cp = e <= eb, p <= pb
                cov_e += ce
                cov_p += cp
                cov += ce and cp
                pbars.append(pb)
                ebars.append(eb)
                for d in (3, 5):
                    c = certificate(Wt[d], pb, eb)
                    certs[d].append(c)
                    held = c >= eps[d]["eps"]
                    cert_ok[d] += held
                    if ce and cp and c < eps[d]["eps"] - 3 * eps[d]["se"]:
                        failures.append(f"(ii) certificate below eps_{d} on a covering run at ({p}, {e}), M = {M}")
            T = a.trials
            cov_rate = cov / T
            tol = 3 * np.sqrt(delta * (1 - delta) / T)
            ok = cov_rate >= 1 - delta - tol
            if not ok:
                failures.append(f"(i) coverage {cov_rate:.4f} < {1 - delta} - {tol:.4f} at ({p}, {e}), M = {M}")
            run = dict(M=M, trials=T, coverage=cov_rate, coverage_e=cov_e / T, coverage_p=cov_p / T, coverage_ok=bool(ok),
                       p_bar_median=float(np.median(pbars)), e_bar_median=float(np.median(ebars)),
                       cert_median={str(d): float(np.median(certs[d])) for d in (3, 5)},
                       cert_ge_eps_fraction={str(d): cert_ok[d] / T for d in (3, 5)},
                       cert_nonvacuous_fraction={str(d): float(np.mean(np.array(certs[d]) < 1)) for d in (3, 5)})
            pt["runs"].append(run)
            print(f"    M = {M:6d}: coverage {cov_rate:.4f} (e {cov_e / T:.4f}, p {cov_p / T:.4f})  "
                  f"median p_bar = {run['p_bar_median']:.4f}, e_bar = {run['e_bar_median']:.4f};  "
                  f"median cert d=3: {run['cert_median']['3']:.2e}, d=5: {run['cert_median']['5']:.2e}")
        out["points"].append(pt)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    out.update(script="experiments/certification/cc_certificate.py", args=vars(a), failures=failures, git_commit=commit,
               git_dirty_src=dirty, versions=dict(lcd=lcd.__version__, numpy=np.__version__),
               date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (REPO / "results" / "cc_certificate.json").write_text(json.dumps(out, indent=2, default=float))
    print("FAILURES:" if failures else "ALL CERTIFICATE CHECKS PASSED", *failures, sep="\n  ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
