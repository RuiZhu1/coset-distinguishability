"""Checks of the results that close the SKETCH/PLAN items and open problems O1, O2, O3, O6 (theory/sec3-5).

  K1  geometry of the genie argument (Prop. 5.1, Step 1) on the rotated surface code of lcd, every interior column,
      d = 3..41: the Paulis supported on the column with zero syndrome are exactly {I, Xbar_column}
  K2  finite-d genie bound (Prop. 5.1): eps_genie(d) >= (1/2) B^d exp(-sqrt(d sigma^2)/2), exact eps_genie, d <= 61
  K3  tilted union bound for burst mixtures (Lemma 4.8, Thm 4.10):  for every logical G
          BC(P;G) <= E_P[sqrt(Q(X+G)/Q(X))] <= B_burst^wt(G)   and   eps* <= 1/2 sum_G BC(P;G) <= 1/2 W(B_burst),
      exact ML on [[5,1,3]] (with erasure) and on the d = 3 rotated surface code (no erasure)
  K4  O1 is false: exact rational computation of a burst model on the d = 3 rotated surface code in which eps*
      DEcreases when the burst strength p_b increases from 1/2 to 3/4 (Prop. 4.11)
  K5  [[5,1,3]]: certificate (exact integer polynomials) that flagging one qubit never changes the ML decision for
      0 < p < 3/4, hence R^(d) = c exactly at e0 = 0 (Prop. 3.32); tie-aware: the d = 3 surface code violates (E1),
      Steane satisfies (E1) but has exact ML ties (its R^(d) < c comes from (E2))
  K6  concavity of eps*_A along one qubit's rate and the right-derivative form of the rate bound (Lemma 3.28, Thm 3.30)
  K7  contamination only lowers the parity expectation of a check (Lemma 4.12(b)), exact

Run:  python theory/checks/completion_checks.py      (numpy only; about 30 s)
Exit status is non-zero if a check fails.
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from math import comb, exp, log, sqrt

import numpy as np

from exact_small_codes import Code, five_qubit, rotated_surface_d3, steane
from structures import Structure, all_paulis, rref

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def oplus(a, b):
    """Composition of depolarizing rates: lambda(a (+) b) = lambda(a) lambda(b)."""
    return a + b - 4 * a * b / 3


# --------------------------------------------------------------------------------------------------------------
# rotated surface code with the conventions of src/lcd/codes/rotated_surface.py (copied so that numpy suffices)
# --------------------------------------------------------------------------------------------------------------
def rotated(d: int):
    idx = lambda r, c: r * d + c
    hx, hz = [], []
    for r in range(-1, d):
        for c in range(-1, d):
            bulk = 0 <= r <= d - 2 and 0 <= c <= d - 2
            kind = "X" if (r + c) % 2 == 0 else "Z"
            qs = [idx(rr, cc) for rr, cc in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)) if 0 <= rr < d and 0 <= cc < d]
            row = np.zeros(d * d, np.uint8)
            row[qs] = 1
            if bulk:
                (hx if kind == "X" else hz).append(row)
            elif r in (-1, d - 1) and 0 <= c <= d - 2 and kind == "X":
                hx.append(row)
            elif c in (-1, d - 1) and 0 <= r <= d - 2 and kind == "Z":
                hz.append(row)
    return np.array(hx), np.array(hz)


def nullity(M: np.ndarray) -> tuple[int, np.ndarray]:
    """Dimension of the kernel of M over F_2, and a basis of it."""
    R, piv = rref(M)
    free = [c for c in range(M.shape[1]) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(M.shape[1], np.uint8)
        v[f] = 1
        for row, c in zip(R, piv):
            if row[f]:
                v[c] = 1
        basis.append(v)
    return len(free), np.array(basis)


def k1_genie_geometry() -> None:
    print("\n=== K1 genie geometry (Prop. 5.1, Step 1) ===")
    ok, ncol = True, 0
    for d in range(3, 42, 2):
        HX, HZ = rotated(d)
        for c0 in range(1, d - 1):
            T = [r * d + c0 for r in range(d)]
            kx, bx = nullity(HZ[:, T])       # X part on the column: must be {0, all-ones}
            kz, _ = nullity(HX[:, T])        # Z part: must be {0}
            ok &= kx == 1 and bx[0].all() and kz == 0
            ncol += 1
        # every top-row qubit lies in exactly one Z check, every left-column qubit in exactly one X check (Thm 4.15)
        ok &= (HZ[:, :d].sum(axis=0) == 1).all() and (HX[:, [r * d for r in range(d)]].sum(axis=0) == 1).all()
        ok &= HZ.sum(axis=1).max() <= 4 and HX.sum(axis=1).max() <= 4
    check("K1 ker(sigma) on an interior column = {I, Xbar}; boundary rows/columns in one check; check weight <= 4",
          ok, f"({ncol} columns, d = 3..41)")


# --------------------------------------------------------------------------------------------------------------
def beta_dep(p):
    return 2 * sqrt(p * (1 - p) / 3) + 2 * p / 3


def B(p, e):
    return e + (1 - e) * beta_dep(p)


def genie_exact(p, e, d):
    """eps_genie(d) = 1/2 E_P[min(1, exp(-Lambda))]; per qubit Lambda in {+L, -L, 0} with probs (1-e)(1-p), (1-e)p/3, rest."""
    a, b = (1 - e) * (1 - p), (1 - e) * p / 3
    c = 1 - a - b
    rho = (p / 3) / (1 - p)                       # exp(-L)
    tot = 0.0
    for kp in range(d + 1):
        for km in range(d + 1 - kp):
            k0 = d - kp - km
            w = comb(d, kp) * comb(d - kp, km) * a ** kp * b ** km * c ** k0
            s = kp - km                                # Lambda = s L
            tot += w * (rho ** s if s > 0 else 1.0)
    return 0.5 * tot


def genie_lower(p, e, d):
    Bv = B(p, e)
    L = log(3 * (1 - p) / p)
    sigma2 = 2 * (1 - e) * sqrt((1 - p) * p / 3) * L * L / Bv
    return 0.5 * Bv ** d * exp(-0.5 * sqrt(d * sigma2))


def k2_genie_bound() -> None:
    print("\n=== K2 finite-d genie bound (Prop. 5.1) ===")
    worst, worst_rate = np.inf, 0.0
    for p in (0.005, 0.02, 0.06, 0.15, 0.3):
        for e in (0.0, 0.05, 0.3):
            for d in (3, 5, 11, 21, 41, 61):
                ex, lo = genie_exact(p, e, d), genie_lower(p, e, d)
                worst = min(worst, ex / lo)
            # exponent of the exact genie error approaches ln(1/B)
            d1, d2 = 101, 201
            rate = -(log(genie_exact(p, e, d2)) - log(genie_exact(p, e, d1))) / (d2 - d1)
            worst_rate = max(worst_rate, abs(rate / -log(B(p, e)) - 1))
    check("K2 eps_genie(d) >= (1/2) B^d exp(-sqrt(d sigma^2)/2)", worst >= 1 - 1e-12, f"(min ratio exact/bound {worst:.3f})")
    check("K2 slope of -ln eps_genie between d = 101 and 201 is ln(1/B) to 5%", worst_rate < 0.05, f"(max rel. dev. {worst_rate:.3f})")


# --------------------------------------------------------------------------------------------------------------
def code_structure(Sgen):
    c = Code("x", Sgen)
    return Structure(c.n, np.array(Sgen, np.uint8), c.N)


def span(basis, n2):
    out = [np.zeros(n2, np.uint8)]
    for b in basis:
        out += [o ^ b for o in out]
    return out


def logicals(st: Structure, Sgen, Nbasis):
    """All G in N \\ S as (index into all_paulis, Pauli weight)."""
    n = st.n
    Sset = {tuple(v) for v in span(rref(np.array(Sgen, np.uint8))[0], 2 * n)}
    out = []
    for v in span(rref(Nbasis)[0], 2 * n):
        if tuple(v) in Sset:
            continue
        gi = sum(int(v[j]) << (2 * j) for j in range(n)) + sum(int(v[n + j]) << (2 * j + 1) for j in range(n))
        out.append((gi, int(np.count_nonzero(v[:n] | v[n:]))))
    return out


def mixture_law(st: Structure, p, e, regions, rho, pb, erased=()):
    """P(E | erased set) for background Dep_p, erased qubits uniform, regions independently active w.p. rho, each active
    region adds Dep_pb on its qubits (unflagged).  Returns the vector over all 4^n Paulis (structures.all_paulis order)."""
    n = st.n
    P = np.zeros(len(st.cls))
    for act in itertools.product([0, 1], repeat=len(regions)):
        w = float(np.prod([rho if a else 1 - rho for a in act]))
        q = np.full(n, float(p))
        for a, R in zip(act, regions):
            if a:
                for j in R:
                    q[j] = oplus(q[j], pb)
        q[list(erased)] = 0.75
        L = np.where(st.nz == 1, q / 3, 1 - q)
        P += w * np.prod(L, axis=1)
    return P


def eps_of(st: Structure, P):
    W = np.bincount(st.cls, weights=P, minlength=st.ncls)[st.order]
    return float(P.sum() - np.maximum.reduceat(W, st.starts).sum())


def b_burst(p, e, rho_bar, V, t):
    tau = (t / 3) / (1 - t)
    m = lambda s: e + (1 - e) * ((1 - s) * sqrt(tau) + (s / 3) / sqrt(tau) + 2 * s / 3)
    m0, m1 = m(p), m(0.75)
    kappa = m1 / m0
    return log(m0) + rho_bar * (kappa ** V - 1) / V          # ln B_burst (B_burst itself may overflow)


def k3_burst_union() -> None:
    print("\n=== K3 tilted union bound for burst mixtures (Lemma 4.8, Theorem 4.10) ===")
    cases = [("[[5,1,3]]", five_qubit(), [(0, 1), (2, 3, 4)], (0.0, 0.05, 0.2)),
             ("rotated d=3", rotated_surface_d3(), [(0, 1, 3, 4), (4, 5, 7, 8), (1, 2, 4, 5), (3, 4, 6, 7)], (0.0,))]
    worst_G, worst_eps, worst_tilt, ncase = 0.0, 0.0, 0.0, 0
    for name, Sg, regions, es in cases:
        st = code_structure(Sg)
        c = Code("x", Sg)
        Gs = logicals(st, Sg, c.N)
        n = st.n
        idx = np.arange(len(st.cls))
        V = max(len(R) for R in regions)
        cover = np.zeros(n)
        for R in regions:
            cover[list(R)] += 1
        for p in (0.01, 0.05):
            for e in es:
                for rho in (0.02, 0.1):
                    for pb in (0.2, 0.75):
                        rho_bar = rho * cover.max()
                        # exact BC(P;G) with flags: sum over erased sets A of Pr[A] * BC of the conditional law
                        pats = [A for a in range(n + 1) for A in itertools.combinations(range(n), a)] if e > 0 else [()]
                        PA = [(e ** len(A) * (1 - e) ** (n - len(A)), mixture_law(st, p, e, regions, rho, pb, A)) for A in pats]
                        eps = sum(w * eps_of(st, P) for w, P in PA)
                        for t in (p, 0.2):
                            lB = b_burst(p, e, rho_bar, V, t)
                            tau = (t / 3) / (1 - t)
                            tot_bc = 0.0
                            for gi, wt in Gs:
                                bc = sum(w * np.sqrt(P * P[idx ^ gi]).sum() for w, P in PA)
                                # tilted expectation E_P[prod_j g_j], g_j = sqrt(Q(a+G_j)/Q(a)), Q = N_cc(t, e)
                                g = np.ones(len(idx))
                                Gx = [(gi >> (2 * j)) & 1 for j in range(n)]
                                Gz = [(gi >> (2 * j + 1)) & 1 for j in range(n)]
                                for j in range(n):
                                    if Gx[j] or Gz[j]:
                                        aj = st.V[:, j] | (st.V[:, n + j] << 1)
                                        gj_code = Gx[j] | (Gz[j] << 1)
                                        g *= np.where(aj == 0, sqrt(tau), np.where(aj == gj_code, 1 / sqrt(tau), 1.0))
                                tilt = 0.0
                                for (w, P), A in zip(PA, pats):
                                    gA = g.copy()
                                    for j in A:          # flagged qubits: Q uniform -> factor 1
                                        if Gx[j] or Gz[j]:
                                            aj = st.V[:, j] | (st.V[:, n + j] << 1)
                                            gj_code = Gx[j] | (Gz[j] << 1)
                                            gA /= np.where(aj == 0, sqrt(tau), np.where(aj == gj_code, 1 / sqrt(tau), 1.0))
                                    tilt += w * (P * gA).sum()
                                worst_tilt = max(worst_tilt, bc / tilt - 1)
                                worst_G = max(worst_G, log(tilt) - wt * lB)
                                tot_bc += bc
                            W_bound = sum(exp(min(w * lB, 700.0)) for _, w in Gs)
                            worst_eps = max(worst_eps, eps / (0.5 * tot_bc) - 1, tot_bc / W_bound - 1)
                            ncase += 1
    check("K3 BC(P;G) <= tilted expectation (every logical G)", worst_tilt <= 1e-12, f"(max excess {worst_tilt:.1e})")
    check("K3 tilted expectation <= B_burst^wt(G) (every logical G)", worst_G <= 1e-12, f"(max excess {worst_G:.1e})")
    check("K3 eps* <= 1/2 sum_G BC <= 1/2 W(B_burst)", worst_eps <= 1e-12, f"({ncase} cases; max excess {worst_eps:.1e})")


# --------------------------------------------------------------------------------------------------------------
def k4_o1_counterexample() -> None:
    print("\n=== K4 O1: eps* is not monotone in the burst strength (Prop. 4.11) ===")
    st = code_structure(rotated_surface_d3())
    R = [1, 3, 5, 7]
    out = [j for j in range(9) if j not in R]
    kin = st.nz[:, R].sum(axis=1)
    kout = st.nz[:, out].sum(axis=1)
    p, rho = Fraction(1, 200), Fraction(1, 20)

    def eps_exact(pb):
        q = oplus(p, pb)
        f0 = lambda k, nn: (p / 3) ** k * (1 - p) ** (nn - k)
        f1 = lambda k, nn: (q / 3) ** k * (1 - q) ** (nn - k)
        val = {(a, b): (1 - rho) * f0(a, 4) * f0(b, 5) + rho * f1(a, 4) * f0(b, 5) for a in range(5) for b in range(6)}
        tab = {}
        for c, a, b in zip(st.cls.tolist(), kin.tolist(), kout.tolist()):
            tab.setdefault(c, {}).setdefault((a, b), 0)
            tab[c][(a, b)] += 1
        W = {c: sum(cnt * val[k] for k, cnt in t.items()) for c, t in tab.items()}
        syn_of = dict(zip(st.cls.tolist(), st.syn.tolist()))
        best: dict[int, Fraction] = {}
        for c, w in W.items():
            s = syn_of[c]
            best[s] = max(best.get(s, Fraction(0)), w)
        return 1 - sum(best.values())

    e_half, e_34 = eps_exact(Fraction(1, 2)), eps_exact(Fraction(3, 4))
    check("K4 eps*(p_b = 3/4) < eps*(p_b = 1/2), exact rationals", e_34 < e_half,
          f"({float(e_half):.6e} -> {float(e_34):.6e}, relative drop {float(1 - e_34 / e_half):.3f})")


# --------------------------------------------------------------------------------------------------------------
def class_polys(st: Structure, flagged: tuple[int, ...]):
    """Integer coefficient arrays W_c(r) = sum_{E in c} r^{wt of E off the flagged qubits}."""
    keep = [j for j in range(st.n) if j not in flagged]
    w = st.nz[:, keep].sum(axis=1)
    M = np.zeros((st.ncls, st.n + 1), dtype=object)
    M[:] = 0
    np.add.at(M, (st.cls, w), 1)
    return M


def positive_on_unit_interval(coef, strict: bool) -> bool:
    """Certificate that P(r) = sum coef[k] r^k is > 0 (strict) or >= 0 on (0, 1): all coefficients of
    (1+x)^m P(1/(1+x)) are >= 0 (and not all zero for strict).  Exact integer arithmetic."""
    m = len(coef) - 1
    Q = [0] * (m + 1)
    for k, a in enumerate(coef):
        if a == 0:
            continue
        for i in range(m - k + 1):                  # a (1+x)^(m-k)
            Q[i] += a * comb(m - k, i)
    if any(c < 0 for c in Q):
        return False
    return any(c > 0 for c in Q) if strict else True


def single_flag_certificate(Sg) -> tuple[bool, str]:
    st = code_structure(Sg)
    syn_of = np.full(st.ncls, -1)
    syn_of[st.cls] = st.syn
    W0 = class_polys(st, ())
    r0 = 1e-3
    ok, why = True, ""
    for s in np.unique(st.syn):
        cl = np.flatnonzero(syn_of == s)
        vals = [float(sum(int(a) * r0 ** k for k, a in enumerate(W0[c]))) for c in cl]
        cs = int(cl[int(np.argmax(vals))])
        for c in cl:
            if c != cs and not positive_on_unit_interval(list(W0[cs] - W0[c]), strict=True):
                return False, f"unflagged decision not uniquely constant on (0,3/4) (syndrome {s})"
        for j in range(st.n):
            Wj = class_polys(st, (j,))
            for c in cl:
                if c != cs and not positive_on_unit_interval(list(Wj[cs] - Wj[c]), strict=False):
                    ok, why = False, f"flagging qubit {j} changes the decision at syndrome {s}"
                    return ok, why
    return ok, why


def k5_perfect_code() -> None:
    print("\n=== K5 single-flag invariance of the ML decision (Prop. 3.32) ===")
    ok, why = single_flag_certificate(five_qubit())
    check("K5 [[5,1,3]]: unique ML decision, unchanged by any single flag, for all 0 < p < 3/4 (exact certificate)", ok, why)
    # tie-aware (E1): some class that is ML-optimal without the flag stays ML-optimal with it.  The d = 3 surface code
    # violates it; Steane satisfies it but has exact ML ties, and R^(d) < c there comes from (E2) (Thm 3.31).
    # (An earlier version took the first argmax and so reported a spurious violation for Steane.)
    for name, Sg, expect_violation in (("Steane", steane(), False), ("rotated d=3", rotated_surface_d3(), True)):
        st = code_structure(Sg)
        syn_of = np.full(st.ncls, -1)
        syn_of[st.cls] = st.syn
        bad, ties = 0, 0
        for p in (0.01, 0.05):
            P0 = mixture_law(st, p, 0, [], 0, 0)
            W0 = np.bincount(st.cls, weights=P0, minlength=st.ncls)
            for s in np.unique(st.syn):
                cl = np.flatnonzero(syn_of == s)
                ties += int((W0[cl] >= W0[cl].max() * (1 - 1e-9)).sum() > 1)
            for j in range(st.n):
                Pj = mixture_law(st, p, 0, [], 0, 0, erased=(j,))
                Wj = np.bincount(st.cls, weights=Pj, minlength=st.ncls)
                for s in np.unique(st.syn):
                    cl = np.flatnonzero(syn_of == s)
                    O0 = set(cl[W0[cl] >= W0[cl].max() * (1 - 1e-9)])
                    Oj = set(cl[Wj[cl] >= Wj[cl].max() * (1 - 1e-9)])
                    bad += not (O0 & Oj)
        if expect_violation:
            check(f"K5 {name}: (E1) violated (tie-aware; p = 0.01, 0.05)", bad > 0, f"({bad} syndrome-flag pairs)")
        else:
            check(f"K5 {name}: (E1) holds (tie-aware) but the ML class is not unique (p = 0.01, 0.05)", bad == 0 and ties > 0,
                  f"({ties} tied syndromes)")


# --------------------------------------------------------------------------------------------------------------
def k6_concavity_and_right_derivative() -> None:
    print("\n=== K6 concavity along one qubit; right-derivative rate bound (Lemma 3.28, Theorem 3.30) ===")
    worst_conc, worst_rate = 0.0, -np.inf
    for Sg in (five_qubit(), steane(), rotated_surface_d3()):
        st = code_structure(Sg)
        n = st.n
        qs = np.linspace(0, 0.75, 31)
        for p in (0.02, 0.1):
            for A in ((), (0,), (1, 2)):
                for j in [k for k in range(n) if k not in A][:3]:
                    vals = []
                    for qj in qs:
                        q = np.full(n, p)
                        q[list(A)] = 0.75
                        q[j] = qj
                        L = np.where(st.nz == 1, q / 3, 1 - q)
                        vals.append(eps_of(st, np.prod(L, axis=1)))
                    worst_conc = max(worst_conc, np.diff(vals, 2).max())
        for p in (0.02, 0.06, 0.2):
            for e in ((0.0, 0.05, 0.3) if n <= 7 else (0.0,)):
                h = 1e-7
                dp = (st.eps(p + h, e) - st.eps(p, e)) / h
                if e == 0:
                    de = sum(st.eps_A(p, (j,)) for j in range(n)) - n * st.eps_A(p, ())
                else:
                    de = (st.eps(p, e + h) - st.eps(p, e - h)) / (2 * h)
                worst_rate = max(worst_rate, de - (0.75 - p) / (1 - e) * dp)
    check("K6 eps*_A concave in one qubit's rate (second differences <= 0)", worst_conc <= 1e-12, f"(max {worst_conc:.1e})")
    check("K6 d eps*/de <= c * right d eps*/dp", worst_rate <= 1e-6, f"(max excess {worst_rate:.1e})")


# --------------------------------------------------------------------------------------------------------------
def k7_contamination() -> None:
    print("\n=== K7 contamination cannot raise the parity expectation of a check (Lemma 4.12(b)) ===")
    # one weight-4 check on qubits 0..3; background Dep_p; bursts on overlapping regions; exact over all 4^4 Paulis
    n = 4
    V = all_paulis(n)
    nz = (V[:, :n] | V[:, n:]).astype(int)
    parity = V[:, n:].astype(int).sum(axis=1) % 2         # an X-type check flips on the Z components (int: 1 - 2*parity must not wrap)
    worst = -np.inf
    for p in (0.0, 0.01, 0.1):
        lam = 1 - 4 * p / 3
        for regions in ([(0, 1)], [(0, 1, 2), (2, 3)], [(1,), (0, 1, 2, 3)]):
            for rho in (0.05, 0.5):
                for pb in (0.1, 0.5, 0.75):
                    P = np.zeros(len(V))
                    for act in itertools.product([0, 1], repeat=len(regions)):
                        w = float(np.prod([rho if a else 1 - rho for a in act]))
                        q = np.full(n, p)
                        for a, R in zip(act, regions):
                            if a:
                                for j in R:
                                    q[j] = oplus(q[j], pb)
                        P += w * np.prod(np.where(nz == 1, q / 3, 1 - q), axis=1)
                    corr = float((P * (1 - 2 * parity)).sum())
                    worst = max(worst, corr - lam ** n)
    check("K7 E[(-1)^bit] <= lambda^w0 under burst contamination", worst <= 1e-12, f"(max excess {worst:.1e})")


def main() -> int:
    k1_genie_geometry()
    k2_genie_bound()
    k3_burst_union()
    k4_o1_counterexample()
    k5_perfect_code()
    k6_concavity_and_right_derivative()
    k7_contamination()
    print("\nSUMMARY:", "ALL COMPLETION CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
