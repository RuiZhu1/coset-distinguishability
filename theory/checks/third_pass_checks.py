"""Checks of the third pen-and-paper pass (theory sec 0, 3, 4.7-4.8, 5.1, 5.5-5.6).  Light: a few minutes, < 1 GB.

  M1  Theorem 5.16 (pure-erasure exponent): (1/8)(2e)^d (1-e)^(d-1) <= eps*_d(0,e) <= row-count bound, d = 3 exact,
      d = 5 exact for |A| <= 6 and sampled above; the path facts behind the lower bound (|Gamma| >= 2^(d-2), paths inside
      the lattice, zero syndrome, odd overlap with the Z logical) for d <= 15
  M2  Proposition 5.2(b) (degeneracy): the plaquette family is pairwise disjoint and disjoint from column 0, of size
      (d-1)(d-3)/4, for d <= 21; W~_d(x) >= x^d (1+x^4)^((d-1)(d-3)/4) against the exact weight distributions (d = 3, 5)
  M3  Proposition 0.13 and Theorems 0.14, 0.15 (R-c): monotonicity and multiplicativity of Phi_alpha and g on random
      objects and random reductions; Lorenz deficit at typical flag counts -> 0 when H_Q > H_P (trade pair), not when H_Q < H_P
  M5  Theorem 4.24 (weak bursts): thinning reproduces the law of the uniformized set exactly; the bound on the correlated
      coverage; the certified efficiency eta(theta_0)
  M6  Theorem 4.23 (Peierls with bursts): P[path bad] <= B_{M,b}^|path| by exact enumeration on short paths
  M7  Observations 4.25, 4.26 (O1): the region and duration counterexamples in exact rational arithmetic
  M8  Observation 3.38 (codes from qubit AME states): (E1) tie-aware and exact R/c, all choices of the input legs

Run:  python theory/checks/third_pass_checks.py        Writes results/third_pass_checks.json.  Exit status non-zero on failure.
"""
from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from math import comb, exp, log, log2, sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_small_codes as esc  # noqa: E402
from completion_checks import code_structure  # noqa: E402
from sharp_codes_scan import exact_rate, symp_complement  # noqa: E402
from structures import Structure, rref  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.codes import RotatedSurfaceCode  # noqa: E402

failures: list[str] = []
OUT: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def rank2(M: np.ndarray) -> int:
    return 0 if M.size == 0 else len(rref(M % 2)[1])


# ----------------------------------------------------------------------------------------------------------------
def r_of(code, A) -> int:
    """number of independent logical operators supported on A (CSS): Lemma 3.20"""
    n = code.n
    Ac = np.setdiff1d(np.arange(n), A)
    HX, HZ = code.HX.astype(np.uint8), code.HZ.astype(np.uint8)
    rX = len(A) - rank2(HZ[:, A]) - rank2(HX) + rank2(HX[:, Ac])
    rZ = len(A) - rank2(HX[:, A]) - rank2(HZ) + rank2(HZ[:, Ac])
    return rX + rZ


def eps_erasure(d, e, exact_up_to, samples, rng):
    code = RotatedSurfaceCode(d)
    n = code.n
    f = {}
    for w in range(d, n + 1):
        if w <= exact_up_to:
            subs = [np.array(A) for A in itertools.combinations(range(n), w)]
        else:
            subs = [np.sort(rng.choice(n, w, replace=False)) for _ in range(samples)]
        f[w] = float(np.mean([1 - 2.0 ** (-r_of(code, A)) for A in subs]))
    return lambda e_: sum(comb(n, w) * e_ ** w * (1 - e_) ** (n - w) * f[w] for w in f)


def row_bound(d, x):
    return d * x * (2 * x * (1 + 4 * x + 16 * x * x)) ** (d - 1) / (0.5 - 2 * x - 8 * x * x)


def min_weight_paths(d):
    """minimum-weight X logicals from the centre of row 0 with at most (d-1)/2 diagonal moves (the set Gamma)"""
    c0 = (d - 1) // 2
    out = []
    for moves in itertools.product([0, 1], repeat=d - 1):
        if sum(moves) > (d - 1) // 2:
            continue
        cols = [c0]
        ok = True
        for r, mv in enumerate(moves):
            c = cols[-1]
            diag = 1 if (r + c) % 2 == 1 else -1          # Z plaquette with corner (r,c) if r+c odd, else (r,c-1)
            c2 = c + diag if mv else c
            if not 0 <= c2 < d:
                ok = False
                break
            cols.append(c2)
        if ok:
            out.append(cols)
    return out


def m1_erasure() -> None:
    print("\n=== M1 pure-erasure exponent (Theorem 5.16) ===")
    rng = np.random.default_rng(20261002)
    rows, ok = [], True
    for d, exact_up_to, samples in ((3, 9, 0), (5, 6, 2000)):
        eps = eps_erasure(d, 0.0, exact_up_to, samples, rng)
        for e in (0.03, 0.05, 0.1, 0.15):
            v = eps(e)
            lo = (2 * e) ** d * (1 - e) ** (d - 1) / 8
            up = 2 * row_bound(d, e)
            rows.append(dict(d=d, e=e, eps=v, lower=lo, upper=up))
            ok &= lo <= v <= up
            print(f"    d={d} e={e}: {lo:.2e} <= {v:.3e} <= {up:.2e}")
    check("M1 lower <= eps*(0,e) <= upper (d = 3 exact, d = 5 exact up to |A| = 6)", ok)
    ok2 = True
    for d in range(3, 16, 2):
        code = RotatedSurfaceCode(d)
        paths = min_weight_paths(d)
        ok2 &= len(paths) >= 2 ** (d - 2)
        for cols in paths[:: max(1, len(paths) // 50)]:
            x = np.zeros(code.n, np.uint8)
            for r, c in enumerate(cols):
                x[r * d + c] = 1
            ok2 &= not ((code.HZ @ x) % 2).any() and int(x @ code.LZ) % 2 == 1
    check("M1 |Gamma| >= 2^(d-2); sampled paths are X logicals (zero syndrome, odd overlap with Zbar), d <= 15", ok2)
    OUT["M1"] = rows


# ----------------------------------------------------------------------------------------------------------------
def m2_degeneracy() -> None:
    print("\n=== M2 degeneracy of the union bound (Proposition 5.2(b)) ===")
    ok = True
    for d in range(5, 22, 2):
        fam = [(2 * i, 2 * j) for i in range((d - 1) // 2) for j in range(1, (d - 1) // 2)]
        sup = [{(r + a) * d + (c + b) for a in (0, 1) for b in (0, 1)} for r, c in fam]
        ok &= len(fam) == (d - 1) * (d - 3) // 4
        ok &= all((r + c) % 2 == 0 and r <= d - 2 and c <= d - 2 for r, c in fam)
        ok &= all(not any(q % d == 0 for q in s) for s in sup)
        ok &= all(not (sup[a] & sup[b]) for a in range(len(sup)) for b in range(a))
    check("M2 plaquette family: X type, pairwise disjoint, off column 0, size (d-1)(d-3)/4 (d <= 21)", ok)
    W = json.load(open(REPO / "results" / "cc_certificate.json"))["logical_weight_distribution"]
    ok2 = True
    for dd, dist in W.items():
        d = int(dd)
        for x in (0.05, 0.2, 0.5, 1.0):
            exact = sum(c * x ** w for w, c in enumerate(dist))
            ok2 &= exact >= x ** d * (1 + x ** 4) ** ((d - 1) * (d - 3) // 4) * (1 - 1e-12)
    check("M2 W~_d(x) >= x^d (1+x^4)^((d-1)(d-3)/4) against exact distributions (d = 3, 5)", ok2)
    cross = {x: next(d for d in range(3, 10 ** 7, 2) if (d - 1) * (d - 3) / 4 * log(1 + x ** 4) + d * log(x) > log(2))
             for x in (0.25, 0.085)}
    print(f"    the lower bound alone exceeds 2 (so 1/2 W~ > 1) from d = {cross[0.25]} at x = 0.25 and d = {cross[0.085]} at x = 0.085")
    OUT["M2"] = {str(k): v for k, v in cross.items()}


# ----------------------------------------------------------------------------------------------------------------
def norm_alpha(v, a):
    if a == np.inf:
        return v.max()
    if a == -np.inf:
        return v.min()
    if a < 0 and (v == 0).any():
        return 0.0
    return float((v ** a).sum() ** (1 / a))


def Phi(P, a):
    return sum(norm_alpha(P[y], a) for y in range(P.shape[0]) if P[y].sum() > 0)


def g_peak(P):
    return max(P[y].max() / P[y].sum() for y in range(P.shape[0]) if P[y].sum() > 0)


def random_reduction(P, rng, ny2=3):
    """Q_{y'} = sum_{y,pi} M(y',pi|y) pi P_y with a random kernel over a few random permutations"""
    ny, L = P.shape
    perms = [rng.permutation(L) for _ in range(4)]
    M = rng.random((ny, ny2, len(perms)))
    M /= M.sum(axis=(1, 2), keepdims=True)
    Q = np.zeros((ny2, L))
    for y in range(ny):
        for y2 in range(ny2):
            for k, pi in enumerate(perms):
                Q[y2, pi] += M[y, y2, k] * P[y]
    return Q


def lorenz_deficit(p, m, pp, mp, n):
    from math import log as ln
    import bisect

    def groups(p_, m_):
        if p_ == 0:
            out = [(-(n - m_) * ln(4), 4 ** (n - m_))]
        else:
            out = sorted((((m_ - j) * ln(1 - p_) + j * ln(p_ / 3) - (n - m_) * ln(4), comb(m_, j) * 3 ** j * 4 ** (n - m_))
                          for j in range(m_ + 1)), key=lambda t: -t[0])
        tot = sum(c for _, c in out)
        if tot < 4 ** n:
            out.append((float("-inf"), 4 ** n - tot))
        return out

    def lorenz(gr):
        K, S, LV = [0], [0.0], []
        for lv, c in gr:
            K.append(K[-1] + c)
            S.append(S[-1] + (exp(lv + ln(c)) if lv > float("-inf") else 0.0))
            LV.append(lv)
        return K, S, LV

    def S_at(Lr, k):
        K, S, LV = Lr
        i = min(bisect.bisect_right(K, k) - 1, len(K) - 2)
        extra = 0.0 if (LV[i] == float("-inf") or k == K[i]) else exp(LV[i] + ln(k - K[i]))
        return S[i] + extra

    Ls, Lt = lorenz(groups(p, m)), lorenz(groups(pp, mp))
    return max(S_at(Lt, k) - S_at(Ls, k) for k in sorted(set(Ls[0]) | set(Lt[0])))


def H4(p):
    return 0.0 if p == 0 else -(1 - p) * log2(1 - p) - p * log2(p / 3)


def m3_rc() -> None:
    print("\n=== M3 tensor-power conversions (Proposition 0.13, Theorems 0.14, 0.15) ===")
    rng = np.random.default_rng(3)
    alphas = [-np.inf, -1.0, 0.5, 2.0, np.inf]
    ok_mono, ok_mult = True, True
    for _ in range(300):
        P = rng.random((3, 4)) ** 3
        P /= P.sum()
        Q = random_reduction(P, rng)
        for a in alphas:
            if a >= 1:
                ok_mono &= Phi(Q, a) <= Phi(P, a) + 1e-12
            else:
                ok_mono &= Phi(Q, a) >= Phi(P, a) - 1e-12
        ok_mono &= g_peak(Q) <= g_peak(P) + 1e-12
        R = rng.random((2, 4)) ** 2
        R /= R.sum()
        PR = np.einsum("ya,zb->yzab", P, R).reshape(6, 16)
        for a in alphas:
            ok_mult &= abs(Phi(PR, a) - Phi(P, a) * Phi(R, a)) <= 1e-12 * max(1, Phi(PR, a))
        ok_mult &= abs(g_peak(PR) - g_peak(P) * g_peak(R)) <= 1e-12
    check("M3 Phi_alpha (alpha in {-inf,-1,0.5,2,inf}) and g monotone under 300 random reductions", ok_mono)
    check("M3 Phi_alpha and g multiplicative under tensor products", ok_mult)
    rows = {}
    for name, (p, e, pp, ep) in (("trade pair (H_Q > H_P)", (0.0213, 0.05, 0.0, 0.25)),
                                 ("less erasure (H_Q < H_P)", (0.0213, 0.05, 0.0213, 0.04))):
        hP, hQ = 2 * e + (1 - e) * H4(p), 2 * ep + (1 - ep) * H4(pp)
        defs = [lorenz_deficit(p, int((1 - e) * n + 0.5), pp, int((1 - ep) * n + 0.5), n) for n in (10, 40, 100, 200)]
        rows[name] = dict(h_P=hP, h_Q=hQ, deficits=defs)
        print(f"    {name}: h_P={hP:.3f}, h_Q={hQ:.3f} bits; deficit at n = 10, 40, 100, 200: {['%.4f' % x for x in defs]}")
    a, b = rows["trade pair (H_Q > H_P)"]["deficits"], rows["less erasure (H_Q < H_P)"]["deficits"]
    check("M3 Lorenz deficit -> 0 for the trade pair, stays positive when H_Q < H_P", a[-1] < 1e-3 and a[0] > a[-1] and b[-1] > 0.1)
    OUT["M3"] = rows


# ----------------------------------------------------------------------------------------------------------------
def m5_weak_bursts() -> None:
    print("\n=== M5 weak bursts: thinning and the coverage bound (Theorem 4.24) ===")
    rng = np.random.default_rng(5)
    worst_law, ok_cov, ok_eta = 0.0, True, True
    for _ in range(40):
        V = int(rng.integers(2, 5))
        r = float(rng.uniform(0.01, 0.5))
        thetas = rng.uniform(0, 1, 3)
        wts = rng.dirichlet(np.ones(3))
        # exact law of the set uniformized at least once, original model: N ~ Poisson(r) events
        pats = list(itertools.product([0, 1], repeat=V))

        def one_event(th):
            return np.array([np.prod([th if b else 1 - th for b in pat]) for pat in pats])

        ev = sum(w * one_event(th) for w, th in zip(wts, thetas))          # law of the set A of one event
        idx = {pat: i for i, pat in enumerate(pats)}
        law = np.zeros(len(pats))
        law[0] = 1.0
        orig = np.zeros(len(pats))
        pk = exp(-r)
        for k in range(60):                                                   # union over k events
            orig += pk * law
            new = np.zeros(len(pats))
            for i, a in enumerate(pats):
                for j, b in enumerate(pats):
                    new[idx[tuple(x | y for x, y in zip(a, b))]] += law[i] * ev[j]
            law, pk = new, pk * r / (k + 1)
        # thinned model: independent Poisson per nonempty set with rate r * ev[A]
        thin = np.zeros(len(pats))
        thin[0] = 1.0
        for j, b in enumerate(pats):
            if not any(b):
                continue
            q = 1 - exp(-r * ev[j])
            new = (1 - q) * thin
            for i, a in enumerate(pats):
                new[idx[tuple(x | y for x, y in zip(a, b))]] += q * thin[i]
            thin = new
        worst_law = max(worst_law, float(np.abs(orig - thin).max()))
        # coverage bound (b) for qubit 0, threshold theta0
        th0 = float(rng.uniform(0.05, 0.9))
        RA = {pat: r * ev[j] for j, pat in enumerate(pats) if any(pat)}
        rho_m = sum(v for pat, v in RA.items() if pat[0] and sum(pat) >= 2)
        R0 = RA[tuple([1] + [0] * (V - 1))]
        rs = r * sum(w for w, th in zip(wts, thetas) if th >= th0)
        ok_cov &= rho_m <= rs + (V - 1) * th0 * (1 - th0) ** (1 - V) * R0 + 1e-12
        # certified efficiency: P(some qubit of a k-subset uniformized | one strong event) >= 1 - (1 - th0)^k
        k = int(rng.integers(1, V + 1))
        for w, th in zip(wts, thetas):
            if th >= th0:
                ok_eta &= 1 - (1 - th) ** k >= 1 - (1 - th0) ** k - 1e-15
    check("M5 thinning reproduces the law of the uniformized set (40 random models)", worst_law < 1e-12, f"(max diff {worst_law:.1e})")
    check("M5 correlated coverage <= strong rate + (V-1) th0 (1-th0)^(1-V) R_j", ok_cov)
    check("M5 efficiency of strong events >= 1/2 (1 - (1-th0)^k) (as a probability of uniformizing)", ok_eta)


# ----------------------------------------------------------------------------------------------------------------
def m6_peierls_bursts() -> None:
    print("\n=== M6 Peierls with bursts, one path (Theorem 4.23) ===")
    rng = np.random.default_rng(6)
    worst = -np.inf
    for _ in range(60):
        Lp = int(rng.integers(3, 8))
        p, e = float(rng.uniform(0, 0.1)), float(rng.choice([0.0, 0.05]))
        nreg = int(rng.integers(1, 4))
        regs = [tuple(sorted(rng.choice(Lp, int(rng.integers(1, min(Lp, 3) + 1)), replace=False))) for _ in range(nreg)]
        rhos = rng.uniform(0, 0.2, nreg)
        pbs = rng.uniform(0, 0.75, nreg)
        V = max(len(z) for z in regs)
        rho_bar = max(sum(rh for z, rh in zip(regs, rhos) if j in z) for j in range(Lp))
        # exact probability that, among unflagged qubits, at least half carry an X-type error
        bad = 0.0
        for act in itertools.product([0, 1], repeat=nreg):
            wact = float(np.prod([rh if a else 1 - rh for a, rh in zip(act, rhos)]))
            s = np.full(Lp, p)
            for a, z, pb in zip(act, regs, pbs):
                if a:
                    for j in z:
                        s[j] = s[j] + pb - 4 * s[j] * pb / 3
            qx = 2 * s / 3
            for flags in itertools.product([0, 1], repeat=Lp):
                wf = float(np.prod([e if f else 1 - e for f in flags]))
                un = [j for j in range(Lp) if not flags[j]]
                for xs in itertools.product([0, 1], repeat=len(un)):
                    if 2 * sum(xs) >= len(un):
                        bad += wact * wf * float(np.prod([qx[j] if x else 1 - qx[j] for j, x in zip(un, xs)]))
        q = 2 * p / 3
        best = np.inf
        for sig in np.linspace(1.0, 12.0, 45):
            m = lambda t: e + (1 - e) * ((1 - t) / sig + t * sig)
            kap = m(0.5) / m(q)
            lk = V * log(kap)
            if lk < 700:                                                     # otherwise the bound is useless
                best = min(best, m(q) * exp(min(rho_bar * (exp(lk) - 1) / V, 700.0)))
        worst = max(worst, log(bad) - Lp * log(best) if bad > 0 else -np.inf)
    check("M6 P[path bad] <= B_{M,b}^|path| (60 random short paths with regions)", worst <= 1e-9, f"(max log excess {worst:.2f})")


# ----------------------------------------------------------------------------------------------------------------
def oplus_f(a, b):
    return a + b - Fraction(4, 3) * a * b


def m7_o1_exact() -> None:
    print("\n=== M7 O1 counterexamples in exact arithmetic (Observations 4.25, 4.26) ===")
    st = code_structure(esc.rotated_surface_d3())
    syn_of = dict(zip(st.cls.tolist(), st.syn.tolist()))
    p, rho, pb = Fraction(1, 200), Fraction(1, 20), Fraction(3, 4)

    def eps_region(R):
        R = list(R)
        out = [j for j in range(9) if j not in R]
        kin = st.nz[:, R].sum(axis=1).tolist()
        kout = st.nz[:, out].sum(axis=1).tolist()
        q = oplus_f(p, pb)
        f0 = lambda k, n: (p / 3) ** k * (1 - p) ** (n - k)
        f1 = lambda k, n: (q / 3) ** k * (1 - q) ** (n - k)
        nin, nout = len(R), len(out)
        val = {(a, b): (1 - rho) * f0(a, nin) * f0(b, nout) + rho * f1(a, nin) * f0(b, nout)
               for a in range(nin + 1) for b in range(nout + 1)}
        W = {}
        for c, a, b in zip(st.cls.tolist(), kin, kout):
            W[c] = W.get(c, 0) + val[(a, b)]
        best = {}
        for c, w in W.items():
            s = syn_of[c]
            best[s] = max(best.get(s, Fraction(0)), w)
        return 1 - sum(best.values())

    e3, e4 = eps_region((0, 1, 3)), eps_region((0, 1, 3, 4))
    print(f"    region {{0,1,3}} -> {{0,1,3,4}}: {float(e3):.4e} -> {float(e4):.4e}")
    check("M7 region growth lowers eps* (exact rationals)", e4 < e3)
    # duration: bit-flip repetition code, 3 rounds, perfect measurements
    R_, Lr = 3, 9
    F = list(itertools.product([0, 1], repeat=Lr))
    H = [(1, 1, 0), (0, 1, 1)]

    def key(f):
        cum = [0, 0, 0]
        s_prev = (0, 0)
        det = []
        for t in range(R_):
            cum = [c ^ f[3 * t + j] for j, c in enumerate(cum)]
            s = tuple(sum(h[j] * cum[j] for j in range(3)) % 2 for h in H)
            det += [s[0] ^ s_prev[0], s[1] ^ s_prev[1]]
            s_prev = s
        return tuple(det), cum[0]

    keys = [key(f) for f in F]
    pp, rho2, pb2 = Fraction(1, 100), Fraction(1, 20), Fraction(1, 2)

    def eps_dur(T):
        comps = [(1 - rho2, [pp] * Lr)]
        for t0 in range(R_):
            q = [pp] * Lr
            for t in range(t0, min(R_, t0 + T)):
                for j in (0, 1):
                    i = 3 * t + j
                    q[i] = q[i] + pb2 - 2 * q[i] * pb2
            comps.append((rho2 / R_, q))
        tab = {}
        for f, (det, ob) in zip(F, keys):
            pr = sum(w * np.prod([qi if b else 1 - qi for qi, b in zip(q, f)]) for w, q in comps)
            tab.setdefault(det, [Fraction(0), Fraction(0)])
            tab[det][ob] += pr
        return 1 - sum(max(v) for v in tab.values())

    d1, d2 = eps_dur(1), eps_dur(2)
    print(f"    duration 1 -> 2 rounds: {float(d1):.4e} -> {float(d2):.4e}")
    check("M7 longer burst lowers eps* (exact rationals)", d2 < d1)
    OUT["M7"] = dict(region=[float(e3), float(e4)], duration=[float(d1), float(d2)])


# ----------------------------------------------------------------------------------------------------------------
def pauli(s):
    n = len(s)
    v = np.zeros(2 * n, np.uint8)
    for i, ch in enumerate(s):
        if ch in "XY":
            v[i] = 1
        if ch in "ZY":
            v[n + i] = 1
    return v


def span(B, n2):
    out = [np.zeros(n2, np.uint8)]
    for b in B:
        out += [o ^ b for o in out]
    return np.array(out)


def code_from_state(G, inputs):
    N = G.shape[1] // 2
    outl = [i for i in range(N) if i not in inputs]
    allg = span(rref(G)[0], 2 * N)
    io = outl + [N + i for i in outl]
    ii = list(inputs) + [N + i for i in inputs]
    S = [g[io] for g in allg if not g[ii].any() and g[io].any()]
    return (rref(np.array(S))[0] if S else np.zeros((0, 2 * len(outl)), np.uint8)), len(outl)


def e1_and_ties(st, p):
    syn_of = np.full(st.ncls, -1)
    syn_of[st.cls] = st.syn

    def W(A):
        mask = np.ones(st.n, bool)
        mask[list(A)] = False
        k = st.nz[:, mask].sum(axis=1)
        m = (st.n - len(A)) - k
        return np.bincount(st.cls, weights=0.25 ** len(A) * (1 - p) ** m * (p / 3) ** k, minlength=st.ncls)

    W0 = W(())
    bad, ties = 0, 0
    syns = np.unique(st.syn)
    opt = {}
    for s in syns:
        cl = np.flatnonzero(syn_of == s)
        opt[s] = set(cl[W0[cl] >= W0[cl].max() * (1 - 1e-9)])
        ties += len(opt[s]) > 1
    for j in range(st.n):
        Wj = W((j,))
        for s in syns:
            cl = np.flatnonzero(syn_of == s)
            bad += not (opt[s] & set(cl[Wj[cl] >= Wj[cl].max() * (1 - 1e-9)]))
    return bad, ties


def m8_ame() -> None:
    print("\n=== M8 codes from qubit AME states (Observation 3.38) ===")
    states = {"AME(3)": ["XXX", "ZZI", "IZZ"],
              "AME(5)": ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ", "ZZZZZ"],
              "AME(6)": [s + "I" for s in ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]] + ["XXXXXX", "ZZZZZZ"]}
    ps = (1e-3, 0.01, 0.05, 0.2, 0.5)
    rows, all_e1, only513 = [], True, True
    for name, gens in states.items():
        G = np.array([pauli(s) for s in gens])
        N = G.shape[1] // 2
        allg = span(rref(G)[0], 2 * N)[1:]
        minw = int(min(np.count_nonzero(g[:N] | g[N:]) for g in allg))
        assert minw == N // 2 + 1, (name, minw)                              # AME: every nontrivial element > N/2
        for k in (1, 2):
            if N - k < 2:
                continue
            for inputs in itertools.combinations(range(N), k):
                S, n = code_from_state(G, list(inputs))
                st = Structure(n, S, symp_complement(S, n))
                res = [e1_and_ties(st, p) for p in ps]
                Rc = [exact_rate(st, p)[0] / (0.75 - p) for p in ps]
                e1 = all(b == 0 for b, _ in res)
                unique = all(t == 0 for _, t in res)
                attains = all(abs(x - 1) < 1e-6 for x in Rc)
                all_e1 &= e1
                if attains and not (n == 5 and k == 1):
                    only513 = False
                rows.append(dict(state=name, inputs=list(inputs), n=n, k=k, E1=e1, unique_ML=unique,
                                 R_over_c=[round(float(x), 4) for x in Rc]))
    seen = {}
    for r in rows:
        seen.setdefault((r["state"], r["n"], r["k"]), []).append(r)
    for (st_, n, k), rs in seen.items():
        rc = sorted({tuple(r["R_over_c"]) for r in rs})
        print(f"    {st_} -> [[{n},{k}]] ({len(rs)} input choices): E1 {all(r['E1'] for r in rs)}, unique ML {all(r['unique_ML'] for r in rs)}, R/c at p={ps}: {rc}")
    check("M8 every code from a qubit AME state (all input choices) satisfies (E1) at all tested p", all_e1)
    check("M8 only [[5,1,3]] attains R = c", only513)
    OUT["M8"] = rows


def main() -> int:
    m1_erasure()
    m2_degeneracy()
    m3_rc()
    m5_weak_bursts()
    m6_peierls_bursts()
    m7_o1_exact()
    m8_ame()
    OUT["failures"] = failures
    (REPO / "results" / "third_pass_checks.json").write_text(json.dumps(OUT, indent=1, default=float))
    print("\nSUMMARY:", "ALL THIRD-PASS CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
