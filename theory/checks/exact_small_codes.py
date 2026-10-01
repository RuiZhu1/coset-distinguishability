"""Exact finite-size checks of the theory notes (theory/*.tex).

For small stabilizer codes (n <= 9) we compute the *exact* ML (Bayes-optimal)
logical failure probability eps*(p, e) of the code-capacity noise model N(p, e):

    each qubit independently: with prob. e  -> erased (flag = 1, Pauli uniform on {I,X,Y,Z})
                              otherwise     -> depolarizing with rate p (X,Y,Z each p/3).

and verify, with no Monte Carlo and no decoder approximation, the finite-d
statements that the theory relies on:

  C1  monotonicity in p and in e                                   (Thm 1.2 / Prop 1.4)
  C2  superposition of independent depolarizing noise              (Thm 1.2, F2)
  C3  partial flag discarding: eps*(p,e0+d) <= eps*(p'',e0)        (Thm 3.1)
          p'' = p + d (3/4 - p) / (1 - e0)
  C4  local-free-preorder necessity                                (Thm 3.3)
  C5  Bhattacharyya (union) upper bound  eps* <= W(B)/2            (Lemma 4.1)
  C6  genie lower bound eps* >= eps*_genie  (rotated d=3 only)     (Prop 5.1)
  C7  finite-d exchange rate  R^(d) = (d eps*/de)/(d eps*/dp)  vs  the bound c = (3/4-p)/(1-e)

Run:  python theory/checks/exact_small_codes.py
Exit status is non-zero if any *inequality* check is violated beyond 1e-12.
"""
import itertools
import sys

import numpy as np

# --------------------------------------------------------------------------
# GF(2) helpers, codes
# --------------------------------------------------------------------------

def gf2_rank_basis(rows):
    """Row-reduce; return (basis rows as array, pivots)."""
    M = np.array(rows, dtype=np.uint8) % 2
    piv, r = [], 0
    for c in range(M.shape[1]):
        k = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if k is None:
            continue
        M[[r, k]] = M[[k, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv


def in_span(basis, v):
    if len(basis) == 0:
        return not v.any()
    B, _ = gf2_rank_basis(np.vstack([basis, v]))
    return len(B) == len(basis)


def nullspace(A):
    """Basis of {v : A v = 0 mod 2}."""
    A = np.array(A, dtype=np.uint8) % 2
    m, n = A.shape
    R, piv = gf2_rank_basis(A)
    free = [c for c in range(n) if c not in piv]
    out = []
    for f in free:
        v = np.zeros(n, np.uint8)
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = R[i, f]
        out.append(v)
    return np.array(out, dtype=np.uint8)


def symp(a, b, n):
    """Symplectic product of (x|z) vectors."""
    return int((a[:n] @ b[n:] + a[n:] @ b[:n]) % 2)


def pauli_from_string(s):
    n = len(s)
    v = np.zeros(2 * n, np.uint8)
    for j, ch in enumerate(s):
        if ch in "XY":
            v[j] = 1
        if ch in "ZY":
            v[n + j] = 1
    return v


def css(HX, HZ):
    n = HX.shape[1]
    rows = [np.concatenate([h, np.zeros(n, np.uint8)]) for h in HX]
    rows += [np.concatenate([np.zeros(n, np.uint8), h]) for h in HZ]
    return np.array(rows, dtype=np.uint8)


def rotated_surface_d3():
    d = 3
    idx = lambda r, c: r * d + c
    HX, HZ = [], []
    for r in range(-1, d):
        for c in range(-1, d):
            bulk = 0 <= r <= d - 2 and 0 <= c <= d - 2
            kind = 'X' if (r + c) % 2 == 0 else 'Z'
            qs = [idx(rr, cc) for rr, cc in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
                  if 0 <= rr < d and 0 <= cc < d]
            row = np.zeros(d * d, np.uint8)
            row[qs] = 1
            if bulk:
                (HX if kind == 'X' else HZ).append(row)
            elif r in (-1, d - 1) and 0 <= c <= d - 2 and kind == 'X':
                HX.append(row)                      # X-type boundary checks: top / bottom
            elif c in (-1, d - 1) and 0 <= r <= d - 2 and kind == 'Z':
                HZ.append(row)                      # Z-type boundary checks: left / right
    return css(np.array(HX), np.array(HZ))


def steane():
    H = np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], np.uint8)
    return css(H, H)


def five_qubit():
    return np.array([pauli_from_string(s) for s in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")], np.uint8)


class Code:
    def __init__(self, name, S):
        self.name, self.S = name, S
        self.n = S.shape[1] // 2
        n = self.n
        # syndrome of v: <v, g> for each generator g
        self.r = len(S)
        Sb, _ = gf2_rank_basis(S)
        assert len(Sb) == self.r == n - 1, (name, len(Sb), self.r)
        # normalizer = nullspace of S * Omega
        Om = np.zeros((2 * n, 2 * n), np.uint8)
        Om[:n, n:] = np.eye(n, dtype=np.uint8)
        Om[n:, :n] = np.eye(n, dtype=np.uint8)
        self.N = nullspace((S @ Om) % 2)           # basis of N(S), dim n+1
        # logical pair (Xbar, Zbar) in N(S) \ S with <Xbar,Zbar> = 1
        found = None
        for v1 in self.N:
            if in_span(Sb, v1):
                continue
            for v2 in self.N:
                if not in_span(Sb, v2) and symp(v1, v2, n) == 1:
                    found = (v1, v2)
                    break
            if found:
                break
        self.Xbar, self.Zbar = found
        # enumerate all 4^n Paulis: syndrome id and raw logical label
        idx = np.arange(4 ** n, dtype=np.int64)
        a = np.stack([(idx >> (2 * (n - 1 - j))) & 3 for j in range(n)], axis=1)  # 0=I 1=X 2=Z 3=Y
        x = (a & 1).astype(np.uint8)
        z = (a >> 1).astype(np.uint8)
        V = np.concatenate([x, z], axis=1)                       # (4^n, 2n)
        self.V = V
        Sm = (S @ Om) % 2                                       # <v,g_i> = v . (Om g_i)
        synd = (V.astype(np.int64) @ Sm.T.astype(np.int64)) % 2
        self.syn = synd @ (1 << np.arange(self.r))
        lx = (V.astype(np.int64) @ ((Om @ self.Zbar) % 2).astype(np.int64)) % 2   # <E, Zbar>
        lz = (V.astype(np.int64) @ ((Om @ self.Xbar) % 2).astype(np.int64)) % 2   # <E, Xbar>
        self.lab = lx + 2 * lz
        self.joint_idx = self.syn * 4 + self.lab
        self.nsyn = 1 << self.r
        self.pop = np.array([bin(m).count("1") for m in range(1 << n)])


def dep_vec(p):
    return np.array([1 - p, p / 3, p / 3, p / 3])   # order I,X,Z,Y -- symmetric, order irrelevant


def eps_given_flags(code, p):
    """g[mask] = ML failure prob given erased set = mask (bit j <-> qubit j)."""
    n = code.n
    g = np.zeros(1 << n)
    q_dep, q_er = dep_vec(p), np.full(4, 0.25)
    for mask in range(1 << n):
        P = np.ones(1)
        for j in range(n):
            P = np.kron(P, q_er if (mask >> j) & 1 else q_dep)
        joint = np.bincount(code.joint_idx, weights=P, minlength=code.nsyn * 4).reshape(code.nsyn, 4)
        g[mask] = 1.0 - joint.max(axis=1).sum()
    return g


class Eps:
    """eps*(p,e) with caching of the flag-pattern tables g_p."""
    def __init__(self, code):
        self.code, self.cache = code, {}

    def table(self, p):
        key = round(float(p), 12)
        if key not in self.cache:
            self.cache[key] = eps_given_flags(self.code, float(p))
        return self.cache[key]

    def __call__(self, p, e):
        n = self.code.n
        g = self.table(p)
        k = self.code.pop
        w = (e ** k) * ((1 - e) ** (n - k))
        return float((w * g).sum())


# --------------------------------------------------------------------------
# theory quantities
# --------------------------------------------------------------------------

def lam(p):            # Pauli eigenvalue of depolarizing(p)
    return 1 - 4 * p / 3

def beta_dep(p):       # Bhattacharyya coefficient of {P_p, P_p shifted by a Pauli}
    return 2 * np.sqrt(p * (1 - p) / 3) + 2 * p / 3

def B(p, e):
    return e + (1 - e) * beta_dep(p)

def bound_c(p, e):     # rigorous upper bound on the exchange rate (Thm 3.1)
    return (0.75 - p) / (1 - e)

def reachable(p, e, p2, e2, tol=1e-15):
    """Local free preorder (Thm 3.3): can (p2,e2) be obtained from (p,e)?"""
    mu, mu2 = (1 - e) * lam(p), (1 - e2) * lam(p2)
    if e2 >= e:
        return lam(p2) <= lam(p) + tol
    return mu2 <= mu + tol


def weight_enum_N_minus_S(code, x):
    n = code.n
    # enumerate N(S) (2^(n+1) elements) and S
    def span(basis):
        out = [np.zeros(2 * n, np.uint8)]
        for b in basis:
            out += [(o ^ b) for o in out]
        return out
    Sb, _ = gf2_rank_basis(code.S)
    Sset = {tuple(v) for v in span(Sb)}
    tot = 0.0
    for v in span(code.N):
        if tuple(v) in Sset:
            continue
        w = int(np.count_nonzero(v[:n] | v[n:]))
        tot += x ** w
    return tot


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

TOL = 1e-12
PGRID = np.array([0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75])
failures = []

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def run_code(code, rng):
    print(f"\n=== {code.name}: n={code.n}, 4^n={4**code.n} Paulis ===")
    eps = Eps(code)
    ps = PGRID
    es = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])

    # C1 monotonicity
    grid = np.array([[eps(p, e) for e in es] for p in ps])
    dp = np.diff(grid, axis=0).min()
    de = np.diff(grid, axis=1).min()
    check("C1 eps* non-decreasing in p", dp > -TOL, f"(min step {dp:.2e})")
    check("C1 eps* non-decreasing in e", de > -TOL, f"(min step {de:.2e})")

    # C2 superposition of independent depolarizing noise: p1 -> p2 >= p1 is "add depolarizing with
    #    lambda_q = lambda(p2)/lambda(p1)"; erasure: e1 -> e2 >= e1 is "add erasure".  (Same data as C1,
    #    but tested on random pairs, including simultaneous changes.)
    worst = np.inf
    for _ in range(200):
        i, j = sorted(rng.integers(0, len(PGRID), 2))
        e1 = rng.uniform(0, 0.8)
        e2 = e1 + (1 - e1) * rng.uniform(0, 1)       # e2 = 1-(1-e1)(1-delta)
        worst = min(worst, eps(PGRID[j], e2) - eps(PGRID[i], e1))
    check("C2 adding independent depolarizing / erasure layers never decreases eps*", worst > -TOL,
          f"(min slack {worst:.2e})")

    # C3 partial flag discarding.  Draw (p, p'') from the grid and e0, then solve for delta.
    worst, rel = np.inf, np.inf
    n3 = 0
    for _ in range(400):
        i, j = sorted(rng.integers(0, len(PGRID), 2))
        p, p2 = PGRID[i], PGRID[j]
        if p2 >= 0.75 or p2 == p:
            continue
        e0 = rng.uniform(0, 0.6)
        dlt = (p2 - p) * (1 - e0) / (0.75 - p)       # solves p'' = p + dlt (3/4 - p)/(1 - e0)
        if e0 + dlt > 1:
            continue
        slack = eps(p2, e0) - eps(p, e0 + dlt)
        worst = min(worst, slack)
        rel = min(rel, slack / max(eps(p2, e0), 1e-300))
        n3 += 1
    check("C3 eps*(p, e0+d) <= eps*(p'', e0)  (partial flag discarding)", worst > -TOL,
          f"({n3} cases; min slack {worst:.2e}; min relative slack {rel:.2e})")

    # C4 local-free-preorder necessity, and size of the operational-vs-free gap
    viol, n_reach, n_op_unreach, n_unreach = 0, 0, 0, 0
    for _ in range(3000):
        p, p2 = PGRID[rng.integers(0, len(PGRID))], PGRID[rng.integers(0, len(PGRID))]
        e, e2 = rng.uniform(0, 0.8), rng.uniform(0, 0.8)
        if reachable(p, e, p2, e2):
            n_reach += 1
            if eps(p2, e2) < eps(p, e) - TOL:
                viol += 1
        else:
            n_unreach += 1
            if eps(p2, e2) >= eps(p, e) - TOL:
                n_op_unreach += 1
    check("C4 free-reachable pairs are eps*-ordered", viol == 0,
          f"({n_reach} reachable pairs; of {n_unreach} non-reachable pairs, {n_op_unreach} are nevertheless eps*-ordered)")

    # C5 Bhattacharyya union bound
    worst = np.inf
    for p in (0.01, 0.02, 0.04, 0.1):
        for e in (0.0, 0.05, 0.2):
            W = weight_enum_N_minus_S(code, B(p, e))
            worst = min(worst, 0.5 * W - eps(p, e))
    check("C5 eps* <= (1/2) W_{N\\S}(B(p,e))", worst > -TOL, f"(min slack {worst:.2e})")

    # C7 finite-d exchange rate vs the rigorous bound
    print("  C7 finite-d rate R^(d)=(d eps*/de)/(d eps*/dp)  vs  bound c=(3/4-p)/(1-e)  [R_B, R_hash heuristics]")
    print("      p0     e0     R^(d)    c(bound)  R_B     R_hash")
    ok = True
    for p0 in (0.02, 0.04, 0.06):
        for e0 in (0.0, 0.02, 0.05):
            h = 1e-5
            dEdp = (eps(p0 + h, e0) - eps(p0 - h, e0)) / (2 * h)
            dEde = (eps(p0, e0 + h) - eps(p0, e0 - h)) / (2 * h) if e0 > 0 else (eps(p0, h) - eps(p0, 0)) / h
            R = dEde / dEdp
            c = bound_c(p0, e0)
            bb = (1 - beta_dep(p0)) / ((1 - e0) * (beta_prime(p0)))
            H4 = -(1 - p0) * np.log2(1 - p0) - p0 * np.log2(p0 / 3)
            Rh = (2 - H4) / ((1 - e0) * np.log2(3 * (1 - p0) / p0))
            ok &= (R <= c + 1e-9)
            print(f"      {p0:.2f}   {e0:.2f}   {R:7.4f}  {c:7.4f}  {bb:7.4f} {Rh:7.4f}")
    check("C7 R^(d) <= c at all work points", ok)


def beta_prime(p, h=1e-7):
    return (beta_dep(p + h) - beta_dep(p - h)) / (2 * h)


def genie_check(code):
    """Prop 5.1 at d=3 rotated surface code: eps* >= genie-aided error on an interior column."""
    n, d = code.n, 3
    T = [1, 4, 7]                                   # middle column (r, c=1)
    Xb = np.zeros(2 * n, np.uint8); Xb[T] = 1
    # (i) no stabilizer supported inside T, (ii) ker(syndrome) restricted to P_T = {I, Xbar_T}
    Sb, _ = gf2_rank_basis(code.S)
    sup_in_T = 0
    for bits in itertools.product([0, 1], repeat=len(Sb)):
        v = np.zeros(2 * n, np.uint8)
        for b, row in zip(bits, Sb):
            if b:
                v ^= row
        supp = set(np.nonzero(v[:n] | v[n:])[0])
        if supp and supp <= set(T):
            sup_in_T += 1
    kernel_T = 0
    Om = np.zeros((2 * n, 2 * n), np.uint8); Om[:n, n:] = np.eye(n, dtype=np.uint8); Om[n:, :n] = np.eye(n, dtype=np.uint8)
    for a in itertools.product(range(4), repeat=len(T)):
        v = np.zeros(2 * n, np.uint8)
        for j, aj in zip(T, a):
            v[j] = aj & 1
            v[n + j] = aj >> 1
        if not ((code.S @ Om @ v) % 2).any():
            kernel_T += 1
    assert sup_in_T == 0 and kernel_T == 2, (sup_in_T, kernel_T)
    assert in_span(gf2_rank_basis(code.N)[0], Xb)    # Xbar_T commutes with S

    from math import comb

    def genie(p, e):
        q = dep_vec(p)                               # index 0=I,1=X,2=Z,3=Y ; adding Xbar flips bit 0
        tot = 0.0
        for u in range(d + 1):                       # u = number of unflagged qubits on T
            w_u = comb(d, u) * (1 - e) ** u * e ** (d - u)
            s_u = 0.0
            for a in itertools.product(range(4), repeat=u):
                P = np.prod([q[aj] for aj in a]) if u else 1.0
                Q = np.prod([q[aj ^ 1] for aj in a]) if u else 1.0
                s_u += min(P, Q)
            tot += w_u * 0.5 * s_u
        return tot

    eps = Eps(code)
    worst = np.inf
    for p in (0.01, 0.02, 0.04, 0.1, 0.2):
        for e in (0.0, 0.05, 0.2, 0.5):
            worst = min(worst, eps(p, e) - genie(p, e))
    check("C6 eps* >= genie-aided error on interior column", worst > -TOL, f"(min slack {worst:.2e})")


def main():
    rng = np.random.default_rng(0)
    for code in (Code("[[5,1,3]] five-qubit", five_qubit()),
                 Code("[[7,1,3]] Steane", steane()),
                 Code("[[9,1,3]] rotated surface d=3", rotated_surface_d3())):
        run_code(code, rng)
        if code.n == 9:
            genie_check(code)
    print("\nSUMMARY:", "ALL INEQUALITY CHECKS PASSED" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
