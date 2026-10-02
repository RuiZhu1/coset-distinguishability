"""Falsification tests for the resource-theoretic structure of the notes (theory/sec0-constitution.tex).

Everything here is an exact finite computation (linear programs over simulable reductions in kernel form); no sampling.

  N1  single-qubit Pauli+erasure class: LP feasibility of "exists a simulable reduction P(p,e) -> P(p',e')" must agree
      with condition (ii) of Theorem 3.7 (this tests Proposition 0.10, which is stronger than Theorem 3.7: arbitrary
      label permutations, not only translations), and the monotones p and mu=(1-e)lambda(p) must be ordered on every
      reachable pair.
  N2  completeness (Theorem 0.6): for random finite objects, primal feasibility of P -> Q agrees with the dual
      certificate LP (a loss c with Phi_c(Q) > Phi_c(P)); for infeasible pairs the certificate is re-evaluated directly.
  N3  every reachable pair is ordered by many random members of the complete family (monotonicity).
  N4  parallel composition: 1 - eps*(P (x) Q) = (1 - eps*(P)) (1 - eps*(Q)).
  N5  eps* alone is not complete: explicit pair with eps*(P) < eps*(Q) but Q not reachable from P.
  N6  golden rule: every object reaches every free object; eps* is faithful (maximal iff free).
  N7  the reference monotones of sec5: the Bhattacharyya parameter B(p,e) is nondecreasing and the hashing rate
      Q_hash(p,e) = 1 - 2e - (1-e) H_4(p) is nonincreasing on every pair satisfying condition (ii); on unreachable pairs
      both directions occur (so the test is not vacuous).  Q_hash is NUMERICAL only (no proof in the notes).

    python theory/checks/constitution.py          # about 20 s; non-zero exit status on any violation
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
from scipy.optimize import linprog

RNG = np.random.default_rng(20241101)
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- objects, reductions in kernel form
def perms(m: int) -> np.ndarray:
    """All permutations of range(m), shape (m!, m); pi[l] = image of label l."""
    return np.array(list(itertools.permutations(range(m))))


def permute(v: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """(pi v)(pi(l)) = v(l)."""
    w = np.empty_like(v)
    w[pi] = v
    return w


def reachable_lp(P: np.ndarray, Q: np.ndarray) -> bool:
    """Is there a kernel M(y',pi|y) with Q(.,y') = sum_{y,pi} M(y',pi|y) pi P(.,y)?   P: (|Lam|,|Y|), Q: (|Lam|,|Y'|)."""
    lam, ny = P.shape
    nyp = Q.shape[1]
    S = perms(lam)
    ns = len(S)
    nvar = ny * nyp * ns
    idx = lambda y, yp, s: (y * nyp + yp) * ns + s
    A_eq, b_eq = [], []
    for y in range(ny):                                   # each input outcome is processed with total probability one
        row = np.zeros(nvar)
        for yp in range(nyp):
            for s in range(ns):
                row[idx(y, yp, s)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    for yp in range(nyp):                                  # output joint law
        for l in range(lam):
            row = np.zeros(nvar)
            for y in range(ny):
                for s in range(ns):
                    row[idx(y, yp, s)] = permute(P[:, y], S[s])[l]
            A_eq.append(row)
            b_eq.append(Q[l, yp])
    res = linprog(np.zeros(nvar), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method="highs")
    return res.status == 0


def dual_certificate(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray]:
    """max over c in [-1,1]^{Lam x Y'} of <c,Q> - sum_y max_{y',pi} <c_{y'}, pi P_y>   (LP); > 0 iff Q is not reachable."""
    lam, ny = P.shape
    nyp = Q.shape[1]
    S = perms(lam)
    nc = lam * nyp
    # variables: c (flattened [yp*lam + l]), t_y
    obj = np.concatenate([-Q.T.reshape(-1), np.ones(ny)])
    A_ub, b_ub = [], []
    for y in range(ny):
        for yp in range(nyp):
            for s in S:
                row = np.zeros(nc + ny)
                row[yp * lam:(yp + 1) * lam] = permute(P[:, y], s)
                row[nc + y] = -1.0
                A_ub.append(row)
                b_ub.append(0.0)
    bounds = [(-1, 1)] * nc + [(None, None)] * ny
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    c = res.x[:nc].reshape(nyp, lam)
    return -res.fun, c


def phi_c(v: np.ndarray, c: np.ndarray, S: np.ndarray) -> float:
    return max(float(c[yp] @ permute(v, s)) for yp in range(c.shape[0]) for s in S)


def Phi_c(P: np.ndarray, c: np.ndarray, S: np.ndarray) -> float:
    return sum(phi_c(P[:, y], c, S) for y in range(P.shape[1]))


def eps_star(P: np.ndarray) -> float:
    return 1.0 - float(P.max(axis=0).sum())


def ncc_single_qubit(p: float, e: float) -> np.ndarray:
    """Joint law of (label = Pauli in order I,X,Y,Z ; flag) for one qubit of N_cc(p,e); shape (4,2)."""
    P = np.zeros((4, 2))
    P[:, 0] = (1 - e) * np.array([1 - p, p / 3, p / 3, p / 3])
    P[:, 1] = e / 4
    return P


def lam(p: float) -> float:
    return 1 - 4 * p / 3


def cond_ii(p, e, pp, ep) -> tuple[bool, float]:
    """Condition (ii) of Theorem 3.7 and its (signed) slack."""
    if ep >= e:
        slack = lam(p) - lam(pp)
        return slack >= 0, slack
    slack = (1 - e) * lam(p) - (1 - ep) * lam(pp)
    return slack >= 0, slack


# ---------------------------------------------------------------- N1
def n1(npairs: int = 300) -> None:
    print("N1: single-qubit class, LP reachability vs Theorem 3.7 (ii)")
    bad, mono_bad, n_reach, n_unreach, n_tight = 0, 0, 0, 0, 0
    for _ in range(npairs):
        p, pp = RNG.uniform(0, 0.74, 2)
        e, ep = RNG.uniform(0, 0.9, 2)
        if RNG.random() < 0.5:                       # stress the boundary mu' = mu (e' < e): move slightly off it
            e = RNG.uniform(0.05, 0.9)
            ep = RNG.uniform(0, e)
            lam_target = (1 - e) * lam(p) / (1 - ep)
            pp = 0.75 * (1 - lam_target) + RNG.choice([-1, 1]) * 10 ** RNG.uniform(-4, -2)
            pp = float(np.clip(pp, 0, 0.74))
        pred, slack = cond_ii(p, e, pp, ep)
        if abs(slack) < 1e-7:
            continue
        got = reachable_lp(ncc_single_qubit(p, e), ncc_single_qubit(pp, ep))
        n_reach += got
        n_unreach += (not got)
        n_tight += abs(slack) < 1e-3
        if got != pred:
            bad += 1
            print(f"    mismatch p={p:.4f} e={e:.4f} -> p'={pp:.4f} e'={ep:.4f}: LP {got}, (ii) {pred}, slack {slack:.2e}")
        if got:                                      # monotones of the local order: p nondecreasing, mu nonincreasing
            if pp < p - 1e-9 or (1 - ep) * lam(pp) > (1 - e) * lam(p) + 1e-9:
                mono_bad += 1
    check("LP reachability == condition (ii)", bad == 0, f"({n_reach} reachable, {n_unreach} unreachable, {n_tight} within 1e-3 of the boundary)")
    check("p is nondecreasing and mu nonincreasing on all LP-reachable pairs", mono_bad == 0)
    # exact boundary: mu' = mu, e' < e must be reachable (tolerance of the LP)
    ok = True
    for _ in range(20):
        p = RNG.uniform(0, 0.7)
        e = RNG.uniform(0.1, 0.9)
        ep = RNG.uniform(0, e)
        pp = 0.75 * (1 - (1 - e) * lam(p) / (1 - ep))
        ok &= reachable_lp(ncc_single_qubit(p, e), ncc_single_qubit(pp, ep))
    check("exact boundary mu' = mu (e' < e) is reachable", ok)


# ---------------------------------------------------------------- N2, N3
def random_object(lam_: int, ny: int) -> np.ndarray:
    P = RNG.random((lam_, ny)) ** RNG.uniform(1, 4)
    return P / P.sum()


def random_reduction_apply(P: np.ndarray, nyp: int) -> np.ndarray:
    lam_, ny = P.shape
    S = perms(lam_)
    Q = np.zeros((lam_, nyp))
    for y in range(ny):
        M = RNG.dirichlet(np.ones(nyp * len(S)) * 0.3).reshape(nyp, len(S))
        for yp in range(nyp):
            for s in range(len(S)):
                Q[:, yp] += M[yp, s] * permute(P[:, y], S[s])
    return Q


def n2_n3(ntrial: int = 60) -> None:
    print("N2/N3: completeness of the family Phi_c (random finite objects)")
    agree, viol_mono, viol_cert = 0, 0, 0
    nfeas = ninf = 0
    for t in range(ntrial):
        lam_ = int(RNG.choice([2, 3, 4]))
        ny, nyp = int(RNG.integers(2, 4)), int(RNG.integers(2, 4))
        P = random_object(lam_, ny)
        if t % 2 == 0:
            Q = random_reduction_apply(P, nyp)                  # reachable by construction
        else:
            Q = random_object(lam_, nyp)                        # usually unreachable
        S = perms(lam_)
        feas = reachable_lp(P, Q)
        val, c = dual_certificate(P, Q)
        nfeas += feas
        ninf += (not feas)
        agree += (feas == (val <= 1e-9))
        if not feas:
            gap = Phi_c(Q, c, S) - Phi_c(P, c, S)                # direct re-evaluation of the certificate
            viol_cert += gap <= 1e-12
        else:
            for _ in range(200):                                 # monotonicity under random members of the family
                cc = RNG.normal(size=(nyp, lam_))
                if Phi_c(Q, cc, S) > Phi_c(P, cc, S) + 1e-10:
                    viol_mono += 1
                    break
    check("primal feasibility == (dual certificate value <= 0)", agree == ntrial, f"({nfeas} feasible, {ninf} infeasible)")
    check("every infeasible pair has a loss c with Phi_c(Q) > Phi_c(P)", viol_cert == 0)
    check("Phi_c(Q) <= Phi_c(P) on every reachable pair (200 random c each)", viol_mono == 0)


# ---------------------------------------------------------------- N4, N5, N6
def n4() -> None:
    print("N4: parallel composition")
    worst = 0.0
    for _ in range(200):
        P, Q = random_object(int(RNG.integers(2, 5)), int(RNG.integers(1, 5))), random_object(int(RNG.integers(2, 5)), int(RNG.integers(1, 5)))
        PQ = np.einsum("ay,bz->abyz", P, Q).reshape(P.shape[0] * Q.shape[0], P.shape[1] * Q.shape[1])
        worst = max(worst, abs((1 - eps_star(PQ)) - (1 - eps_star(P)) * (1 - eps_star(Q))))
    check("1 - eps*(P (x) Q) = (1 - eps*(P))(1 - eps*(Q))", worst < 1e-14, f"(worst abs deviation {worst:.1e})")


def n5() -> None:
    print("N5: eps* is not a complete monotone")
    P = np.array([[0.4, 0.2, 0.2, 0.2]]).T          # trivial observation, label law (0.4, 0.2, 0.2, 0.2)
    Q = np.array([[0.35, 0.35, 0.3, 0.0]]).T
    check("eps*(P) < eps*(Q)", eps_star(P) < eps_star(Q), f"({eps_star(P):.2f} < {eps_star(Q):.2f})")
    check("but Q is not reachable from P (majorization fails: 0.7 > 0.6)", not reachable_lp(P, Q))
    val, c = dual_certificate(P, Q)
    check("and the dual LP produces a separating loss", val > 1e-6, f"(violation {val:.3f})")


def n6() -> None:
    print("N6: free objects")
    ok_reach, ok_faith = True, True
    for _ in range(40):
        lam_ = int(RNG.integers(2, 5))
        P = random_object(lam_, int(RNG.integers(1, 4)))
        ny = int(RNG.integers(1, 4))
        Y = RNG.random(ny)
        Y /= Y.sum()
        U = np.outer(np.ones(lam_) / lam_, Y)             # free object: uniform label, independent of the observation
        ok_reach &= reachable_lp(P, U)
        ok_faith &= abs(eps_star(U) - (1 - 1 / lam_)) < 1e-14 and eps_star(P) < 1 - 1 / lam_ - 1e-12
    check("every object reaches every free object (golden rule)", ok_reach)
    check("eps* equals 1 - 1/|Lambda| on free objects and is strictly smaller on non-free objects", ok_faith)


def Bpar(p: float, e: float) -> float:
    return e + (1 - e) * (2 * np.sqrt(p * (1 - p) / 3) + 2 * p / 3)


def Qhash(p: float, e: float) -> float:
    H4 = -(1 - p) * np.log2(1 - p) - (p * np.log2(p / 3) if p > 0 else 0.0)
    return 1 - 2 * e - (1 - e) * H4


def n7(npairs: int = 20000) -> None:
    print("N7: reference monotones B and Q_hash along the local free order")
    bad_B = bad_Q = 0
    n_reach = 0
    up_B = down_B = up_Q = down_Q = 0
    for _ in range(npairs):
        p, pp = RNG.uniform(0, 0.749, 2)
        e, ep = RNG.uniform(0, 0.99, 2)
        pred, slack = cond_ii(p, e, pp, ep)
        dB = Bpar(pp, ep) - Bpar(p, e)
        dQ = Qhash(pp, ep) - Qhash(p, e)
        if pred:
            n_reach += 1
            bad_B += dB < -1e-12
            bad_Q += dQ > 1e-12
        else:
            up_B += dB > 0
            down_B += dB < 0
            up_Q += dQ > 0
            down_Q += dQ < 0
    check("B nondecreasing on all reachable pairs", bad_B == 0, f"({n_reach} reachable pairs)")
    check("Q_hash nonincreasing on all reachable pairs (numerical)", bad_Q == 0)
    check("on unreachable pairs both directions occur (B and Q_hash are not trivially ordered)",
          min(up_B, down_B, up_Q, down_Q) > 0, f"(B: {up_B} up / {down_B} down; Q: {up_Q} up / {down_Q} down)")


if __name__ == "__main__":
    n1()
    n2_n3()
    n4()
    n5()
    n6()
    n7()
    if FAILED:
        print("\nVIOLATIONS:", ", ".join(FAILED))
        sys.exit(1)
    print("\nALL STRUCTURAL CHECKS PASSED")
