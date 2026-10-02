"""Counterexample search for the single-loss form of Conjecture 3.11 (code-universal eps*-order = local free order).

Conjecture (single-loss form).  If (p', e') is not reachable from (p, e) by a local free operation (condition (ii) of Theorem 3.7), then some
linear decoding structure D (on any number n of qubits) has eps*_D(p', e') < eps*_D(p, e), i.e. reverses the pair.

What is proved in the notes (section 0 and below) and checked here:
  * the uncoded qubit has eps* = (3/4)(1 - mu), mu = (1-e) lambda(p); it reverses exactly the pairs with mu' > mu;
    hence the conjecture holds for every unreachable pair with e' < e;
  * the two-qubit structure F2 ("observe E0 + E1, guess E0") has eps* = (3/4)(1 - nu), nu = (1 - e^2) lambda(p);
    it reverses exactly the pairs with nu' > nu.
The remaining unreachable pairs have e' >= e, p' < p, mu' <= mu, nu' <= nu ("trade Pauli noise for erasure").

This script searches for further reversals with exact enumeration:
  S1  closed forms of the uncoded qubit and of F2 against the enumerator;
  S2  EXHAUSTIVE over all linear decoding structures on n <= 3 qubits (all nested pairs S < N of subspaces of F_2^{2n};
      structures with identical eps* tables are merged):  minimal marginal exchange rate R_D / c at several work points;
  S3  EXHAUSTIVE n <= 3, random unreachable pairs: is the set of reversed pairs exactly {mu' > mu} U {nu' > nu}?
  S3b first-order wedge at (p0, e0) = (0.0213, 0.2): pairs just below the free-order boundary that are reversed by Rep_3 but by neither
      the uncoded qubit nor F2 (the reversed set on n <= 3 is therefore larger than {mu' > mu} U {nu' > nu});
  S4  the "trade" pairs A, B, D, E: exhaustive n <= 3 plus randomized hill-climbing for n = 4, 5 (a heuristic, not a proof).
  S2d randomized hill-climbing for the smallest marginal rate R_D / c at n = 4, 5 (a heuristic upper bound on the infimum over structures);
  S2c marginal rate of the Pauli repetition family Rep_k (k copies of the Pauli, label E_0; Rep_1 = uncoded, Rep_2 = F2) and its large-k
      limit R_B / c.

    python theory/checks/universal_order_search.py [--budget SECONDS] [--jobs J] [--out results/universal_order_search.json]
Takes about 10 minutes with 4 cores.  Exit status is non-zero only if S1 fails; S2-S4 are reports.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structures import (Structure, all_chains, eps_from_table, eps_table, random_chain, rref)  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
P_REF = (0.0213, 0.0777)          # reference values used to merge structures with identical eps* tables


def lam(p): return 1 - 4 * p / 3
def mu(p, e): return (1 - e) * lam(p)
def nu(p, e): return (1 - e * e) * lam(p)
def reachable(p, e, pp, ep):
    return (pp >= p) if ep >= e else (mu(pp, ep) <= mu(p, e))


# ------------------------------------------------------------------ structures
def uncoded() -> Structure:
    return Structure(1, np.zeros((0, 2), np.uint8), np.eye(2, dtype=np.uint8))


def f2() -> Structure:
    """Two qubits, N = <XX, ZZ>, S = 0: syndrome = E0 + E1, label = E0."""
    N = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], np.uint8)       # (x0 x1 | z0 z1)
    return Structure(2, np.zeros((0, 4), np.uint8), N)


def rep(k: int) -> Structure:
    """k copies of the Pauli (S = 0, N = <X^k, Z^k>): label E_0, syndrome E_0 + E_i.  rep(1) = uncoded, rep(2) = F2."""
    N = np.zeros((2, 2 * k), np.uint8)
    N[0, :k] = 1
    N[1, k:] = 1
    return Structure(k, np.zeros((0, 2 * k), np.uint8), N)


def _key(args):
    S, N, n = args
    st = Structure(n, S, N)
    G, _ = eps_table(st, P_REF)
    return tuple(np.round(G.ravel(), 12))


def unique_structures(n: int, jobs: int):
    """All linear structures on n qubits, merged by their eps* tables at two reference values of p."""
    chains = list(all_chains(n))
    with ProcessPoolExecutor(jobs) as ex:
        keys = list(ex.map(_key, [(S, N, n) for S, N in chains], chunksize=256))
    seen: dict = {}
    for (S, N), k in zip(chains, keys):
        seen.setdefault(k, (S, N, 0))
        S0, N0, c = seen[k]
        seen[k] = (S0, N0, c + 1)
    return len(chains), [(S, N, c) for (S, N, c) in seen.values()]


# ------------------------------------------------------------------ S1
def s1() -> bool:
    print("S1: closed forms")
    rng = np.random.default_rng(1)
    ok = True
    u, d2 = uncoded(), f2()
    worst = 0.0
    for _ in range(40):
        p, e = rng.uniform(0, 0.74), rng.uniform(0, 0.99)
        worst = max(worst, abs(u.eps(p, e) - 0.75 * (1 - mu(p, e))), abs(d2.eps(p, e) - 0.75 * (1 - nu(p, e))))
    ok = worst < 1e-12
    print(f"  [{'ok' if ok else 'FAIL'}] eps*(uncoded) = (3/4)(1-mu), eps*(F2) = (3/4)(1-nu); worst deviation {worst:.1e}")
    return ok


# ------------------------------------------------------------------ S2
BASES = [(0.0213, 0.0), (0.0213, 0.02), (0.0213, 0.05), (0.0213, 0.1), (0.0613, 0.02), (0.1013, 0.1), (0.0213, 0.2)]


def _rates(args):
    S, N, n = args
    st = Structure(n, S, N)
    out = []
    for p0, e0 in BASES:
        R, dp, de = st.rate(p0, e0)
        c = (0.75 - p0) / (1 - e0)
        out.append(R / c if (np.isfinite(R) and dp > 1e-9 and de > -1e-9) else np.nan)
    return out


def s2(uniq: dict, jobs: int) -> dict:
    print("S2: minimal marginal rate R_D / c over ALL structures on n <= 3 qubits (exhaustive)")
    res: dict[int, dict] = {}
    for n, lst in uniq.items():
        with ProcessPoolExecutor(jobs) as ex:
            rows = np.array(list(ex.map(_rates, [(S, N, n) for S, N, _ in lst], chunksize=16)))
        res[n] = {b: float(rows[:, k][np.isfinite(rows[:, k])].min()) for k, b in enumerate(BASES)}
    for (p0, e0) in BASES:
        line = "  ".join(f"n<={n}: {min(res[m][(p0, e0)] for m in res if m <= n):.5f}" for n in uniq)
        print(f"  p0={p0}, e0={e0}: 2e0/(1+e0) = {2 * e0 / (1 + e0):.5f};  min R/c  {line}")
    return {f"p0={p0},e0={e0}": {f"n<={n}": min(res[m][(p0, e0)] for m in res if m <= n) for n in uniq} for (p0, e0) in BASES}


def s2c() -> dict:
    print("S2c: marginal rate R/c of the Pauli repetition family Rep_k (large-k limit: R_B/c, independent of e0)")

    def beta(p): return 2 * np.sqrt(p * (1 - p) / 3) + 2 * p / 3
    out = {}
    for (p0, e0) in BASES:
        c = (0.75 - p0) / (1 - e0)
        vals = [rep(k).rate(p0, e0)[0] / c for k in range(1, 8)]
        h = 1e-7
        RB = (1 - beta(p0)) / ((1 - e0) * (beta(p0 + h) - beta(p0 - h)) / (2 * h)) / c
        print(f"  p0={p0}, e0={e0}: " + " ".join(f"k={k}:{v:.4f}" for k, v in enumerate(vals, 1)) + f"   R_B/c = {RB:.4f}")
        out[f"p0={p0},e0={e0}"] = dict(rep=vals, RB_over_c=RB)
    return out


# ------------------------------------------------------------------ S3 / S4
def _pair_eval(args):
    S, N, n, pairs = args
    st = Structure(n, S, N)
    ps = sorted({q for P, Pp in pairs for q in (P[0], Pp[0])})
    G, sizes = eps_table(st, ps)
    row = {p: G[i] for i, p in enumerate(ps)}
    return [eps_from_table(row[Pp[0]], sizes, n, Pp[1]) - eps_from_table(row[P[0]], sizes, n, P[1]) for P, Pp in pairs]


def min_diff(uniq_n: list, n: int, pairs: list, jobs: int) -> np.ndarray:
    with ProcessPoolExecutor(jobs) as ex:
        rows = np.array(list(ex.map(_pair_eval, [(S, N, n, pairs) for S, N, _ in uniq_n], chunksize=8)))
    return rows.min(axis=0)             # most negative eps*(P') - eps*(P): < 0 means some structure reverses the pair


def random_unreachable_pairs(k: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < k:
        p, pp = rng.uniform(0.005, 0.12, 2)
        e, ep = rng.uniform(0, 0.6, 2)
        if not reachable(p, e, pp, ep) and abs(mu(pp, ep) - mu(p, e)) > 1e-3 and abs(nu(pp, ep) - nu(p, e)) > 1e-3:
            out.append(((float(p), float(e)), (float(pp), float(ep))))
    return out


def s3(uniq: dict, jobs: int, k: int = 24) -> dict:
    print(f"S3: {k} random unreachable pairs; reversed by some structure on n <= 3  vs  predicted by (mu' > mu) or (nu' > nu)")
    pairs = random_unreachable_pairs(k, 7)
    mind = np.minimum.reduce([min_diff(uniq[n], n, pairs, jobs) for n in uniq])
    rows, mism = [], 0
    for ((p, e), (pp, ep)), d in zip(pairs, mind):
        pred = mu(pp, ep) > mu(p, e) or nu(pp, ep) > nu(p, e)
        rev = d < -1e-12
        mism += pred != rev
        rows.append(dict(P=(p, e), Pprime=(pp, ep), predicted=bool(pred), reversed=bool(rev), min_diff=float(d)))
    n_pred = sum(r["predicted"] for r in rows)
    print(f"  predicted-and-reversed agree on {k - mism}/{k} pairs ({n_pred} predicted reversible, {k - n_pred} predicted not reversible)")
    for r in rows:
        if r["predicted"] != r["reversed"]:
            print("   MISMATCH", r)
    return dict(pairs=rows, mismatches=mism)


def _climb_rate(args):
    n, p0, e0, seed, budget = args
    c = (0.75 - p0) / (1 - e0)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = 9.0

    def score(B, s, m):
        R, dp, de = Structure(n, B[:s], B[:m]).rate(p0, e0)
        return R / c if (np.isfinite(R) and dp > 1e-9 and de > -1e-9) else 9.0

    while time.time() - t0 < budget:
        S, N, B = random_chain(n, rng)
        s, m = len(S), len(N)
        cur, stale = score(B, s, m), 0
        while stale < 60 and time.time() - t0 < budget:
            B2, s2, m2 = B.copy(), s, m
            r = rng.random()
            if r < 0.75:
                B2[int(rng.integers(0, m2))] = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
            elif r < 0.875 and s2 + 1 < m2:
                s2 += 1
            elif m2 > s2 + 1:
                m2 -= 1
            else:
                continue
            if len(rref(B2[:m2])[1]) != m2:
                continue
            new = score(B2, s2, m2)
            if new <= cur + 1e-12:
                stale = 0 if new < cur - 1e-9 else stale + 1
                cur, B, s, m = new, B2, s2, m2
            else:
                stale += 1
        best = min(best, cur)
    return n, p0, e0, best


def s2d(jobs: int, budget: float) -> dict:
    print(f"S2d: hill-climbing for the smallest R/c at n = 4, 5 ({budget:.0f} s per job; heuristic)")
    pts = [(0.0213, 0.05), (0.0213, 0.2)]
    out = {}
    with ProcessPoolExecutor(jobs) as ex:
        for n, p0, e0, b in ex.map(_climb_rate, [(n, p0, e0, 10 * n, budget) for (p0, e0) in pts for n in (4, 5)]):
            print(f"  p0={p0}, e0={e0}, n={n}: best R/c found = {b:.5f}   (2e0/(1+e0) = {2 * e0 / (1 + e0):.5f})", flush=True)
            out[f"p0={p0},e0={e0},n={n}"] = b
    return out


def s3b(uniq: dict, jobs: int) -> dict:
    p0, e0, D = 0.0213, 0.2, 0.01
    c = (0.75 - p0) / (1 - e0)
    print(f"S3b: first-order wedge at ({p0}, {e0}): P' = (p0 - kappa*c*D, e0 + D), D = {D}; Rep_3 has R/c = {rep(3).rate(p0, e0)[0] / c:.4f}, F2 has {2 * e0 / (1 + e0):.4f}")
    kappas = [0.15, 0.22, 0.30, 0.40]
    pairs = [((p0, e0), (p0 - k * c * D, e0 + D)) for k in kappas]
    mind = np.minimum.reduce([min_diff(uniq[n], n, pairs, jobs) for n in uniq])
    out = []
    for k, ((p, e), (pp, ep)), d in zip(kappas, pairs, mind):
        pred = mu(pp, ep) > mu(p, e) or nu(pp, ep) > nu(p, e)
        print(f"  kappa = {k:.2f}: reversed by some structure on n <= 3: {d < -1e-15} (min diff {d:+.2e});  predicted by mu or nu alone: {pred}")
        out.append(dict(kappa=k, reversed=bool(d < -1e-15), min_diff=float(d), predicted_by_mu_nu=bool(pred)))
    return dict(p0=p0, e0=e0, D=D, pairs=out)


TRADE = {   # name: (P, P') ; P' has less Pauli noise and more erasure; mu' < mu and nu' < nu
    "A": ((0.0213, 0.05), (0.0183, 0.15)),
    "B": ((0.0213, 0.05), (0.0100, 0.25)),
    "D": ((0.0613, 0.02), (0.0500, 0.25)),
    "E": ((0.0213, 0.05), (0.0000, 0.25)),
    "F": ((0.0213, 0.00), (0.0150, 0.30)),
    "G": ((0.0213, 0.00), (0.0200, 0.10)),
}


def _climb(args):
    n, key, seed, budget = args
    P, Pp = TRADE[key]
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = (9.0, None)

    def score(B, s, m):
        st = Structure(n, B[:s], B[:m])
        G, sizes = eps_table(st, [P[0], Pp[0]])
        a, b = eps_from_table(G[0], sizes, n, P[1]), eps_from_table(G[1], sizes, n, Pp[1])
        return b / a if a > 1e-14 else 9.0

    while time.time() - t0 < budget:
        S, N, B = random_chain(n, rng)
        s, m = len(S), len(N)
        cur, stale = score(B, s, m), 0
        while stale < 50 and time.time() - t0 < budget:
            B2, s2, m2 = B.copy(), s, m
            r = rng.random()
            if r < 0.75:
                B2[int(rng.integers(0, m2))] = rng.integers(0, 2, size=2 * n, dtype=np.uint8)
            elif r < 0.875 and s2 + 1 < m2:
                s2 += 1
            elif m2 > s2 + 1:
                m2 -= 1
            else:
                continue
            if len(rref(B2[:m2])[1]) != m2:
                continue
            new = score(B2, s2, m2)
            if new <= cur + 1e-12:
                stale = 0 if new < cur - 1e-9 else stale + 1
                cur, B, s, m = new, B2, s2, m2
            else:
                stale += 1
        best = min(best, (cur, (s, m)))
    return n, key, best


def s4(uniq: dict, jobs: int, budget: float) -> dict:
    print("S4: trade pairs (less Pauli noise, more erasure; mu' < mu, nu' < nu); smallest eps*(P')/eps*(P) found (< 1 would be a reversal)")
    out = {}
    pairs = list(TRADE.values())
    exh = [np.minimum.reduce([min_diff(uniq[n], n, pairs, jobs) for n in uniq])]
    for key, d in zip(TRADE, exh[0]):
        P, Pp = TRADE[key]
        print(f"  pair {key}: {P} -> {Pp}:  n <= 3 exhaustive, min eps*(P') - eps*(P) = {d:+.3e} ({'REVERSED' if d < -1e-12 else 'not reversed'})")
        out[key] = dict(P=P, Pprime=Pp, exhaustive_n_le_3_min_diff=float(d))
    jobs_list = [(n, key, 100 * n + i, budget) for key in TRADE for n in (4, 5) for i in range(1)]
    with ProcessPoolExecutor(jobs) as ex:
        for n, key, b in ex.map(_climb, jobs_list):
            print(f"  pair {key}: hill-climb n={n} ({budget:.0f} s): min ratio {b[0]:.4f} at (dimS, dimN) = {b[1]}", flush=True)
            out[key][f"climb_n{n}_min_ratio"] = b[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "universal_order_search.json")
    args = ap.parse_args()
    ok = s1()
    uniq, totals = {}, {}
    for n in (1, 2, 3):
        t0 = time.time()
        tot, lst = unique_structures(n, args.jobs)
        uniq[n], totals[n] = lst, tot
        print(f"n={n}: {tot} structures, {len(lst)} distinct eps* tables ({time.time() - t0:.0f} s)", flush=True)
    r2 = s2(uniq, args.jobs)
    r2c = s2c()
    r2d = s2d(args.jobs, args.budget)
    r3 = s3(uniq, args.jobs)
    r3b = s3b(uniq, args.jobs)
    r4 = s4(uniq, args.jobs, args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "theory/checks", "src"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    json.dump(dict(script="theory/checks/universal_order_search.py", budget_seconds=args.budget, structures_total=totals,
                   distinct_tables={n: len(v) for n, v in uniq.items()}, s2_min_R_over_c=r2, s2c_rep_family=r2c, s2d_hill_climb_rates=r2d, s3=r3, s3b=r3b, s4=r4,
                   numpy=np.__version__, git_commit=commit, git_dirty_src=dirty,
                   date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), open(args.out, "w"), indent=2, default=float)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
