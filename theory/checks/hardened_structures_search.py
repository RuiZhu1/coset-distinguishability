"""Search for erasure-hardened structures and checks of the obstruction lemmas (theory section 3.6, Lemmas 3.19-3.21).

Background.  A structure is a nested pair S < N of subspaces of F_2^{2n}; eps*_A(p) is the ML failure probability with erased set A.
The single-loss form of Conjecture 3.11 would follow from "erasure-hardened" structures with eps*_A ~ p for all |A| < k, for every k.

  S1  Lemma 3.19 (erasure jump).  If G in N \\ S is supported on T, then eps*_A >= 1/2 for every A containing T.  Checked on every
      structure on n <= 3 qubits (exhaustive) at several p.
  S2  Lemma 3.20 (erasure-only failure).  eps*_A(p = 0) = 1 - 2^{-r(A)} with r(A) = dim(N cap V_A) - dim(S cap V_A) (V_A = Paulis supported
      on A).  Checked exhaustively for n <= 3.
  S3  Lemma 3.21 (first-order failure needs a short logical).  If eps*(p, 0) / p does not tend to 0 as p -> 0 then the minimum weight d_K of
      N \\ S is at most 2.  Checked exhaustively for n <= 3 (eps*/p at p = 1e-4 against d_K).
  S4  Smallest marginal rate R/c at p -> 0 (p0 = 1e-4), exhaustively for n <= 3: is it still 2 e0 / (1 + e0)?
  S5  Randomized hill-climbing for the smallest marginal rate R/c at n = 6 (heuristic).

    python theory/checks/hardened_structures_search.py [--budget SECONDS] [--out results/hardened_structures_search.json]
About 10 minutes on one core.  Exit status is non-zero if S1-S3 fail.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structures import Structure, all_chains, eps_from_table, eps_table, random_chain, rref  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILED.append(name)


def min_logical(st: Structure):
    """(weight, support) of a minimum-weight element of N \\ S, or None if N = S."""
    in_N = st.syn == 0
    zero_cls = st.cls[0]
    logical = in_N & (st.cls != zero_cls)
    if not logical.any():
        return None
    wts = st.nz.sum(axis=1)
    i = int(np.argmin(np.where(logical, wts, 99)))
    return int(wts[i]), tuple(np.flatnonzero(st.nz[i]))


def r_of_A(st: Structure, A: tuple[int, ...]) -> int:
    """dim(N cap V_A) - dim(S cap V_A) from the sizes of the two subgroups of Paulis supported on A."""
    supp_in_A = np.ones(len(st.cls), bool)
    for j in range(st.n):
        if j not in A:
            supp_in_A &= st.nz[:, j] == 0
    nN = int((supp_in_A & (st.syn == 0)).sum())
    nS = int((supp_in_A & (st.cls == st.cls[0])).sum())
    return int(round(np.log2(nN / nS)))


def exhaustive(n: int, p_list, p_small: float = 1e-4, e_list=(0.05, 0.2), p0r: float = 1e-4):
    jump_viol = rep_viol = first_viol = 0
    n_struct = n_first = 0
    minR = {e: 9.0 for e in e_list}
    worst_jump = 1.0
    for S, N in all_chains(n):
        st = Structure(n, S, N)
        n_struct += 1
        ml = min_logical(st)
        # S1: erasure jump at A = supp(G) for the minimum-weight logical (monotonicity in A gives all supersets)
        if ml is not None:
            w, T = ml
            for p in p_list:
                v = st.eps_A(p, T)
                worst_jump = min(worst_jump, v)
                jump_viol += v < 0.5 - 1e-12
        # S2: erasure-only failure
        for a in range(n + 1):
            for A in itertools.combinations(range(n), a):
                rep_viol += abs(st.eps_A(0.0, A) - (1 - 2.0 ** (-r_of_A(st, A)))) > 1e-12
        # S3: first-order failure needs weight <= 2
        e0 = st.eps_A(p_small, ())
        if e0 / p_small > 0.01:
            n_first += 1
            first_viol += (ml is None) or ml[0] > 2
        # S4: marginal rate at p -> 0
        if ml is not None:
            for e0r in e_list:
                c = (0.75 - p0r) / (1 - e0r)
                G, sizes = eps_table(st, [p0r - 1e-7, p0r, p0r + 1e-7])
                dp = (eps_from_table(G[2], sizes, n, e0r) - eps_from_table(G[0], sizes, n, e0r)) / 2e-7
                a_ = sizes
                t1 = np.where(a_ > 0, a_ * e0r ** np.maximum(a_ - 1, 0), 0.0) * (1 - e0r) ** (n - a_)
                t2 = np.where(n - a_ > 0, (n - a_) * e0r ** a_ * (1 - e0r) ** np.maximum(n - a_ - 1, 0), 0.0)
                de = float(((t1 - t2) * G[1]).sum())
                if dp > 1e-9 and de > -1e-12:
                    minR[e0r] = min(minR[e0r], de / dp / c)
    return dict(structures=n_struct, jump_violations=jump_viol, worst_jump_value=worst_jump, replace_violations=rep_viol,
                first_order_structures=n_first, first_order_violations=first_viol, min_R_over_c_p_to_0=minR)


def s5(n: int, p0: float, e0: float, budget: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    c = (0.75 - p0) / (1 - e0)
    t0 = time.time()
    best, evals = 9.0, 0

    def score(B, s, m):
        st = Structure(n, B[:s], B[:m])
        G, sizes = eps_table(st, [p0 - 1e-6, p0, p0 + 1e-6])
        dp = (eps_from_table(G[2], sizes, n, e0) - eps_from_table(G[0], sizes, n, e0)) / 2e-6
        a = sizes
        t1 = np.where(a > 0, a * e0 ** np.maximum(a - 1, 0), 0.0) * (1 - e0) ** (n - a)
        t2 = np.where(n - a > 0, (n - a) * e0 ** a * (1 - e0) ** np.maximum(n - a - 1, 0), 0.0)
        de = float(((t1 - t2) * G[1]).sum())
        return de / dp / c if dp > 1e-9 and de > -1e-12 else 9.0

    while time.time() - t0 < budget:
        S, N, B = random_chain(n, rng)
        s, m = len(S), len(N)
        cur, stale = score(B, s, m), 0
        evals += 1
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
            evals += 1
            if new <= cur + 1e-12:
                stale = 0 if new < cur - 1e-9 else stale + 1
                cur, B, s, m = new, B2, s2, m2
            else:
                stale += 1
        best = min(best, cur)
    return dict(n=n, p0=p0, e0=e0, best_R_over_c=best, two_e_over_one_plus_e=2 * e0 / (1 + e0), evaluations=evals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "hardened_structures_search.json")
    args = ap.parse_args()
    out = {}
    for n in (1, 2, 3):
        t0 = time.time()
        print(f"n = {n}: exhaustive over all linear structures", flush=True)
        r = exhaustive(n, p_list=(0.0213, 0.2, 0.7))
        out[f"exhaustive_n{n}"] = r
        print(f"  {r['structures']} structures ({time.time() - t0:.0f} s)")
        check(f"S1 erasure jump (n={n}): eps*_A >= 1/2 on the support of a minimum-weight logical", r["jump_violations"] == 0,
              f"(smallest value {r['worst_jump_value']:.4f})")
        check(f"S2 erasure-only failure = 1 - 2^-r(A) (n={n})", r["replace_violations"] == 0)
        check(f"S3 first-order failure => d_K <= 2 (n={n})", r["first_order_violations"] == 0,
              f"({r['first_order_structures']} structures fail at first order in p)")
        print("  S4 smallest R/c at p0 = 1e-4: " + ", ".join(f"e0={e}: {v:.5f} (2e/(1+e) = {2 * e / (1 + e):.5f})" for e, v in r["min_R_over_c_p_to_0"].items()), flush=True)
    print(f"S5: hill-climbing for the smallest R/c at n = 6 ({args.budget:.0f} s per point)", flush=True)
    out["s5_n6"] = []
    for (p0, e0) in [(0.0213, 0.05), (0.0213, 0.2)]:
        r = s5(6, p0, e0, args.budget, 61)
        out["s5_n6"].append(r)
        print(f"  p0={p0}, e0={e0}: best R/c = {r['best_R_over_c']:.5f}  (2e0/(1+e0) = {r['two_e_over_one_plus_e']:.5f}; {r['evaluations']} evaluations)", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "theory/checks", "src"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    out.update(script="theory/checks/hardened_structures_search.py", budget_seconds=args.budget, git_commit=commit, git_dirty_src=dirty,
               numpy=np.__version__, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    if FAILED:
        print("VIOLATIONS:", ", ".join(FAILED))
        return 1
    print("ALL LEMMA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
