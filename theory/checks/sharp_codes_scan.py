"""Which codes attain the exchange-rate bound at e0 = 0?  (Observation 3.5, open problem O6; theory/sec6 correspondence table)

At e0 = 0, d eps*/de = sum_j [eps*_j(p) - eps*(p)], where eps*_j is the ML error when qubit j alone carries Dep_{3/4} (uniform law;
flag knowledge is then irrelevant).  As a function of the depolarizing rate q of qubit j alone, eps* is a minimum of linear functions,
hence concave, so eps*_j <= eps* + (3/4 - p) d eps*/dp_j (tangent at q = p, right derivative) -- summing over j gives Theorem 3.2 at e0 = 0.
Equality holds for every j  iff  eps* is linear in q on [p, 3/4] for every j  iff

    (F)  for every qubit j and every syndrome, some class that is ML-optimal at Dep_p on all qubits stays ML-optimal
         when qubit j is replaced by the uniform law ("flagging one qubit never changes the ML decision").

(at a p without ML ties, so that the right derivative is the derivative).  So R^(d) = c  <=>  (F).  This script checks the
equivalence numerically on a library of codes and records which codes attain the bound:
    [[5,1,3]], Steane [[7,1,3]], Shor [[9,1,3]], rotated surface d = 3, and random [[n,1]] stabilizer codes, n = 4..8.

    python theory/checks/sharp_codes_scan.py [--per-n 60]

Writes results/sharp_codes_scan.json; exit status non-zero if R^(d) > c anywhere or if (F) and R^(d) = c disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_small_codes as esc  # noqa: E402
from structures import Structure, rref  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PS = [0.02, 0.06, 0.15]
TOL_EQ = 1e-7          # |R/c - 1| below this counts as attaining the bound (finite-difference p-derivative)
RTOL_TIE = 1e-12


def symp_complement(S: np.ndarray, n: int) -> np.ndarray:
    """Basis of {v : <v, s> = 0 for all s in S} (symplectic form x.z' + z.x')."""
    A = np.concatenate([S[:, n:], S[:, :n]], axis=1) % 2       # <v, s> = A_s . v
    R, piv = rref(A) if len(A) else (np.zeros((0, 2 * n), np.uint8), [])
    free = [c for c in range(2 * n) if c not in piv]
    basis = []
    for f in free:
        v = np.zeros(2 * n, np.uint8)
        v[f] = 1
        for i, pc in enumerate(piv):
            v[pc] = R[i, f]
        basis.append(v)
    return np.array(basis, np.uint8)


def random_isotropic(n: int, dim: int, rng) -> np.ndarray:
    """Random isotropic subspace of F_2^{2n} of the given dimension (greedy: commuting, independent random vectors)."""
    while True:
        rows = []
        for _ in range(50 * dim):
            v = rng.integers(0, 2, 2 * n, dtype=np.uint8)
            if not v.any() or any(esc.symp(v, r, n) for r in rows):
                continue
            if len(rref(np.array(rows + [v]))[1]) == len(rows) + 1:
                rows.append(v)
                if len(rows) == dim:
                    return np.array(rows, np.uint8)
        # restart if stuck


def class_weights(st: Structure, p: float, A: tuple[int, ...]) -> np.ndarray:
    n = st.n
    mask = np.ones(n, dtype=bool)
    mask[list(A)] = False
    k = st.nz[:, mask].sum(axis=1)
    m = (n - len(A)) - k
    P = 0.25 ** len(A) * (1 - p) ** m * (p / 3) ** k
    return np.bincount(st.cls, weights=P, minlength=st.ncls)[st.order]


def class_slopes(st: Structure, p: float) -> np.ndarray:
    """s[j, c] = d W_c / d p_j at Dep_p on every qubit (W_c the weight of class c, in st.order), p_j the rate of qubit j alone."""
    n = st.n
    k = st.nz.sum(axis=1)
    P = (1 - p) ** (n - k) * (p / 3) ** k
    out = np.empty((n, st.ncls))
    for j in range(n):
        g = np.where(st.nz[:, j] == 1, 1.0 / p, -1.0 / (1 - p))           # d log law_j / dq at q = p
        out[j] = np.bincount(st.cls, weights=P * g, minlength=st.ncls)[st.order]
    return out


def segments(starts: np.ndarray, total: int):
    return list(zip(starts, list(starts[1:]) + [total]))


def argmax_set(w: np.ndarray) -> np.ndarray:
    return np.flatnonzero(w >= w.max() * (1 - RTOL_TIE))


def exact_rate(st: Structure, p: float) -> tuple[float, float, float, bool]:
    """(R+, d eps*/dp right derivative, d eps*/de at e = 0, criterion (F*)), all from exact class weights."""
    n = st.n
    W = class_weights(st, p, ())
    sl = class_slopes(st, p)
    Wf = [class_weights(st, p, (j,)) for j in range(n)]
    eps = 1.0 - sum(W[a:b].max() for a, b in segments(st.starts, len(W)))
    de = sum(1.0 - sum(Wj[a:b].max() for a, b in segments(st.starts, len(W))) for Wj in Wf) - n * eps
    dp_plus = 0.0
    F = True
    for a, b in segments(st.starts, len(W)):
        T = a + argmax_set(W[a:b])                                         # ML classes at p
        dp_plus -= sl[:, T].sum(axis=0).max()                              # right derivative of 1 - max over tied classes
        smax = sl[:, T].max(axis=1, keepdims=True)
        tol = 1e-9 * np.abs(sl[:, T]).max() + 1e-300
        C = T[(sl[:, T] >= smax - tol).all(axis=0)]                        # (b): maximal slope in every direction
        ok = [c for c in C if all(Wj[c] >= Wj[a:b].max() * (1 - RTOL_TIE) for Wj in Wf)]   # (c): ML under every single flag
        F &= len(ok) > 0
    return de / dp_plus, dp_plus, de, F


def distance(st: Structure) -> int:
    zero_syn, zero_cls = st.syn[0], st.cls[0]
    sel = (st.syn == zero_syn) & (st.cls != zero_cls)
    return int(st.nz[sel].sum(axis=1).min())


def named_codes():
    out = []
    for name, S in [("[[5,1,3]]", esc.five_qubit()), ("Steane [[7,1,3]]", esc.steane()),
                    ("rotated surface d=3", esc.rotated_surface_d3())]:
        out.append((name, np.array(S, np.uint8)))
    # Shor code: Z-type checks Z1Z2, Z2Z3, ... within blocks; X-type checks X^6 on blocks (12)(23)
    n = 9
    rows = []
    for b in range(3):
        for i in range(2):
            v = np.zeros(2 * n, np.uint8)
            v[n + 3 * b + i] = v[n + 3 * b + i + 1] = 1
            rows.append(v)
    for b in range(2):
        v = np.zeros(2 * n, np.uint8)
        v[3 * b:3 * b + 6] = 1
        rows.append(v)
    out.append(("Shor [[9,1,3]]", np.array(rows, np.uint8)))
    return out


def analyse(name: str, S: np.ndarray, failures: list) -> dict:
    n = S.shape[1] // 2
    N = symp_complement(S, n)
    st = Structure(n, S, N)
    rec = dict(name=name, n=n, d=distance(st), per_p=[])
    for p in PS:
        R, dp, de, F = exact_rate(st, p)
        R_fd = st.rate(p, 0.0)[0]                                          # central finite difference, for comparison
        c = 0.75 - p
        attains = abs(R / c - 1) < TOL_EQ
        W = class_weights(st, p, ())
        ties = any(len(argmax_set(W[a:b])) > 1 for a, b in segments(st.starts, len(W)))
        rec["per_p"].append(dict(p=p, R=R, R_central_fd=R_fd, c=c, ratio=R / c, attains=bool(attains), F_star=bool(F), ml_ties=bool(ties)))
        if R > c * (1 + TOL_EQ):
            failures.append(f"{name}: R = {R} > c = {c} at p = {p}")
        if F != attains:
            failures.append(f"{name}: (F*) = {F} but R/c = {R / c:.10f} at p = {p}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    failures: list[str] = []
    codes = []
    for name, S in named_codes():
        rec = analyse(name, S, failures)
        codes.append(rec)
        print(f"{name:22s} n={rec['n']} d={rec['d']}: R/c = " + ", ".join(f"{r['ratio']:.8f}{'*' if r['F_star'] else ''}" for r in rec["per_p"]))
    summary = {}
    for n in range(4, 9):
        recs = [analyse(f"random n={n} #{i}", random_isotropic(n, n - 1, rng), failures) for i in range(a.per_n)]
        codes += recs
        for r in recs:
            key = f"n={n}, d={r['d']}"
            s = summary.setdefault(key, dict(codes=0, attain_all_p=0, attain_some_p=0, ties=0))
            s["codes"] += 1
            s["attain_all_p"] += all(x["attains"] for x in r["per_p"])
            s["attain_some_p"] += any(x["attains"] for x in r["per_p"])
            s["ties"] += any(x["ml_ties"] for x in r["per_p"])
    print("\nrandom [[n,1]] codes (* in the table above marks (F*)):")
    for key, s in summary.items():
        print(f"  {key:12s}: {s['codes']:3d} codes, attain c at every p: {s['attain_all_p']:3d}, at some p: {s['attain_some_p']:3d}, "
              f"ML ties at some p: {s['ties']}")
    out = dict(script="theory/checks/sharp_codes_scan.py", ps=PS, tol=TOL_EQ, args=vars(a), summary=summary, codes=codes,
               failures=failures)
    (REPO / "results" / "sharp_codes_scan.json").write_text(json.dumps(out, indent=2, default=float))
    print("FAILURES:" if failures else "ALL SHARPNESS CHECKS PASSED ((F*) <=> R+ = c on every code and p; R+ <= c everywhere)",
          *failures, sep="\n  ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
