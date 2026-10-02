"""Genie lower bound (Proposition 5.1) beyond d = 3 (theory/sec6: "recheck with exact ML for d >= 5").

(a) Step 1 of the proof, exactly over GF(2): for every interior column T of the rotated surface code (d = 3..21), the Paulis supported
    on T with zero syndrome are exactly {I, Xbar_T}, and Xbar_T flips the logical observable (it is not a stabilizer).
(b) eps_genie(d) in closed form.  On an unflagged qubit, a in {Z, Y} has P = Q = p/3 (Z + X = Y), and a in {I, X} swaps
    (1 - p) <-> p/3 under the shift; flagged qubits have P = Q.  Hence
        eps_genie = 1/2 sum_u C(d,u) (1-e)^u e^(d-u) sum_{i+x+r=u} u!/(i! x! r!) 2^r (p/3)^r min((1-p)^i (p/3)^x, (1-p)^x (p/3)^i),
    cross-checked against brute-force enumeration at d = 3, 5.
(c) eps*_d >= eps_genie(d) for the ML data of P1 (results/p1_exchange_rate.json, d = 5..13: the certified upper limit of the
    stratified estimate must not lie below eps_genie) and of the certificate check (results/cc_certificate.json, d = 5).
    The bound holds for any decoder, so this is a falsification test of the proposition, not of the decoder.

    python theory/checks/genie_large_d.py          (seconds)
Exit status non-zero if a check fails.
"""
from __future__ import annotations

import itertools
import json
import sys
from math import comb, factorial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.codes import RotatedSurfaceCode  # noqa: E402

failures: list[str] = []


def gf2_rank(M: np.ndarray) -> int:
    M = M.copy() % 2
    r = 0
    for c in range(M.shape[1]):
        k = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if k is None:
            continue
        M[[r, k]] = M[[k, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def step1(d: int) -> None:
    code = RotatedSurfaceCode(d)
    for c0 in range(1, d - 1):
        T = [r * d + c0 for r in range(d)]
        kx = len(T) - gf2_rank(code.HZ[:, T])      # dim of X parts on T with zero syndrome
        kz = len(T) - gf2_rank(code.HX[:, T])      # dim of Z parts on T with zero syndrome
        ones_ok = not (code.HZ[:, T].sum(axis=1) % 2).any()
        flips = int(code.LZ[T].sum()) % 2 == 1      # Xbar_T anticommutes with the Z logical (row 0)
        if not (kx == 1 and kz == 0 and ones_ok and flips):
            failures.append(f"Step 1 fails at d = {d}, column {c0}: dim X = {kx}, dim Z = {kz}, ones {ones_ok}, flips {flips}")


def genie_closed(d: int, p: float, e: float) -> float:
    a, b = 1 - p, p / 3
    tot = 0.0
    for u in range(d + 1):
        s = 0.0
        for i in range(u + 1):
            for x in range(u - i + 1):
                r = u - i - x
                m = factorial(u) // (factorial(i) * factorial(x) * factorial(r))
                s += m * 2 ** r * b ** r * min(a ** i * b ** x, a ** x * b ** i)
        tot += comb(d, u) * (1 - e) ** u * e ** (d - u) * s
    return 0.5 * tot


def genie_brute(d: int, p: float, e: float) -> float:
    q = [1 - p, p / 3, p / 3, p / 3]                 # 0=I 1=X 2=Z 3=Y; adding X flips bit 0
    tot = 0.0
    for u in range(d + 1):
        s = 0.0
        for a in itertools.product(range(4), repeat=u):
            s += min(np.prod([q[t] for t in a]), np.prod([q[t ^ 1] for t in a]))
        tot += comb(d, u) * (1 - e) ** u * e ** (d - u) * 0.5 * s
    return tot


def main() -> None:
    for d in range(3, 22, 2):
        step1(d)
    print("(a) Step 1 (kernel on an interior column = {I, Xbar_T}), d = 3..21, all interior columns:",
          "ok" if not failures else "FAILED")
    for d in (3, 5):
        for p, e in [(0.02, 0.0), (0.06, 0.05), (0.2, 0.3)]:
            g1, g2 = genie_closed(d, p, e), genie_brute(d, p, e)
            if abs(g1 - g2) > 1e-14:
                failures.append(f"closed form {g1} != brute force {g2} at d = {d}, ({p}, {e})")
    print("(b) closed form = brute force at d = 3, 5:", "ok" if not any("closed" in f for f in failures) else "FAILED")
    rows = []
    path = REPO / "results" / "p1_exchange_rate.json"
    if path.exists():
        for pt in json.loads(path.read_text())["points"]:
            for r in pt["per_d"]:
                rows.append(("P1 ML (stratified)", r["d"], pt["p0"], pt["e0"], r["p_L"], r["p_L_cert"][1]))
    path = REPO / "results" / "cc_certificate.json"
    if path.exists():
        for pt in json.loads(path.read_text())["points"]:
            r = pt["eps"]["5"]
            rows.append(("certificate check (stratified)", 5, pt["p"], pt["e"], r["eps"], r["cert_interval"][1]))
            r3 = pt["eps"]["3"]
            rows.append(("certificate check (exact)", 3, pt["p"], pt["e"], r3["eps"], r3["eps"]))
    print("(c) eps*_d >= eps_genie(d):")
    print("    source                           d   p      e      eps*_d      eps_genie   eps*/genie")
    for src, d, p, e, eps, hi in rows:
        g = genie_closed(d, p, e)
        if hi < g * (1 - 1e-12):
            failures.append(f"eps*_{d}({p}, {e}): certified upper limit {hi:.3e} < genie {g:.3e} ({src})")
        print(f"    {src:32s} {d:2d}  {p:.3f}  {e:.3f}  {eps:.3e}   {g:.3e}   {eps / g:7.2f}")
    print("FAILURES:" if failures else "ALL GENIE CHECKS PASSED", *failures, sep="\n  ")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
