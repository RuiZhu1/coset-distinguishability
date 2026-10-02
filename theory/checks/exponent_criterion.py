"""Large-n reversal criterion for trade pairs via channel-coding exponents (theory section 3.6, Remark 3.23).

A structure with S = 0 on n qubits and 2^q labels is a linear code N of size 2^q used on the additive Pauli channel with erasure flags known
to the decoder (n uses; rate R = q ln2 / n nats per qubit), and eps* is its ML (syndrome-decoding) error.  Standard coding theorems then apply:

  random linear codes achieve eps* <= exp(-n E_r(R))  and  exp(-n E_ex(R))        (Gallager; symmetric channel, linear ensemble)
  every code has eps* >= exp(-n (E_sp(R - o(1)) + o(1)))                          (sphere packing, valid for the product channel with flags)

with E_0(rho) = -ln[ e + (1-e) 4^-rho (sum_a P_p(a)^(1/(1+rho)))^(1+rho) ],  E_r(R) = max_{0<=rho<=1}[E_0(rho) - rho R],
E_sp(R) = sup_{rho>=0}[E_0(rho) - rho R],  E_x(rho) = -rho ln[1/4 + (3/4) B^(1/rho)],  E_ex(R) = max_{rho>=1}[E_x(rho) - rho R],
B = e + (1-e)(2 sqrt(p(1-p)/3) + 2p/3).  Hence if  max_R [ max(E_r, E_ex)(R; P') - E_sp(R; P) ] > 0  for a pair (P, P'), then for large n some structure
reverses the pair (eps*(P') < eps*(P)).  This script evaluates that sufficient criterion for the "trade" pairs of universal_order_search.py.
(The criterion concerns S = 0 structures; degenerate structures (S != 0) are not covered by the sphere-packing converse.)

    python theory/checks/exponent_criterion.py
Writes results/exponent_criterion.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PAIRS = {"A": ((0.0213, 0.05), (0.0183, 0.15)), "B": ((0.0213, 0.05), (0.01, 0.25)), "D": ((0.0613, 0.02), (0.05, 0.25)),
         "E": ((0.0213, 0.05), (0.0, 0.25)), "F": ((0.0213, 0.0), (0.0150, 0.30)), "G": ((0.0213, 0.0), (0.0200, 0.10)),
         "C": ((0.0213, 0.0), (0.0100, 0.10))}          # C is reversed on 2 qubits by the structure F2 (mu' < mu, nu' > nu)


def E0(rho, p, e):
    pa = np.array([1 - p, p / 3, p / 3, p / 3])
    return -np.log(e + (1 - e) * 4.0 ** (-rho) * (np.where(pa > 0, pa, 0.0) ** (1.0 / (1.0 + rho))).sum() ** (1.0 + rho))


def B(p, e):
    return e + (1 - e) * (2 * np.sqrt(p * (1 - p) / 3) + 2 * p / 3)


def capacity(p, e):
    pa = np.array([1 - p, p / 3, p / 3, p / 3])
    pa = pa[pa > 0]
    return (1 - e) * (np.log(4) + (pa * np.log(pa)).sum())


RHO_RC = np.linspace(0, 1, 1001)
RHO_SP = np.concatenate([np.linspace(0, 1, 400), np.geomspace(1, 60, 800)])
RHO_EX = np.concatenate([np.linspace(1, 5, 200), np.geomspace(5, 400, 400)])


def Er(R, p, e): return max(E0(r, p, e) - r * R for r in RHO_RC)
def Esp(R, p, e): return max(E0(r, p, e) - r * R for r in RHO_SP)
def Eex(R, p, e):
    b = B(p, e)
    return max(-r * np.log(0.25 + 0.75 * b ** (1.0 / r)) - r * R for r in RHO_EX)


def main() -> None:
    out = {}
    print("pair: max_R [ max(E_r, E_ex)(R; P') - E_sp(R; P) ]   (> 0: large structures reverse the pair);  Bhattacharyya B(P), B(P')")
    for k, (P, Pp) in PAIRS.items():
        Rs = np.linspace(0.005, min(capacity(*P), capacity(*Pp)), 160)[:-1]
        d = np.array([max(Er(R, *Pp), Eex(R, *Pp)) - Esp(R, *P) for R in Rs])
        i = int(d.argmax())
        out[k] = dict(P=P, Pprime=Pp, max_gap=float(d[i]), at_rate=float(Rs[i]), B_P=float(B(*P)), B_Pprime=float(B(*Pp)),
                      capacity_P=float(capacity(*P)), capacity_Pprime=float(capacity(*Pp)))
        print(f"  {k}: {d[i]:+.4f} (at R = {Rs[i]:.3f} nats; capacities {capacity(*P):.3f} / {capacity(*Pp):.3f});  B = {B(*P):.4f} / {B(*Pp):.4f}", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "theory/checks"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    out["meta"] = dict(script="theory/checks/exponent_criterion.py", git_commit=commit, git_dirty_src=dirty, numpy=np.__version__,
                       date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    (REPO / "results" / "exponent_criterion.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
