"""Envelope identity for the exact ML decoder at d >= 5 (beyond the n <= 9 brute-force check C8).

Claim (theory/sec3, Remark "envelope argument for ML derivatives"): at p* the derivative of the Bayes error equals the
derivative of the failure rate of the decoder *frozen* at p*.  The ML decoder depends on p only (erased positions are
known, so e never enters the posterior), hence only the p-derivative needs this argument.

A finite-difference test must separate two effects, so we compute on *common random samples* of each stratum (k, w):

  A         = sum_s (dP_s/dp)(p*) f_s(p*)                                     analytic derivative, frozen table
  B_frozen  = [sum_s P_s(p*+h) f_s(p*) - sum_s P_s(p*-h) f_s(p*)] / 2h        same finite difference, frozen table
  B_matched = [sum_s P_s(p*+h) f_s(p*+h) - sum_s P_s(p*-h) f_s(p*-h)] / 2h    decoders re-matched at p* +- h

* B_frozen - A is the O(h^2) discretisation error of the central difference (a pure power law p^m has a relative bias
  (m-1)(m-2) h^2 / (6 p^2)); it is NOT an envelope effect.
* B_matched - B_frozen isolates the effect of the decoder changing with p.  The envelope identity says it is
  second order (the matched error differs from the frozen one by O((p-p*)^2), symmetrically), so it must be
  compatible with zero within the bootstrap error and small compared to A.
Pre-registered pass criterion: |B_matched - B_frozen| < 3 bootstrap s.e. and < 1% of A.  A second step 2h shows how
the discretisation error and the matched-minus-frozen difference scale.

    python experiments/p1_erasure_pauli/envelope_check.py
    python experiments/p1_erasure_pauli/envelope_check.py --extended    # d = 9, exact and truncated-MPS (chi = 8) ML
Writes results/envelope_check.json (or results/envelope_check_extended.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import lcd
from lcd.analysis import Strata
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]


def run_case(args: tuple) -> dict:
    d, p0, e0, h, N, seed, decoder = args if len(args) == 7 else (*args, "exact")
    code = RotatedSurfaceCode(d)
    if decoder == "exact":
        dec, kw = ExactTNMLDecoder(code), {}
    else:                                           # "mps<chi>": truncated MPS with the convergence monitor of M2
        chi = int(decoder[3:])
        dec, kw = MPSMLDecoder(code, chi=chi), dict(refine_chi=2 * chi, margin=8.0)
    S = Strata(code.n, d, targets=[(p0, e0)], cutoff=1e-9)   # identity holds for the full sum: keep the tail tiny
    free = np.flatnonzero(~S.known_zero)
    offs = np.array([-2, -1, 0, 1, 2])
    thetas = p0 + h * offs
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, int(p0 * 1e4), int(e0 * 1e4)]))
    f = np.zeros((S.size, N, len(offs)))            # per-sample conditional failure probability at each theta
    for i in free:
        ex, ez, er = sample_stratum(code.n, int(S.k[i]), int(S.w[i]), N, rng)
        for j, p in enumerate(thetas):
            f[i, :, j] = dec.fail_prob(ex, ez, make_priors(code.n, p, er), **kw)
    P = {int(o): S.prob(p0 + h * o, e0) for o in offs}
    dPdp, _ = S.dprob(p0, e0)

    def stats(idx=None):
        ff = f if idx is None else np.take_along_axis(f, idx[:, :, None], axis=1)
        m = ff.mean(axis=1)                          # (strata, 5)
        A = float((dPdp * m[:, 2]).sum())
        out = {"A": A}
        for k in (1, 2):                              # step k*h
            Bf = float(((P[k] - P[-k]) * m[:, 2]).sum()) / (2 * k * h)
            Bm = float((P[k] * m[:, 2 + k]).sum() - (P[-k] * m[:, 2 - k]).sum()) / (2 * k * h)
            out[f"Bf{k}"], out[f"Bm{k}"] = Bf, Bm
        return out

    base = stats()
    brng = np.random.default_rng(seed + 1)
    boots = []
    for _ in range(300):
        idx = brng.integers(0, N, size=(S.size, N))
        boots.append(stats(idx))
    res = dict(d=d, p=p0, e=e0, h=h, decoder=decoder, samples_per_stratum=N, strata=int(free.size), A_analytic_frozen=base["A"])
    passed = True
    for k in (1, 2):
        diff = base[f"Bm{k}"] - base[f"Bf{k}"]
        se = float(np.std([b[f"Bm{k}"] - b[f"Bf{k}"] for b in boots]))
        ok = abs(diff) < 3 * se and abs(diff) < 0.01 * base["A"]
        if k == 1:
            passed = ok
        res[f"step_{k}h"] = dict(
            B_frozen=base[f"Bf{k}"], B_matched=base[f"Bm{k}"],
            discretisation_rel=(base[f"Bf{k}"] - base["A"]) / base["A"],
            matched_minus_frozen=diff, matched_minus_frozen_rel=diff / base["A"], se=se,
            z=diff / se if se > 0 else float("nan"), passed=bool(ok))
    res["passed"] = bool(passed)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20241004)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--extended", action="store_true", help="d = 9 cases, exact and truncated-MPS decoders")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    s = args.seed
    if args.extended:
        cases = [(9, 0.06, 0.0, 0.01, 20_000, s, "exact"), (9, 0.06, 0.05, 0.01, 1_000, s, "exact"),
                 (9, 0.06, 0.0, 0.01, 4_000, s, "mps8")]
    else:
        cases = [(5, 0.06, 0.0, 0.01, 60_000, s), (5, 0.10, 0.0, 0.01, 60_000, s),
                 (5, 0.06, 0.05, 0.01, 6_000, s), (7, 0.06, 0.0, 0.01, 30_000, s)]
    if args.out is None:
        args.out = REPO / "results" / ("envelope_check_extended.json" if args.extended else "envelope_check.json")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_case, cases))
    for r in results:
        print({k: (v if not isinstance(v, dict) else {a: round(b, 6) if isinstance(b, float) else b for a, b in v.items()})
               for k, v in r.items()}, flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/envelope_check.py", seed=args.seed, results=results,
                       criterion="|B_matched - B_frozen| < 3 s.e. and < 1% of A (step h)",
                       versions=dict(lcd=lcd.__version__, numpy=np.__version__), git_commit=commit,
                       git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       all_passed=all(r["passed"] for r in results)), fh, indent=2)
    print("ALL PASSED" if all(r["passed"] for r in results) else "SOME FAILED")


if __name__ == "__main__":
    main()
