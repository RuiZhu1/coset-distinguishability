"""Cross-checks of the exact tensor-network ML decoder (M1 gate).

(A) d = 3 with erasures: stratified estimate with the ML decoder vs the exact Bayes error from exhaustive
    enumeration (theory/checks/exact_small_codes.py).  Also exercises the 2D (k, w) strata with a real decoder.
(B) d = 3, 5 without erasures: logical failure rate of qecsim's RotatedPlanarMPSDecoder (an independent
    implementation of the Bravyi-Suchara-Vargo MPS decoder; chi = 16) vs this repository's exact ML decoder.

Both must agree within statistical error.  Requires `pip install qecsim` for (B).

    python experiments/p1_erasure_pauli/crosscheck_ml.py
Writes results/ml_crosscheck.json (configuration, seeds, versions, git commit and all numbers).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import lcd
from lcd.analysis import Strata, StratifiedEstimator
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder
from lcd.noise import sample_iid, sample_stratum

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "theory" / "checks"))


def check_A(seed: int, p: float, e: float, budget: int) -> dict:
    from exact_small_codes import Code, Eps, rotated_surface_d3
    exact = Eps(Code("rotated d=3", rotated_surface_d3()))(p, e)
    code = RotatedSurfaceCode(3)
    dec = ExactTNMLDecoder(code)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))

    def sampler(k, w, N, rng_):
        ex, ez, er = sample_stratum(code.n, k, w, N, rng_)
        return dec.fail(ex, ez, p, er, rng_)

    S = Strata(code.n, 3, targets=[(p, e)])
    est = StratifiedEstimator(S, sampler)
    est.run(budget, rng, n0=2000)
    r = est.estimate(p, e)
    z = (r.p_L - exact) / r.se
    return dict(name="A: d=3 stratified ML with erasures vs exact enumeration", p=p, e=e, exact=exact,
                estimate=r.p_L, se=r.se, z=z, certified=[r.lo, r.hi], covered=bool(r.lo <= exact <= r.hi),
                strata=int(S.size), decodes=r.decodes, passed=bool(abs(z) < 4 and r.lo <= exact <= r.hi))


def qecsim_rate(d: int, p: float, shots: int, chi: int, seed: int) -> tuple[float, float]:
    from qecsim import paulitools as pt
    from qecsim.models.generic import DepolarizingErrorModel
    from qecsim.models.rotatedplanar import RotatedPlanarCode, RotatedPlanarMPSDecoder
    code, em, dec = RotatedPlanarCode(d, d), DepolarizingErrorModel(), RotatedPlanarMPSDecoder(chi=chi)
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(shots):
        err = em.generate(code, p, rng)
        rec = dec.decode(code, pt.bsp(err, code.stabilizers.T), error_probability=p)
        fails += bool(np.any(pt.bsp(err ^ rec, code.logicals.T)))
    return fails / shots, float(np.sqrt(fails / shots * (1 - fails / shots) / shots))


def check_B(seed: int, d: int, p: float, shots_qecsim: int, shots_ours: int, chi: int) -> dict:
    code = RotatedSurfaceCode(d)
    dec = ExactTNMLDecoder(code)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2, d]))
    ex, ez = sample_iid(code.n, p, shots_ours, rng)
    f = dec.fail(ex, ez, p, None, rng)
    ours, ours_se = float(f.mean()), float(np.sqrt(f.mean() * (1 - f.mean()) / shots_ours))
    t = time.time()
    theirs, theirs_se = qecsim_rate(d, p, shots_qecsim, chi, int(np.random.SeedSequence([seed, 3, d]).generate_state(1)[0]))
    secs = time.time() - t
    z = (ours - theirs) / np.hypot(ours_se, theirs_se)
    return dict(name=f"B: d={d} qecsim MPS (chi={chi}) vs exact ML", d=d, p=p, ours=ours, ours_se=ours_se,
                qecsim=theirs, qecsim_se=theirs_se, z=float(z), shots_ours=shots_ours, shots_qecsim=shots_qecsim,
                qecsim_seconds=round(secs, 1), passed=bool(abs(z) < 4))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                            cwd=REPO, text=True).strip())
    except Exception:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20241002)
    ap.add_argument("--skip-qecsim", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "ml_crosscheck.json")
    args = ap.parse_args()
    results = []
    for p, e in ((0.05, 0.10), (0.15, 0.30)):
        results.append(check_A(args.seed, p, e, budget=300_000))
        print(results[-1], flush=True)
    versions = dict(lcd=lcd.__version__, numpy=np.__version__)
    if not args.skip_qecsim:
        import qecsim
        versions["qecsim"] = qecsim.__version__
        for d, sq, so in ((3, 20000, 200000), (5, 6000, 200000)):
            results.append(check_B(args.seed, d, 0.10, sq, so, chi=16))
            print(results[-1], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/crosscheck_ml.py", seed=args.seed, versions=versions,
                       git_commit=git_commit(), git_dirty_src=git_dirty(), date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       results=results, all_passed=all(r["passed"] for r in results)), fh, indent=2)
    print("ALL PASSED" if all(r["passed"] for r in results) else "SOME FAILED")


if __name__ == "__main__":
    main()
