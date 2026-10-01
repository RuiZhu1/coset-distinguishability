"""P1 sampling benchmark: cost of weight-stratified estimation of the MWPM logical failure rate.

Code capacity, depolarizing noise (e = 0), rotated surface code, MWPM with uniform weights (NOT the ML decoder).
For each (d, p) one stratified run with a fixed decode budget; reports p_L, the Jeffreys relative standard error,
the certified interval, and the decodes needed for 10% relative error (extrapolated by 1/eps^2) versus the
shots a naive Monte Carlo would need for the same relative error.

    python experiments/p1_erasure_pauli/sampling_benchmark.py --budget 1.5e6 --dmax 15

Outputs (config + results together, README sec. 5):
    results/p1_sampling_benchmark.csv    one row per (d, p)
    results/p1_sampling_benchmark.json   configuration, seeds, versions, git commit
    results/p1_sampling_benchmark.npz    per-stratum tables (k, w, N, fails); for MWPM one table serves every p
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pymatching
import scipy

import lcd
from lcd.analysis import Strata, StratifiedEstimator
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import MWPMCodeCapacity
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]


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
    ap.add_argument("--dmin", type=int, default=5)
    ap.add_argument("--dmax", type=int, default=15)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.02, 0.04, 0.06])
    ap.add_argument("--budget", type=float, default=1.5e6)
    ap.add_argument("--n0", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20241001)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "p1_sampling_benchmark")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows, tables = [], {}
    for d in range(args.dmin, args.dmax + 1, 2):
        code = RotatedSurfaceCode(d)
        dec = MWPMCodeCapacity(code)
        sampler = lambda k, w, N, rng, code=code, dec=dec: dec.fail(*sample_stratum(code.n, k, w, N, rng)[:2])
        for p in args.ps:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, d, int(round(p * 1e4))]))
            strata = Strata(code.n, d, targets=[(p, 0.0)])
            est = StratifiedEstimator(strata, sampler)
            t0 = time.time()
            est.run(int(args.budget), rng, n0=args.n0)
            r = est.estimate(p, 0.0, delta=args.delta)
            secs = time.time() - t0
            need_strat = est.decodes_for_rel_error(p, 0.0, 0.1)
            need_naive = (1 - r.p_L) / (0.01 * r.p_L) if r.p_L > 0 else float("inf")
            rows.append(dict(
                d=d, p=p, e=0.0, decoder="MWPM", p_L=r.p_L, rel_se=r.rel_se, cert_lo=r.lo, cert_hi=r.hi,
                decodes=r.decodes, dominant_w=r.dominant[1], omitted_mass=r.omitted_mass,
                decodes_for_10pct_stratified=need_strat, shots_for_10pct_naive=need_naive,
                gain=need_naive / need_strat if need_strat > 0 else float("nan"), seconds=round(secs, 1)))
            tables[f"d{d}_p{p}"] = np.stack([strata.k, strata.w, est.N, est.fails])
            print(f"d={d:2d} p={p:.2f}: p_L={r.p_L:.3e} rel.err={100*r.rel_se:5.1f}%  cert=[{r.lo:.2e},{r.hi:.2e}]  "
                  f"10%: strat {need_strat:.1e} vs naive {need_naive:.1e} decodes  ({secs:.0f}s)", flush=True)

    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    np.savez_compressed(args.out.with_suffix(".npz"), **tables)
    config = dict(
        script="experiments/p1_erasure_pauli/sampling_benchmark.py", args={k: str(v) for k, v in vars(args).items()},
        decoder="MWPM (PyMatching), X/Z decoded independently, uniform edge weights; not ML",
        noise="code-capacity depolarizing, e = 0", estimator="weight-stratified, Neyman allocation, 3 rounds",
        versions=dict(lcd=lcd.__version__, numpy=np.__version__, scipy=scipy.__version__,
                      pymatching=pymatching.__version__),
        git_commit=git_commit(), git_dirty_src=git_dirty(), date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        note="npz arrays are [k, w, N, fails] per stratum; the MWPM table does not depend on p")
    with open(args.out.with_suffix(".json"), "w") as fh:
        json.dump(config, fh, indent=2)


if __name__ == "__main__":
    main()
