"""Truncated-MPS ML decoder versus the exact decoder on strata WITH erasures (k erased qubits, w Pauli errors).

`mps_validation.py` covers e = 0 with an MWPM-enriched design; MWPM has no erasure support, so here the strata are chosen
with an O(1e-2 .. 1e-1) exact-ML failure probability (probed beforehand: many erasure strata with 2w + k above the
distance still have a failure probability of exactly 0 and would test nothing), and the comparison is a plain paired
one on the same samples.  Decoders are matched at p.

    python experiments/p1_erasure_pauli/mps_erasure_check.py
Writes results/mps_erasure_check.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import lcd
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]


def run_case(case: tuple) -> dict:
    d, k, w, p, N, chis, seed = case
    code = RotatedSurfaceCode(d)
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, k, w]))
    ex, ez, er = sample_stratum(code.n, k, w, N, rng)
    pri = make_priors(code.n, p, er)
    t0 = time.time()
    fe = ExactTNMLDecoder(code).fail_prob(ex, ez, pri)
    out = dict(d=d, k=k, w=w, p=p, N=N, exact_failure_fraction=float(fe.mean()),
               exact_seconds_per_decode=(time.time() - t0) / N, per_chi={})
    for chi in chis:
        mps = MPSMLDecoder(code, chi=chi)
        t0 = time.time()
        f = mps.fail_prob(ex, ez, pri)
        out["per_chi"][str(chi)] = dict(changed=int((f != fe).sum()), ml_failing=int((fe > 0).sum()),
                                        mean_diff=float((f - fe).mean()), seconds_per_decode=(time.time() - t0) / N,
                                        max_truncation_weight=float(mps.last_truncation.max()),
                                        invalid_classes=mps.last_invalid, samples_repaired_by_fallback=mps.last_fallback)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20241007)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "mps_erasure_check.json")
    args = ap.parse_args()
    chis = [4, 6, 8, 12]
    # strata chosen (by probing the exact decoder) so that ML fails with probability ~1e-2 .. 1e-1; the dense ones are
    # the harshest tests of the truncation
    cases = [(11, 6, 12, 0.05, 4000, chis, args.seed), (11, 6, 14, 0.05, 4000, chis, args.seed),
             (11, 8, 11, 0.05, 4000, chis, args.seed), (11, 8, 13, 0.05, 4000, chis, args.seed),
             (11, 10, 10, 0.05, 4000, chis, args.seed), (11, 10, 12, 0.05, 4000, chis, args.seed),
             (13, 4, 16, 0.05, 1500, chis, args.seed), (13, 8, 15, 0.05, 1500, chis, args.seed),
             (13, 6, 18, 0.05, 1500, chis, args.seed), (13, 8, 17, 0.05, 1500, chis, args.seed)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_case, cases))
    for r in results:
        print(f"d={r['d']} (k={r['k']}, w={r['w']}) exact failure fraction {r['exact_failure_fraction']:.4f}: " +
              "  ".join(f"chi={c}: changed {v['changed']}/{r['N']} (repaired {v['samples_repaired_by_fallback']})" for c, v in r["per_chi"].items()), flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/mps_erasure_check.py", seed=args.seed, results=results,
                       versions=dict(lcd=lcd.__version__, numpy=np.__version__), git_commit=commit, git_dirty_src=dirty,
                       date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), fh, indent=2)


if __name__ == "__main__":
    main()
