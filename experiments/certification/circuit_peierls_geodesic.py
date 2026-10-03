"""Geodesic and gap refinements of the circuit-level Peierls bound (theory Theorems 4.32, 4.34) against Theorem 4.28.

For stim's rotated surface-code memory circuit (d rounds, uniform circuit noise of strength p, as in Proposition 4.29):
  * both bounds for d = 3, 5, 7, 9 and p in {5e-4, 1e-3, 2e-3, 3e-3, 5e-3};
  * for each d, the largest p at which each bound is finite (bisection; lam on a grid);
and, if Google's Willow data are present under data/willow/ (see results/README.md), the certifiability margin s* of
both bounds on the d = 7 hardware detector error models: the largest factor s such that the bound is finite when every
mechanism probability is multiplied by s.

Run:  python experiments/certification/circuit_peierls_geodesic.py        (a few minutes; < 2 GB)
Writes results/circuit_peierls_geodesic.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import stim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.circuit_peierls import peierls_bound  # noqa: E402
from lcd.analysis.circuit_peierls_geodesic import gap_bound_nb, geodesic_bound  # noqa: E402

LAMS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
WILLOW = REPO / "data" / "willow" / "google_105Q_surface_code_d3_d5_d7" / "d7_at_q6_7" / "Z" / "r10" / "decoding_results"


def surface(d: int, p: float) -> stim.DetectorErrorModel:
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                               after_reset_flip_probability=p, before_measure_flip_probability=p,
                               before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True)


def old_bound(dem) -> float:
    return peierls_bound(dem)["bound"]


def new_bound(dem) -> float:
    return geodesic_bound(dem, lams=LAMS)["bound"]


def gap_nb(dem) -> float:
    return gap_bound_nb(dem, lams=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7])["bound"]


def largest_finite(f, lo: float, hi: float, iters: int = 12) -> float:
    """Largest x in [lo, hi] with f(x) finite, by bisection (f finite at lo assumed, monotone)."""
    if np.isfinite(f(hi)):
        return hi
    for _ in range(iters):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if np.isfinite(f(mid)) else (lo, mid)
    return lo


def scaled(dem: stim.DetectorErrorModel, s: float) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(re.sub(r"error\(([0-9.e+-]+)\)",
                                          lambda m: "error(%.17g)" % min(0.5, float(m.group(1)) * s), str(dem)))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "results" / "circuit_peierls_geodesic.json"))
    a = ap.parse_args()
    t0 = time.time()
    out: dict = dict(meta=dict(script="experiments/certification/circuit_peierls_geodesic.py", git_commit=git("rev-parse", "HEAD"),
                               git_dirty_src=bool(git("status", "--porcelain", "src", "experiments")),
                               date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               stim=stim.__version__, lams=LAMS))
    rows = []
    for d in (3, 5, 7, 9):
        for p in (5e-4, 1e-3, 2e-3, 3e-3, 5e-3):
            dem = surface(d, p)
            o, n, g = old_bound(dem), new_bound(dem), gap_nb(dem)
            rows.append(dict(d=d, p=p, theorem_4_28=o, theorem_4_32=n, theorem_4_34=g,
                             ratio=o / n if np.isfinite(n) and n > 0 else None))
            print(f"d={d} p={p:.0e}: Theorem 4.28 {o:.3e}   Theorem 4.32 {n:.3e}   Theorem 4.34 {g:.3e}", flush=True)
    out["bounds"] = rows
    pmax = []
    for d in (5, 7, 9):
        po = largest_finite(lambda p: peierls_bound(surface(d, p), lams=[0.5])["bound"], 1e-4, 1e-2)
        pn = largest_finite(lambda p: new_bound(surface(d, p)), 1e-4, 1e-2)
        pg = largest_finite(lambda p: gap_nb(surface(d, p)), 1e-4, 1e-2)
        pmax.append(dict(d=d, theorem_4_28=po, theorem_4_32=pn, theorem_4_34=pg))
        print(f"d={d}: largest p with a finite bound: Theorem 4.28 {po:.3e}, 4.32 {pn:.3e}, 4.34 {pg:.3e}", flush=True)
    out["largest_finite_p"] = pmax
    if WILLOW.exists():
        hw = []
        for prior in ("si1000_prior", "rl_optimized_prior"):
            dem = stim.DetectorErrorModel.from_file(WILLOW / f"correlated_matching_decoder_with_{prior}" / "error_model.dem").flattened()
            so = largest_finite(lambda s: peierls_bound(scaled(dem, s), lams=[0.5])["bound"], 0.05, 4.0)
            sn = largest_finite(lambda s: new_bound(scaled(dem, s)), 0.05, 4.0)
            sg = largest_finite(lambda s: gap_nb(scaled(dem, s)), 0.05, 4.0)
            hw.append(dict(experiment="Willow 105Q d7_at_q6_7 Z r10", prior=prior, bound_4_28=old_bound(dem),
                           bound_4_32=new_bound(dem), bound_4_34=gap_nb(dem), margin_4_28=so, margin_4_32=sn,
                           margin_4_34=sg))
            print(f"Willow d7 {prior}: margin s* Theorem 4.28 {so:.3f}, 4.32 {sn:.3f}, 4.34 {sg:.3f}", flush=True)
        out["willow"] = hw
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done ({out['meta']['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
