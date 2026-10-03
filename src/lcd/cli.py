"""Command line: ``lcd-peierls`` computes the circuit-level Peierls bounds (theory, Theorems 4.28 and 4.32).

    lcd-peierls circuit.stim                    # a stim circuit (its DEM is built as sinter builds it)
    lcd-peierls model.dem                       # a decomposed detector error model
    lcd-peierls --generated surface_code:rotated_memory_z -d 5 -p 1e-3
    lcd-peierls a.stim b.stim --json            # one JSON object per line

The number printed (by default the smaller of Theorems 4.28 and 4.32) is a rigorous upper bound, per shot, on the
logical failure probability of pymatching (no correlated decoding) on that model, hence of the maximum-likelihood decoder. ``inf`` means the walk sum diverges.
"""
from __future__ import annotations

import argparse
import json
import sys

import stim

from lcd.analysis.circuit_peierls import peierls_bound
from lcd.analysis.circuit_peierls_geodesic import best_bound, geodesic_bound


def _dem_from_path(path: str) -> stim.DetectorErrorModel:
    if path.endswith(".dem"):
        return stim.DetectorErrorModel.from_file(path)
    return stim.Circuit.from_file(path).detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


def _generated(code: str, d: int, rounds: int | None, p: float) -> stim.DetectorErrorModel:
    c = stim.Circuit.generated(code, distance=d, rounds=rounds or d, after_clifford_depolarization=p,
                               after_reset_flip_probability=p, before_measure_flip_probability=p,
                               before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lcd-peierls", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help=".stim circuits or .dem detector error models")
    ap.add_argument("--generated", metavar="CODE", help="a stim generated code task, e.g. surface_code:rotated_memory_z")
    ap.add_argument("-d", type=int, help="distance for --generated")
    ap.add_argument("-r", "--rounds", type=int, help="rounds for --generated (default: d)")
    ap.add_argument("-p", type=float, help="uniform circuit noise strength for --generated")
    ap.add_argument("--lam", type=float, help="evaluate at this lambda instead of minimizing over it")
    ap.add_argument("--method", choices=("best", "4.28", "4.32"), default="best",
                    help="Theorem 4.28, its geodesic refinement 4.32 (exclusive mechanisms only), or the smaller (default)")
    ap.add_argument("--json", action="store_true", help="print one JSON object per model")
    a = ap.parse_args(argv)

    models = [(p, _dem_from_path(p)) for p in a.paths]
    if a.generated:
        if a.d is None or a.p is None:
            ap.error("--generated needs -d and -p")
        models.append((f"{a.generated} d={a.d} r={a.rounds or a.d} p={a.p:g}", _generated(a.generated, a.d, a.rounds, a.p)))
    if not models:
        ap.error("give a .stim/.dem path or --generated")

    status = 0
    for name, dem in models:
        try:
            lams = None if a.lam is None else [a.lam]
            if a.method == "4.28":
                r = dict(peierls_bound(dem, lams=lams), method="4.28")
            elif a.method == "4.32":
                r = dict(geodesic_bound(dem, lams=lams), method="4.32", delta=0.0)
            else:
                r = best_bound(dem) if lams is None else dict(peierls_bound(dem, lams=lams), method="4.28")
        except (ValueError, NotImplementedError) as ex:
            print(f"{name}: not covered: {ex}", file=sys.stderr)
            status = 1
            continue
        if a.json:
            print(json.dumps(dict(model=name, **{k: r.get(k) for k in ("bound", "method", "lam", "rho_upper", "delta")},
                                  stats=r["stats"]), default=float))
        else:
            print(f"{name}: p_L(MWPM) <= {r['bound']:.3e}  (Theorem {r['method']}, lam = {r['lam']:.3f}, "
                  f"rho <= {r['rho_upper']:.4f})")
    return status


if __name__ == "__main__":
    sys.exit(main())
