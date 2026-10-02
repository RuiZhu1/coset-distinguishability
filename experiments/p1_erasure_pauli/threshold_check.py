"""Sanity checks of the known code-capacity thresholds (README sec. 2, "sanity checks that must pass first").

Scenarios (plain Monte Carlo near threshold, where p_L is O(0.1-0.5), so stratification is not needed):

  depolarizing  exact ML decoder and MWPM (X/Z independent); optimal threshold ~ 18.9%, MWPM ~ 15%
  bitflip       MWPM on X-only noise; threshold ~ 10.3%
  erasure       exact ML decoder on pure erasure (p = 0); threshold = 50%

The crossing of consecutive distances is the root of a weighted straight-line fit to the difference of the two curves
(lcd.analysis.crossing).  Small-d crossings drift slowly towards the thermodynamic value: this is a sanity check,
not a threshold determination.

    python experiments/p1_erasure_pauli/threshold_check.py --scenario depolarizing
    python experiments/p1_erasure_pauli/threshold_check.py --scenario depolarizing --reanalyze   # CSV only

Writes results/threshold_check_<scenario>.{csv,json}.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np

import lcd
from lcd.analysis import crossing
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MWPMCodeCapacity
from lcd.noise import sample_bitflip, sample_iid, sample_iid_erasure

REPO = Path(__file__).resolve().parents[2]

# scenario -> {decoder: parameter grid}; the swept parameter is p (depolarizing, bitflip) or e (erasure)
SCENARIOS = {
    "depolarizing": {"ML": [0.16, 0.17, 0.18, 0.19, 0.20, 0.21], "MWPM": [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]},
    "bitflip": {"MWPM": [0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12]},
    "erasure": {"ML": [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56]},
}
REFERENCE = {"depolarizing": "ML ~0.189, MWPM ~0.15", "bitflip": "MWPM ~0.103", "erasure": "ML = 0.5"}


def run_point(scenario, name, dec, code, x, shots, rng):
    if scenario == "depolarizing":
        ex, ez = sample_iid(code.n, x, shots, rng)
        return dec.fail(ex, ez, x, None, rng) if name == "ML" else dec.fail(ex, ez)
    if scenario == "bitflip":
        ex, ez = sample_bitflip(code.n, x, shots, rng)
        return dec.fail(ex, ez)
    ex, ez, er = sample_iid_erasure(code.n, 0.0, x, shots, rng)
    return dec.fail(ex, ez, 0.0, er, rng)


def summarize(rows, ds, grids):
    summary = {}
    for name, xs in grids.items():
        for d1, d2 in zip(ds[:-1], ds[1:]):
            get = lambda d, key: [r[key] for x in xs for r in rows if r["decoder"] == name and r["d"] == d
                                  and abs(r["x"] - x) < 1e-9]
            c, e = crossing(xs, get(d1, "p_L"), get(d2, "p_L"), get(d1, "se"), get(d2, "se"))
            summary[f"{name}_d{d1}_d{d2}"] = dict(crossing=c, se=e)
            print(f"crossing {name} d={d1} vs d={d2}: {c:.4f} +- {e:.4f}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="depolarizing")
    ap.add_argument("--ds", type=int, nargs="+", default=[5, 7, 9])
    ap.add_argument("--shots", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20241003)
    ap.add_argument("--reanalyze", action="store_true", help="recompute the crossings from the existing CSV")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or REPO / "results" / f"threshold_check_{args.scenario}"
    grids = SCENARIOS[args.scenario]

    rows = []
    if args.reanalyze:
        with open(out.with_suffix(".csv")) as fh:
            rows = [dict(decoder=r["decoder"], d=int(r["d"]), x=float(r["x"]), p_L=float(r["p_L"]),
                         se=float(r["se"]), shots=int(r["shots"])) for r in csv.DictReader(fh)]
    else:
        for name, xs in grids.items():
            for d in args.ds:
                code = RotatedSurfaceCode(d)
                dec = ExactTNMLDecoder(code) if name == "ML" else MWPMCodeCapacity(code)
                for x in xs:
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [args.seed, list(SCENARIOS).index(args.scenario), d, int(round(x * 1e4)), name == "ML"]))
                    f = run_point(args.scenario, name, dec, code, x, args.shots, rng)
                    pl = float(f.mean())
                    se = float(np.sqrt(pl * (1 - pl) / args.shots))
                    rows.append(dict(decoder=name, d=d, x=x, p_L=pl, se=se, shots=args.shots))
                    print(f"{name:4s} d={d} {'e' if args.scenario == 'erasure' else 'p'}={x:.3f}: "
                          f"p_L={pl:.4f} +- {se:.4f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out.with_suffix(".csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    summary = summarize(rows, args.ds, grids)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        dirty = True
    with open(out.with_suffix(".json"), "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/threshold_check.py", scenario=args.scenario,
                       swept="e (p = 0)" if args.scenario == "erasure" else "p",
                       reference=REFERENCE[args.scenario], args={k: str(v) for k, v in vars(args).items()},
                       crossings=summary, versions=dict(lcd=lcd.__version__, numpy=np.__version__),
                       git_commit=commit, git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       note="finite-size crossings of small d from a weighted linear fit of the difference; "
                            "sanity check only (no finite-size scaling)"), fh, indent=2)


if __name__ == "__main__":
    main()
