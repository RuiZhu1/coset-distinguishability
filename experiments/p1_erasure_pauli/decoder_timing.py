"""Cost per decode of the exact transfer-matrix ML decoder and of the truncated-MPS ML decoder (single core, batched).

Samples come from a stratum typical of what carries p_L at p = 0.04 (w = (d + 1) / 2 + 3 errors, no erasures); the
decoders are matched at p.  Run on an otherwise idle machine.

    python experiments/p1_erasure_pauli/decoder_timing.py
Writes results/decoder_timing.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import lcd
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[7, 9, 11, 13, 15])
    ap.add_argument("--chis", type=int, nargs="+", default=[4, 6, 8, 12])
    ap.add_argument("--seed", type=int, default=20241006)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "decoder_timing.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    for d in args.ds:
        code = RotatedSurfaceCode(d)
        w = (d + 1) // 2 + 3
        N_exact = {7: 1000, 9: 300, 11: 60, 13: 24, 15: 12}.get(d, 24)
        N_mps = 128
        ex, ez, er = sample_stratum(code.n, 0, w, max(N_exact, N_mps), rng)
        pri = make_priors(code.n, 0.04, er)
        exd = ExactTNMLDecoder(code)
        t0 = time.time()
        exd.fail_prob(ex[:N_exact], ez[:N_exact], pri[:N_exact])
        row = dict(d=d, stratum_w=w, exact_ms=1e3 * (time.time() - t0) / N_exact, mps_ms={})
        for chi in args.chis:
            mps = MPSMLDecoder(code, chi=chi)
            mps.fail_prob(ex[:8], ez[:8], pri[:8])                       # warm up
            t0 = time.time()
            mps.fail_prob(ex[:N_mps], ez[:N_mps], pri[:N_mps])
            row["mps_ms"][str(chi)] = 1e3 * (time.time() - t0) / N_mps
        rows.append(row)
        print(f"d={d:2d}: exact {row['exact_ms']:7.2f} ms   MPS " +
              "   ".join(f"chi={c}: {row['mps_ms'][str(c)]:6.2f} ms ({row['exact_ms'] / row['mps_ms'][str(c)]:5.1f}x)"
                          for c in args.chis), flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/decoder_timing.py", rows=rows, seed=args.seed,
                       machine=dict(platform=platform.platform(), python=platform.python_version(),
                                    numpy=np.__version__, cpus=__import__("os").cpu_count()),
                       versions=dict(lcd=lcd.__version__), git_commit=commit, git_dirty_src=dirty,
                       date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       note="single process, single thread, batched; idle machine"), fh, indent=2)


if __name__ == "__main__":
    main()
