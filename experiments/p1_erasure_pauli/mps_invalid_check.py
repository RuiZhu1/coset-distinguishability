"""Are classes with a non-positive truncated partition function irrelevant for the ML decision?

The MPS decoder re-decodes samples with a non-positive class weight at doubled chi.  At d = 13 some classes stay non-positive even at
max_chi = 64 (their true weight is below the truncation noise of the best class), which used to abort production runs.  The decoder now drops
such classes (-inf) and only raises if a sample has no valid class.  Two checks:

 (A) d = 11 (chi = 64 is exact): chi = 4 with repair up to max_chi = 16 (not exact): the EXACT log-weight gap to the best class (nats) of the classes
     that are dropped at max_chi, and agreement of the decisions with the exact decoder.
 (B) d = 13: how often classes are repaired / dropped in the production configuration (chi = 8, max_chi = 64), and agreement of its decisions
     with chi = 16.

    python experiments/p1_erasure_pauli/mps_invalid_check.py
Writes results/mps_invalid_check.json.
"""
from __future__ import annotations

import datetime as dt
import json
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


def check_a(seed: int = 3) -> dict:
    code = RotatedSurfaceCode(11)
    exact = ExactTNMLDecoder(code)
    mps = MPSMLDecoder(code, chi=4, max_chi=16)            # repairs 4 -> 8 -> 16; chi = 64 would be exact at d = 11
    rng = np.random.default_rng(seed)
    gaps, n_samples, n_fallback, n_persistent_samples, n_dec_diff = [], 0, 0, 0, 0
    for (k, w) in [(0, 8), (0, 12), (0, 18), (0, 26), (2, 10), (2, 18), (4, 12), (4, 20), (6, 24)]:
        ex, ez, er = sample_stratum(code.n, k, w, 500, rng)
        pri = make_priors(code.n, 0.04, er)
        Z = exact.class_weights(ex, ez, pri)
        logZ = np.log(Z)
        gap = logZ.max(axis=1, keepdims=True) - logZ
        L = mps.log_class_weights(ex, ez, pri)
        L = mps._repair_invalid(ex, ez, pri, L.copy())
        n_fallback += mps.last_fallback
        dropped = np.isneginf(L)
        gaps.extend(gap[dropped].tolist())
        n_persistent_samples += int(dropped.any(axis=1).sum())
        n_samples += len(ex)
        n_dec_diff += int((exact.fail_prob(ex, ez, pri) != mps.fail_prob(ex, ez, pri)).sum())
    return dict(samples=n_samples, samples_repaired_by_larger_chi=n_fallback, samples_with_classes_dropped_at_max_chi=n_persistent_samples,
                dropped_classes=len(gaps), min_exact_gap_nats=float(min(gaps)) if gaps else None,
                median_exact_gap_nats=float(np.median(gaps)) if gaps else None, decisions_differing_from_exact=n_dec_diff)


def check_b(seed: int = 5) -> dict:
    code = RotatedSurfaceCode(13)
    rng = np.random.default_rng(seed)
    prod, ref = MPSMLDecoder(code, chi=8, max_chi=64), MPSMLDecoder(code, chi=16, max_chi=64)
    n, n_fallback, n_unrep, n_diff = 0, 0, 0, 0
    t0 = time.time()
    for (k, w) in [(0, 12), (0, 20), (1, 24), (2, 30), (0, 40)]:
        ex, ez, er = sample_stratum(code.n, k, w, 200, rng)
        pri = make_priors(code.n, 0.04, er)
        a = prod.fail_prob(ex, ez, pri)
        n_fallback += prod.last_fallback
        n_unrep += prod.last_unrepaired
        b = ref.fail_prob(ex, ez, pri)
        n += len(ex)
        n_diff += int((a != b).sum())
    return dict(samples=n, samples_repaired_by_larger_chi=n_fallback, samples_with_classes_dropped_at_max_chi=n_unrep,
                decisions_differing_between_chi8_and_chi16=n_diff, seconds=time.time() - t0)


def main() -> None:
    a, b = check_a(), check_b()
    print("(A) d = 11:", a)
    print("(B) d = 13:", b)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    (REPO / "results" / "mps_invalid_check.json").write_text(json.dumps(
        dict(script="experiments/p1_erasure_pauli/mps_invalid_check.py", d11=a, d13=b, versions=dict(lcd=lcd.__version__, numpy=np.__version__),
             git_commit=commit, git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), indent=2))


if __name__ == "__main__":
    main()
