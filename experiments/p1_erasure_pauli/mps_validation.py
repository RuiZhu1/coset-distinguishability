"""Validation of the truncated-MPS ML decoder on the strata that actually carry p_L (M1 gate).

For (d, p) the strata with the largest contribution P_p(w) f(w) to p_L (covering 90% of it, at most `--max-strata`) are
taken from the MWPM table of `results/p1_sampling_benchmark.npz`; the decoders are matched at p.

Ambiguous / failing samples are rare there (f ~ 1e-4 .. 1e-2), so a plain comparison on a few thousand samples would
test nothing.  We therefore use a stratified design with a cheap enrichment variable, MWPM failure:
  * `--screen` samples of the stratum are decoded with MWPM only (tens of microseconds each);
  * ALL MWPM failures (subsampled to `--max-fail` if there are more) form group F; a Bernoulli subsample of the MWPM
    successes (about `--n-ok`) forms group O;
  * the truncated MPS (several chi) and the reference decoder are run on F u O only, and group means are combined with
    Horvitz-Thompson weights  (F_total * mean_F + N_O * mean_O) / N_screen.
ML failures that MWPM does not share are covered by group O (unbiased, noisier).

Comparisons on the SAME samples (paired, so the numbers measure truncation, not sampling noise):
  part A (d = 11, 13): truncated MPS with chi in --chis  versus the exact transfer-matrix decoder;
  part B (d = 15):     truncated MPS versus chi_ref = 16; chi_ref is anchored to the exact decoder on --anchor samples.

Reported per (d, p, chi): the number of changed decisions (total and among ML-failing samples), and the relative change of
the p_L contribution of the selected strata, sum_w P_w (f_mps - f_ref)(w) / sum_w P_w f_ref(w), with a standard error.

    python experiments/p1_erasure_pauli/mps_validation.py
Writes results/mps_validation.json.
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
from scipy.stats import binom

import lcd
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder, MWPMCodeCapacity
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]


def dominant_strata(d: int, p: float, cover: float = 0.9, max_strata: int = 4) -> list[int]:
    k, w, N, fails = np.load(REPO / "results" / "p1_sampling_benchmark.npz")[f"d{d}_p{p}"]
    contrib = binom.pmf(w.astype(int), d * d, p) * fails / np.maximum(N, 1)
    chosen, acc = [], 0.0
    for i in np.argsort(-contrib):
        chosen.append(int(w[i]))
        acc += contrib[i]
        if acc >= cover * contrib.sum() or len(chosen) >= max_strata:
            break
    return sorted(chosen)


def enriched_samples(code, w, screen, max_fail, n_ok, rng, chunk=20000):
    """Group F (MWPM failures) and group O (Bernoulli subsample of MWPM successes) of a stratum."""
    mw = MWPMCodeCapacity(code)
    fx, fz, ox, oz = [], [], [], []
    q = min(1.0, n_ok / screen)
    n_fail_total = 0
    for s in range(0, screen, chunk):
        m = min(chunk, screen - s)
        ex, ez, _ = sample_stratum(code.n, 0, w, m, rng)
        bad = mw.fail(ex, ez)
        n_fail_total += int(bad.sum())
        fx.append(ex[bad]); fz.append(ez[bad])
        keep = (~bad) & (rng.random(m) < q)
        ox.append(ex[keep]); oz.append(ez[keep])
    fx, fz, ox, oz = (np.concatenate(a) for a in (fx, fz, ox, oz))
    if len(fx) > max_fail:
        idx = rng.choice(len(fx), max_fail, replace=False)
        fx, fz = fx[idx], fz[idx]
    return fx, fz, ox, oz, n_fail_total


def combine(values_f, values_o, F_total, N_total):
    """Horvitz-Thompson mean over the stratum and its standard error from the two groups."""
    N_o = N_total - F_total
    mf, mo = (values_f.mean() if len(values_f) else 0.0), (values_o.mean() if len(values_o) else 0.0)
    vf = values_f.var(ddof=1) / len(values_f) if len(values_f) > 1 else 0.0
    vo = values_o.var(ddof=1) / len(values_o) if len(values_o) > 1 else 0.0
    mean = (F_total * mf + N_o * mo) / N_total
    se = float(np.sqrt((F_total / N_total) ** 2 * vf + (N_o / N_total) ** 2 * vo))
    return float(mean), se


def run_task(task: dict) -> dict:
    d, p, w, chis, ref, anchor, seed = (task[k] for k in ("d", "p", "w", "chis", "ref", "anchor", "seed"))
    code = RotatedSurfaceCode(d)
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, int(round(p * 1e4)), w]))
    fx, fz, ox, oz, F_total = enriched_samples(code, w, task["screen"], task["max_fail"], task["n_ok"], rng)
    nF, nO = len(fx), len(ox)
    ex, ez = np.concatenate([fx, ox]), np.concatenate([fz, oz])
    pri = make_priors(code.n, p, np.zeros(ex.shape, bool))
    isF = np.arange(len(ex)) < nF
    out = dict(d=d, p=p, w=w, screen=task["screen"], mwpm_failures_total=F_total, group_F=nF, group_O=nO,
               P_w=float(binom.pmf(w, code.n, p)), per_chi={})
    if ref == "exact":
        t0 = time.time()
        f_ref = ExactTNMLDecoder(code).fail_prob(ex, ez, pri)
        out["ref_seconds_per_decode"] = (time.time() - t0) / len(ex)
    else:
        t0 = time.time()
        f_ref = MPSMLDecoder(code, chi=ref).fail_prob(ex, ez, pri)
        out["ref_seconds_per_decode"] = (time.time() - t0) / len(ex)
        a = min(anchor, len(ex))
        sel = np.concatenate([np.arange(min(nF, a // 2)), nF + np.arange(min(nO, a - min(nF, a // 2)))])
        f_ex = ExactTNMLDecoder(code).fail_prob(ex[sel], ez[sel], pri[sel])
        out["anchor"] = dict(n=int(len(sel)), changed=int((f_ex != f_ref[sel]).sum()),
                             ml_failures=int((f_ex > 0).sum()))
    out["f_ref_mean"], out["f_ref_se"] = combine(f_ref[isF], f_ref[~isF], F_total, task["screen"])
    out["ml_failing_samples_tested"] = int((f_ref > 0).sum())
    for chi in chis:
        mps = MPSMLDecoder(code, chi=chi)
        t0 = time.time()
        f = mps.fail_prob(ex, ez, pri)
        sec = (time.time() - t0) / len(ex)
        diff = f - f_ref
        md, se = combine(diff[isF], diff[~isF], F_total, task["screen"])
        out["per_chi"][str(chi)] = dict(
            changed=int((f != f_ref).sum()), changed_among_ml_failures=int(((f != f_ref) & (f_ref > 0)).sum()),
            mean_diff=md, se_diff=se, seconds_per_decode=sec,
            max_truncation_weight=float(mps.last_truncation.max()), invalid_classes=mps.last_invalid)
    return out


def aggregate(results: list[dict]) -> list[dict]:
    rows = []
    for d, p in sorted({(r["d"], r["p"]) for r in results}):
        rs = [r for r in results if r["d"] == d and r["p"] == p]
        base = sum(r["P_w"] * r["f_ref_mean"] for r in rs)
        for chi in sorted(int(c) for c in rs[0]["per_chi"]):
            num = sum(r["P_w"] * r["per_chi"][str(chi)]["mean_diff"] for r in rs)
            se = float(np.sqrt(sum((r["P_w"] * r["per_chi"][str(chi)]["se_diff"]) ** 2 for r in rs)))
            rows.append(dict(
                d=d, p=p, chi=chi, strata=[r["w"] for r in rs],
                tested_samples=sum(r["group_F"] + r["group_O"] for r in rs),
                ml_failing_samples_tested=sum(r["ml_failing_samples_tested"] for r in rs),
                changed_decisions=sum(r["per_chi"][str(chi)]["changed"] for r in rs),
                changed_among_ml_failures=sum(r["per_chi"][str(chi)]["changed_among_ml_failures"] for r in rs),
                rel_change_of_pL_contribution=num / base if base > 0 else float("nan"),
                rel_se=se / base if base > 0 else float("nan"),
                seconds_per_decode=float(np.mean([r["per_chi"][str(chi)]["seconds_per_decode"] for r in rs]))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20241005)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-strata", type=int, default=4)
    ap.add_argument("--screen", type=int, default=400_000)
    ap.add_argument("--max-fail", type=int, default=1200)
    ap.add_argument("--n-ok", type=int, default=1200)
    ap.add_argument("--anchor", type=int, default=120)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "mps_validation.json")
    args = ap.parse_args()
    common = dict(screen=args.screen, max_fail=args.max_fail, n_ok=args.n_ok, seed=args.seed)
    tasks = []
    for d, ps in ((11, (0.04, 0.06)), (13, (0.04, 0.06))):
        for p in ps:
            for w in dominant_strata(d, p, max_strata=args.max_strata):
                tasks.append(dict(d=d, p=p, w=w, chis=[3, 4, 6, 8], ref="exact", anchor=0, **common))
    for w in dominant_strata(15, 0.04, max_strata=args.max_strata):
        tasks.append(dict(d=15, p=0.04, w=w, chis=[3, 4, 6, 8, 12], ref=16, anchor=args.anchor, **common))
    tasks.sort(key=lambda t: -t["d"])                         # slowest first
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_task, tasks))
    rows = aggregate(results)
    for r in rows:
        print(f"d={r['d']} p={r['p']} chi={r['chi']}: tested {r['tested_samples']} (ML-failing {r['ml_failing_samples_tested']}), "
              f"changed {r['changed_decisions']} (among failures {r['changed_among_ml_failures']}), "
              f"rel. change of p_L contribution {100 * r['rel_change_of_pL_contribution']:+.3f}% "
              f"+- {100 * r['rel_se']:.3f}%, {1e3 * r['seconds_per_decode']:.1f} ms/decode", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"],
                                             cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(dict(script="experiments/p1_erasure_pauli/mps_validation.py", args={k: str(v) for k, v in vars(args).items()},
                       summary=rows, tasks=results, versions=dict(lcd=lcd.__version__, numpy=np.__version__),
                       git_commit=commit, git_dirty_src=dirty, wall_seconds=time.time() - t0,
                       date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")), fh, indent=2)


if __name__ == "__main__":
    main()
