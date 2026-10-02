"""M2: ML exchange rate between erasure and Pauli noise at code capacity (README P1).

For every work point (p0, e0) and distance d the failure table f_d(k, w) of the Bayes-optimal (ML) decoder, matched at p0, is
estimated with weight-stratified sampling (README "Sampling scheme"):
    d <= 11: ExactTNMLDecoder;   d = 13: MPSMLDecoder(chi = 8) with the convergence monitor (refine_chi = 16, margin = 8).
The allocation is a Neyman allocation for p_L AND for both derivatives dp_L/dp, dp_L/de (derivatives are analytic in the
tables: frozen decoder for p by the envelope argument, exact for e because the ML posterior does not depend on e).
Tables are stored in results/p1_m2/tables/ after every round, so a run can be extended (--scale) or resumed.

    python experiments/p1_erasure_pauli/p1_exchange_rate.py run --workers 3 [--scale 1.0] [--ds 5 7 9 11 13] [--points 0.04:0.05 ...]
    python experiments/p1_erasure_pauli/p1_exchange_rate.py analyze [--boot 500]

``analyze`` writes results/p1_exchange_rate.json and prints the table: p_L(d), R^(d) = (dp_L/de)/(dp_L/dp) with errors
(bound: R^(d) <= c = (3/4 - p0)/(1 - e0)), and alpha, d alpha/dp, d alpha/de, R_alpha = (d alpha/de)/(d alpha/dp) for
three fit windows in d.  See lcd.analysis.exchange for the formulas and the sign convention.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import lcd
from lcd.analysis import Strata, StratifiedEstimator
from lcd.analysis.exchange import bootstrap_quantities, fit_rates, quantities, rate_per_distance
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder
from lcd.noise import sample_stratum

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("P1_M2_OUT", REPO / "results" / "p1_m2"))
WORK_POINTS = [(0.06, 0.0), (0.06, 0.02), (0.06, 0.05), (0.04, 0.0), (0.04, 0.02), (0.04, 0.05)]
DS = [5, 7, 9, 11, 13]
CUTOFF = 1e-10
# decodes of the first pass (before --scale); about 4 core-hours in total
BASE = {5: {0.04: 40_000, 0.06: 40_000}, 7: {0.04: 100_000, 0.06: 100_000}, 9: {0.04: 150_000, 0.06: 150_000},
        11: {0.04: 80_000, 0.06: 40_000}, 13: {0.04: 60_000, 0.06: 30_000}}
MS_PER_DECODE = {5: 0.07, 7: 0.25, 9: 1.5, 11: 8.5, 13: 40.0}


def table_path(d, p0, e0) -> Path:
    return OUT / "tables" / f"d{d}_p{p0:.4f}_e{e0:.4f}.npz"


def make_strata(code, d, p0, e0) -> Strata:
    # at e0 = 0 the e-derivative needs the strata with one erasure: add a dummy target with a small erasure rate
    targets = [(p0, e0)] + ([(p0, 0.01)] if e0 == 0.0 else [])
    return Strata(code.n, d, targets, cutoff=CUTOFF)


class MultiTargetEstimator(StratifiedEstimator):
    """Neyman allocation for p_L and for both derivatives at the first target (the sampling error of a derivative is
    governed by |dP_s| rather than P_s)."""

    def _neyman_weights(self) -> np.ndarray:
        S = self.strata
        p, e = S.targets[0]
        ft = self._f_smooth()
        P = S.prob(p, e)
        dPp, dPe = S.dprob(p, e)
        pl, gp, ge = float((P * ft).sum()), float((dPp * ft).sum()), float((dPe * ft).sum())
        terms = [P / pl if pl > 0 else P, np.abs(dPp) / abs(gp) if gp != 0 else np.abs(dPp),
                 np.abs(dPe) / abs(ge) if ge != 0 else np.abs(dPe)]
        a = np.maximum.reduce(terms) * np.sqrt(ft * (1 - ft))
        a[S.known_zero] = 0.0
        return a


def run_job(args) -> dict:
    d, p0, e0, scale, seed = args
    t_start = time.time()
    code = RotatedSurfaceCode(d)
    S = make_strata(code, d, p0, e0)
    if d <= 11:
        dec, kw = ExactTNMLDecoder(code), {}
    else:
        dec, kw = MPSMLDecoder(code, chi=8), dict(refine_chi=16, margin=8.0)
    stats = dict(refined=0, refine_changed=0, fallback=0)
    rng_holder = {}

    def sampler(k, w, N, rng):
        ex, ez, er = sample_stratum(code.n, k, w, N, rng)
        f = dec.fail(ex, ez, p0, er, rng, **kw)
        if d > 11:
            stats["refined"] += dec.last_refined
            stats["refine_changed"] += dec.last_refine_changed
            stats["fallback"] += int(getattr(dec, "last_fallback", 0))
        return f

    est = MultiTargetEstimator(S, sampler)
    path = table_path(d, p0, e0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path, allow_pickle=True)
        assert z["k"].shape == S.k.shape and (z["k"] == S.k).all() and (z["w"] == S.w).all(), "strata changed"
        est.fails, est.N = z["fails"].copy(), z["N"].copy()
        old = json.loads(str(z["stats"]))
        for key in stats:
            stats[key] += old.get(key, 0)
        seconds0 = float(z["seconds"])
    else:
        seconds0 = 0.0
    budget = int(BASE[d][p0] * scale)
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, int(round(p0 * 1e4)), int(round(e0 * 1e4)), int(est.N.sum())]))

    def save():
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, k=S.k, w=S.w, fails=est.fails, N=est.N, p0=p0, e0=e0, d=d, cutoff=CUTOFF, stats=json.dumps(stats),
                 seconds=seconds0 + time.time() - t_start)
        os.replace(tmp, path)

    free = np.flatnonzero(~S.known_zero)
    if est.N.sum() == 0:
        n0 = 20 if d >= 11 else 50
        for i in free:
            est._draw(int(i), n0, rng, 2000)
        save()
    rounds_left = 6
    while est.N.sum() < budget and rounds_left > 0:
        size = max(int((budget - est.N.sum()) / rounds_left), 500)
        a = est._neyman_weights()
        target = a / a.sum() * (est.N.sum() + size)
        add = np.maximum(target - est.N, 0.0)
        if add.sum() <= 0:
            break
        add = np.floor(add / add.sum() * size).astype(int)
        for i in np.flatnonzero(add > 0):
            est._draw(int(i), int(add[i]), rng, 2000)
        save()
        rounds_left -= 1
        print(f"  d={d} p={p0} e={e0}: {int(est.N.sum())}/{budget} decodes, {time.time() - t_start:.0f} s", flush=True)
    pl, gp, ge = quantities(S, est.f_hat(), p0, e0)
    return dict(d=d, p0=p0, e0=e0, decodes=int(est.N.sum()), seconds=seconds0 + time.time() - t_start, p_L=pl,
                dp=gp, de=ge, stats=stats)


def cmd_run(a):
    pts = [tuple(map(float, s.split(":"))) for s in a.points] if a.points else WORK_POINTS
    jobs = [(d, p, e, a.scale, a.seed) for d in a.ds for (p, e) in pts]
    jobs.sort(key=lambda j: -MS_PER_DECODE[j[0]] * BASE[j[0]][j[1]])
    print(f"{len(jobs)} jobs, estimated {sum(MS_PER_DECODE[j[0]] * BASE[j[0]][j[1]] * a.scale for j in jobs) / 3.6e6:.1f} core-hours", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_job, jobs):
            print(f"done d={r['d']} p={r['p0']} e={r['e0']}: {r['decodes']} decodes, {r['seconds'] / 60:.1f} min, p_L = {r['p_L']:.3e}, "
                  f"R^(d) = {r['de'] / r['dp']:.3f}, stats {r['stats']}", flush=True)


def load_table(d, p0, e0):
    z = np.load(table_path(d, p0, e0), allow_pickle=True)
    code = RotatedSurfaceCode(d)
    S = make_strata(code, d, p0, e0)
    assert (z["k"] == S.k).all() and (z["w"] == S.w).all()
    return S, z["fails"], z["N"], json.loads(str(z["stats"])), float(z["seconds"])


def R_B(p0, e0):
    beta = lambda p: 2 * np.sqrt(p * (1 - p) / 3) + 2 * p / 3
    h = 1e-7
    return (1 - beta(p0)) / ((1 - e0) * (beta(p0 + h) - beta(p0 - h)) / (2 * h))


def cmd_analyze(a):
    rng = np.random.default_rng(a.seed)
    result = dict(points=[])
    windows = [(5, 13), (7, 13), (9, 13)]
    print("p0     e0    c      R_B    | d: p_L (se)  R^(d) +- se   [decodes]")
    for (p0, e0) in WORK_POINTS:
        have = [d for d in DS if table_path(d, p0, e0).exists()]
        if len(have) < 3:
            continue
        c = (0.75 - p0) / (1 - e0)
        rows, point, boots = [], dict(p0=p0, e0=e0, c=c, R_B=float(R_B(p0, e0)), per_d=[]), []
        for d in have:
            S, fails, N, stats, secs = load_table(d, p0, e0)
            est = StratifiedEstimator(S, lambda *x: None)
            est.fails, est.N = fails.astype(float), N.astype(float)
            e = est.estimate(p0, e0)
            pl, gp, ge = quantities(S, est.f_hat(), p0, e0)
            rep = bootstrap_quantities(S, fails, N, p0, e0, a.boot, rng)
            Rd = rate_per_distance(rep[:, 0], rep[:, 1], rep[:, 2])
            boots.append(rep)
            point["per_d"].append(dict(d=d, p_L=pl, p_L_se=e.se, p_L_cert=[e.lo, e.hi], dp=gp, de=ge, R=ge / gp, R_se=float(np.nanstd(Rd)),
                                       gp=gp / pl, gp_se=float(np.nanstd(rep[:, 1] / rep[:, 0])), ge=ge / pl,
                                       ge_se=float(np.nanstd(rep[:, 2] / rep[:, 0])), decodes=int(N.sum()), core_minutes=secs / 60,
                                       mps=stats, dominant=e.dominant, omitted_mass=e.omitted_mass))
            print(f"{p0:.2f}  {e0:.2f}  {c:.3f}  {R_B(p0, e0):.3f}  | d={d}: {pl:.3e} ({e.rel_se * 100:.1f}%)  R^(d) = {ge / gp:.3f} +- {np.nanstd(Rd):.3f}  [{int(N.sum())}]")
        boots = np.stack(boots, axis=1)                          # (B, nd, 3)
        pl_b, gp_b, ge_b = boots[..., 0], boots[..., 1], boots[..., 2]
        pl0 = np.array([r["p_L"] for r in point["per_d"]])
        gp0 = np.array([r["dp"] for r in point["per_d"]])
        ge0 = np.array([r["de"] for r in point["per_d"]])
        point["fits"] = []
        for (lo, hi) in windows:
            if sum(lo <= d <= hi for d in have) < 3:
                continue
            f0 = fit_rates(have, pl0, gp0, ge0, (lo, hi))
            fb = fit_rates(have, pl_b, gp_b, ge_b, (lo, hi))
            rec = dict(window=[lo, hi], **{k: float(v) for k, v in f0.items()},
                       **{k + "_se": float(np.nanstd(v)) for k, v in fb.items()},
                       R_alpha_pct16_84=[float(x) for x in np.nanpercentile(fb["R_alpha"], [16, 84])])
            point["fits"].append(rec)
            print(f"        fit d in [{lo},{hi}]: alpha = {rec['alpha']:.3f} +- {rec['alpha_se']:.3f},  R_alpha = {rec['R_alpha']:.3f} +- {rec['R_alpha_se']:.3f}"
                  f"   (c = {c:.3f}, R_B = {R_B(p0, e0):.3f})")
        result["points"].append(point)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    result.update(script="experiments/p1_erasure_pauli/p1_exchange_rate.py", bootstrap=a.boot, cutoff=CUTOFF, git_commit=commit,
                  git_dirty_src=dirty, versions=dict(lcd=lcd.__version__, numpy=np.__version__),
                  date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    out = REPO / "results" / "p1_exchange_rate.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--ds", type=int, nargs="+", default=DS)
    r.add_argument("--points", nargs="*")
    r.add_argument("--seed", type=int, default=20241101)
    r.set_defaults(fn=cmd_run)
    an = sub.add_parser("analyze")
    an.add_argument("--boot", type=int, default=500)
    an.add_argument("--seed", type=int, default=7)
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
