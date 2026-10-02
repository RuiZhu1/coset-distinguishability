"""M2 (second half): MWPM exchange rate and the ML-MWPM decoder margin at code capacity (README P1).

The MWPM decoder (lcd.decoders.MWPMCodeCapacity: X and Z matched separately, edge weight 1, erased qubits weight 0) does not
depend on (p, e), so ONE weight-stratified table f_d(k, w) per distance serves all six work points, and the derivatives
dp_L/dp, dp_L/de computed from it are exact for this decoder (no envelope argument).  The Neyman allocation takes, per
stratum, the maximum over all work points of the normalised weights for p_L and both derivatives.
Tables are stored in results/p1_mwpm/tables/ after every round, so a run can be extended (--scale) or resumed.

    python experiments/p1_erasure_pauli/p1_mwpm_margin.py run --workers 8 [--scale 1.0] [--ds 5 7 ...]
    python experiments/p1_erasure_pauli/p1_mwpm_margin.py analyze [--boot 1000]

``analyze`` writes results/p1_mwpm_margin.json: per work point and distance p_L, R^(d) (bound R^(d) <= c holds for ML only;
for MWPM it is reported, not tested), the fits alpha, R_alpha over several d windows, and the margin against the ML results of
results/p1_exchange_rate.json: the ratio p_L^MWPM / p_L^ML per distance and Delta alpha = alpha_ML - alpha_MWPM on the same
window (theory: alpha_ML >= alpha of any decoder, Remark 1.12).  The ML and MWPM samples are independent, so their bootstrap
errors are combined in quadrature.
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
from lcd.decoders import MWPMCodeCapacity
from lcd.noise import sample_stratum
from p1_exchange_rate import WORK_POINTS, R_B

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("P1_MWPM_OUT", REPO / "results" / "p1_mwpm"))
DS = [5, 7, 9, 11, 13, 15, 17, 21]
# fit windows in d: (d_min, d_max), inverse-variance weighted; (5, 11) is the window of the ML result
FITS = [(5, 11), (7, 11), (5, 15), (9, 15), (11, 21)]
CUTOFF = 1e-10
BASE = {5: 400_000, 7: 600_000, 9: 800_000, 11: 1_000_000, 13: 1_200_000, 15: 1_500_000, 17: 1_500_000, 21: 2_000_000}


def targets():
    # at e0 = 0 the e-derivative needs the strata with one erasure: add a dummy target with a small erasure rate
    return list(WORK_POINTS) + [(p0, 0.01) for (p0, e0) in WORK_POINTS if e0 == 0.0]


def table_path(d) -> Path:
    return OUT / "tables" / f"d{d}.npz"


class AllTargetsEstimator(StratifiedEstimator):
    """Neyman allocation for p_L and both derivatives at every work point (maximum of the normalised weights)."""

    def _neyman_weights(self) -> np.ndarray:
        S = self.strata
        ft = self._f_smooth()
        a = np.zeros(S.size)
        for (p, e) in WORK_POINTS:
            P = S.prob(p, e)
            dPp, dPe = S.dprob(p, e)
            for w in (P, np.abs(dPp), np.abs(dPe)):
                tot = float((w * ft).sum())
                if tot > 0:
                    a = np.maximum(a, w / tot)
        a *= np.sqrt(ft * (1 - ft))
        a[S.known_zero] = 0.0
        return a


def run_job(args) -> dict:
    d, scale, seed = args
    t_start = time.time()
    code = RotatedSurfaceCode(d)
    S = Strata(code.n, d, targets(), cutoff=CUTOFF)
    dec = MWPMCodeCapacity(code)

    def sampler(k, w, N, rng):
        ex, ez, er = sample_stratum(code.n, k, w, N, rng)
        return dec.fail(ex, ez, er if k else None)

    est = AllTargetsEstimator(S, sampler)
    path = table_path(d)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path)
        assert z["k"].shape == S.k.shape and (z["k"] == S.k).all() and (z["w"] == S.w).all(), "strata changed"
        est.fails, est.N = z["fails"].copy(), z["N"].copy()
        seconds0 = float(z["seconds"])
    else:
        seconds0 = 0.0
    budget = int(BASE[d] * scale)
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, int(est.N.sum())]))

    def save():
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, k=S.k, w=S.w, fails=est.fails, N=est.N, d=d, cutoff=CUTOFF, seconds=seconds0 + time.time() - t_start)
        os.replace(tmp, path)

    if est.N.sum() == 0:
        for i in np.flatnonzero(~S.known_zero):
            est._draw(int(i), 200, rng, 20000)
        save()
    rounds_left = 6
    while est.N.sum() < budget and rounds_left > 0:
        size = max(int((budget - est.N.sum()) / rounds_left), 2000)
        a = est._neyman_weights()
        target = a / a.sum() * (est.N.sum() + size)
        add = np.maximum(target - est.N, 0.0)
        if add.sum() <= 0:
            break
        add = np.floor(add / add.sum() * size).astype(int)
        for i in np.flatnonzero(add > 0):
            est._draw(int(i), int(add[i]), rng, 20000)
        save()
        rounds_left -= 1
        print(f"  d={d}: {int(est.N.sum())}/{budget} decodes, {time.time() - t_start:.0f} s", flush=True)
    return dict(d=d, decodes=int(est.N.sum()), seconds=seconds0 + time.time() - t_start)


def cmd_run(a):
    jobs = sorted(((d, a.scale, a.seed) for d in a.ds), key=lambda j: -j[0])
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(run_job, jobs):
            print(f"done d={r['d']}: {r['decodes']} decodes, {r['seconds'] / 60:.1f} min", flush=True)


def load_table(d):
    z = np.load(table_path(d))
    S = Strata(int(d) ** 2, d, targets(), cutoff=CUTOFF)
    assert (z["k"] == S.k).all() and (z["w"] == S.w).all()
    return S, z["fails"], z["N"], float(z["seconds"])


def ml_reference():
    path = REPO / "results" / "p1_exchange_rate.json"
    if not path.exists():
        return {}
    pts = json.loads(path.read_text())["points"]
    return {(pt["p0"], pt["e0"]): pt for pt in pts}


def cmd_analyze(a):
    rng = np.random.default_rng(a.seed)
    have = [d for d in DS if table_path(d).exists()]
    tables = {d: load_table(d) for d in have}
    ml = ml_reference()
    result = dict(points=[], tables={str(d): dict(decodes=int(t[2].sum()), core_minutes=t[3] / 60) for d, t in tables.items()})
    print("p0     e0    c      R_B    | d: p_L^MWPM (rel se)  R^(d) +- se   p_L^MWPM/p_L^ML")
    for (p0, e0) in WORK_POINTS:
        c = (0.75 - p0) / (1 - e0)
        point = dict(p0=p0, e0=e0, c=c, R_B=float(R_B(p0, e0)), per_d=[])
        ml_pt = ml.get((p0, e0))
        ml_d = {r["d"]: r for r in ml_pt["per_d"]} if ml_pt else {}
        boots = []
        for d in have:
            S, fails, N, _ = tables[d]
            est = StratifiedEstimator(S, lambda *x: None)
            est.fails, est.N = fails.astype(float), N.astype(float)
            e = est.estimate(p0, e0)
            pl, gp, ge = quantities(S, est.f_hat(), p0, e0)
            rep = bootstrap_quantities(S, fails, N, p0, e0, a.boot, rng)
            boots.append(rep)
            Rd = rate_per_distance(rep[:, 0], rep[:, 1], rep[:, 2])
            row = dict(d=d, p_L=pl, p_L_se=e.se, p_L_cert=[e.lo, e.hi], dp=gp, de=ge, R=ge / gp, R_se=float(np.nanstd(Rd)),
                       dominant=e.dominant, omitted_mass=e.omitted_mass)
            if d in ml_d:
                r_ml = ml_d[d]
                ratio = pl / r_ml["p_L"]
                row.update(ratio_to_ML=ratio, ratio_to_ML_se=ratio * float(np.hypot(e.se / pl, r_ml["p_L_se"] / r_ml["p_L"])),
                           R_ML=r_ml["R"], R_ML_se=r_ml["R_se"])
            point["per_d"].append(row)
            extra = f"   {row['ratio_to_ML']:.3f} +- {row['ratio_to_ML_se']:.3f}" if "ratio_to_ML" in row else ""
            print(f"{p0:.2f}  {e0:.2f}  {c:.3f}  {R_B(p0, e0):.3f}  | d={d}: {pl:.3e} ({e.rel_se * 100:.1f}%)  "
                  f"R^(d) = {ge / gp:.3f} +- {np.nanstd(Rd):.3f}{extra}")
        boots = np.stack(boots, axis=1)                          # (B, nd, 3)
        pl_b, gp_b, ge_b = boots[..., 0], boots[..., 1], boots[..., 2]
        pl0 = np.array([r["p_L"] for r in point["per_d"]])
        gp0 = np.array([r["dp"] for r in point["per_d"]])
        ge0 = np.array([r["de"] for r in point["per_d"]])
        fin = lambda x: np.where(np.isfinite(x), x, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = dict(lnp=np.nanstd(fin(np.log(pl_b)), axis=0), gp=np.nanstd(fin(gp_b / pl_b), axis=0),
                         ge=np.nanstd(fin(ge_b / pl_b), axis=0))
        point["sigma_per_d"] = {k: [float(x) for x in v] for k, v in sigma.items()}
        point["fits"] = []
        for (lo, hi) in FITS:
            if sum(lo <= d <= hi for d in have) < 3:
                continue
            f0 = fit_rates(have, pl0, gp0, ge0, (lo, hi), sigma)
            fb = fit_rates(have, pl_b, gp_b, ge_b, (lo, hi), sigma)
            rec = dict(window=[lo, hi], **{k: float(v) for k, v in f0.items()},
                       **{k + "_se": float(np.nanstd(v[np.isfinite(v)])) for k, v in fb.items()})
            ml_fit = next((f for f in (ml_pt or {}).get("fits", []) if f["window"] == [lo, hi] and f["weighted"]), None)
            if ml_fit is not None:
                rec.update(alpha_ML=ml_fit["alpha"], alpha_ML_se=ml_fit["alpha_se"], R_alpha_ML=ml_fit["R_alpha"],
                           R_alpha_ML_se=ml_fit["R_alpha_se"], delta_alpha=ml_fit["alpha"] - rec["alpha"],
                           delta_alpha_se=float(np.hypot(ml_fit["alpha_se"], rec["alpha_se"])))
            point["fits"].append(rec)
            margin = (f"   Delta alpha = alpha_ML - alpha_MWPM = {rec['delta_alpha']:.3f} +- {rec['delta_alpha_se']:.3f}"
                      if "delta_alpha" in rec else "")
            print(f"        fit d in [{lo},{hi}]: alpha = {rec['alpha']:.3f} +- {rec['alpha_se']:.3f},  "
                  f"R_alpha = {rec['R_alpha']:.3f} +- {rec['R_alpha_se']:.3f}{margin}")
        result["points"].append(point)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    result.update(script="experiments/p1_erasure_pauli/p1_mwpm_margin.py", decoder="MWPM (X/Z separate, erased weight 0)",
                  bootstrap=a.boot, cutoff=CUTOFF, git_commit=commit, git_dirty_src=dirty,
                  versions=dict(lcd=lcd.__version__, numpy=np.__version__),
                  date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    out = REPO / "results" / "p1_mwpm_margin.json"
    out.write_text(json.dumps(result, indent=2, default=float))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--ds", type=int, nargs="+", default=DS)
    r.add_argument("--seed", type=int, default=20261002)
    r.set_defaults(fn=cmd_run)
    an = sub.add_parser("analyze")
    an.add_argument("--boot", type=int, default=1000)
    an.add_argument("--seed", type=int, default=7)
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
