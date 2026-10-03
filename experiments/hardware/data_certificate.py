"""Certificate of hardware logical error rates from the data's own detection statistics (theory, end of Section 4).

Under assumption (G) (the observable part of the noise is an independent-edge model on the decoder's observable graph G_o;
i.i.d. shots), every edge probability q_e is a function of the first and second moments of the detection events
(the p_ij identity). Simultaneous confidence intervals q_lo <= q_e <= q_hi at level 1 - delta (Bonferroni, delta = 1e-3)
are plugged into Theorem 4.28(c) (at q_hi) and Theorem 4.32 (x_e from q_hi, c_e from q_lo). With probability >= 1 - delta
over the data, the per-shot failure probability of minimum-weight matching with the decoder's (pymatching) weights, and
hence that of the maximum-likelihood decoder, is at most the certified bound. Methods (``lcd.analysis.data_certificate``):
  cp        Clopper-Pearson for every a_i and c_ij, interval arithmetic (the literal recipe);
  paired    direct covariance statistic on shot pairs, betting confidence intervals (sharper for the interior edges);
  paired+T  as paired, with edges of the same type pooled across rounds (additional assumption (T), time stationarity).

Part 1, validation on simulated data (where (G) holds exactly):
  * the p_ij identity on 2e6 shots of stim's rotated memory circuit;
  * coverage: 20 independent data sets of 50,000 shots, each from (a) stim's circuit (d = 3, 9 rounds, p = 2e-3) and
    (b) the published Willow d = 3 decoder model itself (RL-optimized prior, patch d3_at_q6_3, X, 10 rounds): fraction of
    data sets in which every edge interval covers the true q (should be >= 1 - delta);
  * bound vs truth: certified bounds (median over the 20 sets) against the pymatching failure rate on 1e6 shots.
Part 2, hardware: Willow 105Q (Zenodo 10.5281/zenodo.13273331; every d = 3, 5, 7 patch, X and Z, r = 1 and r = 10;
decoder model: correlated-matching RL-optimized prior error_model.dem; weights: pymatching on it) and Sycamore 2022
(Zenodo 10.5281/zenodo.6804040; surface code d = 3 (four patches) and d = 5, X and Z, r = 1 and r = 3; decoder model:
pij_from_even_for_odd.dem). Per experiment: certified bounds (Theorems 4.28 and 4.32) for the three methods, the bounds
at the point estimate, the observed failure rate of pymatching with the same model on all shots and of the stored decoders
(99% Clopper-Pearson), the certified margin s* (largest s with a finite Theorem 4.32 bound when q_lo, q_hi are multiplied
by s; paired, unpooled; bisection to 2%), and two diagnostics of the assumptions: non-edge pairs of G_o detectors with a
significant covariance (zero under (G)), and the drift of the detection fraction over chunks of 1000 shots (i.i.d.).

Run:  python experiments/hardware/data_certificate.py [--quick] [--reps 20]   (about 30 min; < 3 GB)
Writes results/data_certificate.json (not with --quick). Needs data/ (not in the repository) for part 2 and part 1(b).
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
import pymatching
import scipy
import stim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.circuit_peierls import dem_graph  # noqa: E402
from lcd.analysis.circuit_peierls_geodesic import geodesic_bound  # noqa: E402
from lcd.analysis.data_certificate import (certify, clopper_pearson, edge_intervals, nonedge_correlations,  # noqa: E402
                                           point_estimates, stationarity)

DELTA = 1e-3
LAMS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
METHODS = (("cp", False), ("paired", False), ("paired", True))
DATA = REPO / "data"
WILLOW = DATA / "willow" / "google_105Q_surface_code_d3_d5_d7"
RL = "correlated_matching_decoder_with_rl_optimized_prior"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def cp99(k: int, n: int) -> list:
    lo, hi = clopper_pearson([k], n, 0.01)
    return [float(lo[0]), float(hi[0])]


def mwpm_rate(dem, X, obs) -> dict:
    pred = pymatching.Matching.from_detector_error_model(dem).decode_batch(X)[:, 0]
    k = int(np.sum(pred != obs))
    return dict(fails=k, shots=int(obs.size), rate=k / obs.size, cp99=cp99(k, obs.size))


def certificates(dem, X, coords, G, margin: bool = True) -> dict:
    out = {}
    for meth, pool in METHODS:
        if pool and coords is None:
            continue
        t = time.time()
        r = certify(dem, X, DELTA, meth, pool, coords, G=G, lams=LAMS)
        iv = r.pop("_intervals")
        r["seconds"] = round(time.time() - t, 1)
        key = meth + ("+T" if pool else "")
        if margin and key == "paired":
            r["margin_4_32"] = certified_margin(dem, G, iv.q_lo, iv.q_hi)
        out[key] = r
    return out


def certified_margin(dem, G, q_lo, q_hi, s_max: float = 4.0, rtol: float = 0.02) -> float | None:
    """Largest s in (0, s_max] (to rtol) with a finite Theorem 4.32 bound at (s q_lo, s q_hi), clipped at 1/2."""
    def fin(s):
        qu = np.minimum(0.5, s * q_hi)
        return bool(np.isfinite(geodesic_bound(dem, lams=LAMS, G=G, q_upper=qu, q_lower=np.minimum(qu, s * q_lo))["bound"]))
    if fin(s_max):
        return s_max
    hi, lo = s_max, s_max / 2
    while not fin(lo):
        hi, lo = lo, lo / 2
        if lo < 1e-3:
            return None
    while hi / lo > 1 + rtol:
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if fin(mid) else (lo, mid)
    return float(lo)


# ------------------------------------------------------------------ part 1: validation
def stim_circuit(d, rounds, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds,
                                  after_clifford_depolarization=p, after_reset_flip_probability=p,
                                  before_measure_flip_probability=p, before_round_data_depolarization=p)


def validation(reps: int, quick: bool) -> dict:
    out = {}
    c = stim_circuit(3, 3, 2e-3)
    dem = c.detector_error_model(decompose_errors=True)
    G = dem_graph(dem)
    X = c.compile_detector_sampler(seed=11).sample(200_000 if quick else 2_000_000)
    qh = point_estimates(G, X)
    out["identity"] = dict(model="stim rotated_memory_z d=3 r=3 p=2e-3", shots=int(X.shape[0]),
                           edges=len(G.edges), max_abs_error=float(np.abs(qh - G.q).max()), max_q=float(G.q.max()),
                           median_rel_error=float(np.median(np.abs(qh - G.q) / G.q)))
    print("identity:", out["identity"], flush=True)
    models = [("stim d=3 r=9 p=2e-3", stim_circuit(3, 9, 2e-3), None)]
    wd = WILLOW / "d3_at_q6_3" / "X" / "r10"
    if (wd / "decoding_results" / RL / "error_model.dem").exists():
        models.append(("Willow d3_at_q6_3 X r10 RL-prior model", None, wd))
    out["coverage"] = []
    for name, circ, path in models:
        if circ is not None:
            dem = circ.detector_error_model(decompose_errors=True)
            coords = circ.get_detector_coordinates()
            sampler = lambda n, s, circ=circ: circ.compile_detector_sampler(seed=s).sample(n, separate_observables=True)  # noqa: E731
        else:
            dem = stim.DetectorErrorModel.from_file(path / "decoding_results" / RL / "error_model.dem")
            coords = stim.Circuit.from_file(path / "circuit_ideal.stim").get_detector_coordinates()
            sampler = lambda n, s, dem=dem: dem.compile_sampler(seed=s).sample(n)[:2]  # noqa: E731
        G = dem_graph(dem)
        Xs, obs = sampler(400_000 if quick else 1_000_000, 99)
        truth = mwpm_rate(dem, Xs, obs[:, 0])
        row = dict(model=name, edges=len(G.edges), reps=reps, shots=50_000, delta=DELTA, mwpm_simulated=truth,
                   bound_4_32_true_q=geodesic_bound(dem, lams=LAMS, G=G)["bound"], methods={})
        for meth, pool in METHODS:
            cov, b28, b32, rel, worst = 0, [], [], [], []
            for r in range(reps):
                X, _ = sampler(50_000, 1000 + r)
                iv = edge_intervals(G, X, DELTA, meth, pool, coords)
                ok = (iv.q_lo <= G.q + 1e-15) & (G.q <= iv.q_hi + 1e-15)
                cov += bool(ok.all())
                worst.append(int((~ok).sum()))
                rel.append(float(np.median((iv.q_hi - iv.q_lo) / (2 * np.maximum(G.q, 1e-12)))))
                if r < (2 if quick else 5):
                    res = certify(dem, X, DELTA, meth, pool, coords, G=G, lams=LAMS)
                    b28.append(res["bound_4_28"])
                    b32.append(res["bound_4_32"])
            key = meth + ("+T" if pool else "")
            row["methods"][key] = dict(all_covered=cov, uncovered_edges_per_rep=worst, rel_halfwidth_median=float(np.median(rel)),
                                       bound_4_28_median=float(np.median(b28)), bound_4_32_median=float(np.median(b32)),
                                       bound_4_32_min=float(np.min(b32)), bounds_on_first_reps=len(b32))
            print(f"  {name} {key}: covered {cov}/{reps}, rel. half-width {np.median(rel):.2f}, "
                  f"4.32 median {np.median(b32):.3g} (min {np.min(b32):.3g}) vs simulated {truth['rate']:.3g}", flush=True)
        out["coverage"].append(row)
    return out


# ------------------------------------------------------------------ part 2: hardware
def read_b8_one_bit(path: Path, shots: int) -> np.ndarray:
    b = np.fromfile(path, np.uint8)
    if b.size != shots:
        raise ValueError(f"{path}: {b.size} bytes for {shots} shots")
    return b & 1


def read_01(path: Path) -> np.ndarray:
    b = np.frombuffer(path.read_bytes(), np.uint8)
    return (b[(b == ord("0")) | (b == ord("1"))] - ord("0")).astype(np.uint8)


def experiment(device, d: Path, dem_path: Path, X, obs, coords, stored: dict, meta: dict, margin: bool) -> dict:
    dem = stim.DetectorErrorModel.from_file(dem_path)
    G = dem_graph(dem)
    t = time.time()
    row = dict(device=device, **meta, shots=int(X.shape[0]), decoder_model=str(dem_path.relative_to(DATA)), stats=G.stats,
               detection_fraction=float(X.mean()))
    row["mwpm_observed"] = mwpm_rate(dem, X, obs)
    row["stored_decoders"] = {k: dict(rate=float(np.mean(v != obs)), cp99=cp99(int(np.sum(v != obs)), obs.size))
                              for k, v in stored.items()}
    row["diagnostics"] = dict(nonedge=nonedge_correlations(G, X, DELTA), stationarity=stationarity(X))
    row["certificates"] = certificates(dem, X, coords, G, margin)
    row["seconds"] = round(time.time() - t, 1)
    c = row["certificates"]
    print(f"{device:8s} {meta['name']:34s} obs {row['mwpm_observed']['rate']:.4f}  point 4.32 {c['paired']['bound_4_32_point']:.3g}"
          f"  cert 4.32: cp {c['cp']['bound_4_32']:.3g} paired {c['paired']['bound_4_32']:.3g}"
          f" +T {c.get('paired+T', {}).get('bound_4_32', float('nan')):.3g}  4.28 paired {c['paired']['bound_4_28']:.3g}"
          f"  s* {c['paired'].get('margin_4_32')}  non-edge exceed {row['diagnostics']['nonedge']['exceed']}"
          f"  ({row['seconds']} s)", flush=True)
    return row


def willow(rounds_list, quick: bool) -> list:
    rows = []
    for meta_path in sorted(WILLOW.glob("*/*/r*/metadata.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        if int(meta["rounds"]) not in rounds_list or not (d / "detection_events.b8").exists():
            continue
        if quick and not d.parts[-3].startswith("d3_at_q6_3"):
            continue
        dem_path = d / "decoding_results" / RL / "error_model.dem"
        circ = stim.Circuit.from_file(d / "circuit_ideal.stim")
        n = int(meta["shots"])
        X = stim.read_shot_data_file(path=str(d / "detection_events.b8"), format="b8", num_detectors=circ.num_detectors)
        obs = read_b8_one_bit(d / "obs_flips_actual.b8", n)
        stored = {f.parent.name: read_b8_one_bit(f, n) for f in sorted(d.glob("decoding_results/*/obs_flips_predicted.b8"))}
        rows.append(experiment("willow", d, dem_path, X, obs, circ.get_detector_coordinates(), stored,
                               dict(name=str(d.relative_to(WILLOW)), patch=d.parts[-3], basis=meta["basis"],
                                    d=int(meta["distance"]), rounds=int(meta["rounds"])), margin=True))
    return rows


SYC_RE = re.compile(r"surface_code_b([XZ])_d(\d+)_r(\d+)_center_(\d+)_(\d+)$")


def sycamore(rounds_list, quick: bool) -> list:
    rows = []
    for d in sorted(DATA.iterdir()):
        m = SYC_RE.match(d.name)
        if not m or int(m.group(3)) not in rounds_list:
            continue
        if quick and d.name != "surface_code_bZ_d3_r01_center_3_5":
            continue
        props = dict(line.split(": ", 1) for line in (d / "properties.yml").read_text().splitlines() if ": " in line)
        n, nd = int(props["shots"]), int(props["circuit_detectors"])
        X = stim.read_shot_data_file(path=str(d / "detection_events.b8"), format="b8", num_detectors=nd)
        obs = read_01(d / "obs_flips_actual.01")
        stored = {f.stem.removeprefix("obs_flips_predicted_by_"): read_01(f) for f in sorted(d.glob("obs_flips_predicted_by_*.01"))}
        coords = stim.Circuit.from_file(d / "circuit_ideal.stim").get_detector_coordinates()
        rows.append(experiment("sycamore", d, d / "pij_from_even_for_odd.dem", X, obs, coords, stored,
                               dict(name=d.name, patch=f"{m.group(4)}_{m.group(5)}", basis=m.group(1), d=int(m.group(2)),
                                    rounds=int(m.group(3))), margin=True))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="few repetitions and experiments; no output file")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results" / "data_certificate.json"))
    a = ap.parse_args()
    t0 = time.time()
    out: dict = dict(meta=dict(script="experiments/hardware/data_certificate.py", args=vars(a), git_commit=git("rev-parse", "HEAD"),
                               git_dirty_src=bool(git("status", "--porcelain", "src", "experiments")),
                               date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               versions=dict(stim=stim.__version__, pymatching=pymatching.__version__, numpy=np.__version__,
                                             scipy=scipy.__version__),
                               delta=DELTA, lams=LAMS, theorem="theory/sec4-certification.tex, end of Section 4 "
                               "(data certificate), Theorems 4.28(c) and 4.32",
                               data=dict(willow="Zenodo 10.5281/zenodo.13273331 (partial extraction: r01 and r10 detection "
                                                "events, circuits, decoder models, predictions)",
                                         sycamore="Zenodo 10.5281/zenodo.6804040")))
    if not a.skip_validation:
        out["validation"] = validation(3 if a.quick else a.reps, a.quick)
    out["willow"] = willow((1, 10), a.quick)
    out["sycamore"] = sycamore((1, 3), a.quick)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    if not a.quick:
        Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n")
        print("wrote", a.out)
    print(f"done ({out['meta']['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
