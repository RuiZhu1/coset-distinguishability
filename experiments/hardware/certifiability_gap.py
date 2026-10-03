"""Certifiability gap of published hardware detector error models for the circuit-level Peierls bound (theory, Theorem 4.28).

The bound of Theorem 4.28 is infinite on every published hardware model we tried. The certifiability margin s* (see
``lcd.analysis.hardware``) says by how much: every mechanism probability is multiplied by s (clipped at 1/2), pymatching's
weights are rebuilt from the scaled model, and s* is the largest s with a finite bound at lam = 1/2 (geometric bisection to
1e-3 relative precision in (0, 4]; "> 4" if the bound is still finite at s = 4). s* < 1: the noise must be 1/s* times
lower before the theorem certifies anything; s* > 1: headroom.

Data (``data/``, not in the repository):
  Sycamore 2022   Google Quantum AI, "Suppressing quantum errors by scaling a surface code logical qubit", Nature 614 (2023);
                  Zenodo 10.5281/zenodo.6804040, unpacked into data/{code}_b{basis}_d{d}_r{rounds}_center_{row}_{col}/
                  (131 experiments: surface code d = 3 (four patches) and d = 5, bases X and Z, r = 1, 3, ..., 25, and the
                  d = 25 repetition code, r = 50). Models: circuit_detector_error_model.dem (from Google's fitted noisy
                  circuit; decodes all shots with pymatching / correlated matching) and pij_from_even_for_odd.dem /
                  pij_from_odd_for_even.dem (estimated from the detection events of the even / odd shots; used by belief
                  matching and tensor-network decoding on the odd / even shots). Reported for the pij models: both s* and
                  their mean.
  Willow 2024     Google Quantum AI, "Quantum error correction below the surface code threshold", Nature 638 (2025);
                  Zenodo 10.5281/zenodo.13273331, file google_105Q_surface_code_d3_d5_d7.zip, of which only metadata.json,
                  obs_flips_actual.b8, decoding_results/*/obs_flips_predicted.b8 and the correlated-matching error_model.dem
                  (SI1000 prior and RL-optimized prior; the harmony / libra decoders use the same files) were extracted by
                  HTTP range requests, for every patch (nine d = 3, four d = 5, one d = 7), both bases, r = 10 and r = 110,
                  into data/willow/google_105Q_surface_code_d3_d5_d7/{patch}/{basis}/r{rounds}/.

For every experiment and model: whether ``dem_graph`` accepts the model (re-decomposition statistics, or the reason for
refusal), the bound at s = 1 (lam optimized and at lam = 1/2), s*; for every decoder with stored predictions: the observed
per-shot failure rate with a 99% Clopper-Pearson interval and, for these memory experiments, the per-round logical error
rate eps_r from 1 - 2 P_L(r) = (1 - 2 eps_r)^r. eps_r is an estimate: it attributes state preparation and final
measurement to the rounds (Google's papers fit eps over several r instead).

    python experiments/hardware/certifiability_gap.py [--quick] [--procs 4]

Writes results/hardware_gap.json (not with --quick). Deterministic (no sampling).
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pymatching
import scipy
import stim
from scipy.stats import beta as beta_dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.circuit_peierls import dem_graph, peierls_bound  # noqa: E402
from lcd.analysis.hardware import bound_at_scale, certifiability_margin  # noqa: E402

DATA = REPO / "data"
WILLOW = DATA / "willow" / "google_105Q_surface_code_d3_d5_d7"
SYC_RE = re.compile(r"(surface_code|repetition_code)_b([XZ])_d(\d+)_r(\d+)_center_(\d+)_(\d+)$")
S_MAX, RTOL, LAM = 4.0, 1e-3, 0.5


# ------------------------------------------------------------------ statistics
def cp(k: int, n: int, level: float = 0.99) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def per_round(P: float, r: int) -> float:
    """eps with 1 - 2 P = (1 - 2 eps)^r; nan if P >= 1/2."""
    return float((1 - (1 - 2 * P) ** (1 / r)) / 2) if P < 0.5 else float("nan")


def decoder_row(actual: np.ndarray, pred: np.ndarray, rounds: int) -> dict:
    if actual.shape != pred.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {pred.shape}")
    k, n = int(np.sum(actual != pred)), int(actual.size)
    lo, hi = cp(k, n)
    return dict(fails=k, shots=n, rate=k / n, cp99=[lo, hi], eps_round=per_round(k / n, rounds),
                eps_round_cp99=[per_round(lo, rounds), per_round(hi, rounds)])


def read_01(path: Path) -> np.ndarray:
    b = np.frombuffer(path.read_bytes(), np.uint8)
    return (b[(b == ord("0")) | (b == ord("1"))] - ord("0")).astype(np.uint8)


def read_b8_one_bit(path: Path, shots: int) -> np.ndarray:
    """b8 with one bit per shot: every shot is padded to a byte, the bit is the least significant one."""
    b = np.fromfile(path, np.uint8)
    if b.size != shots:
        raise ValueError(f"{path}: {b.size} bytes for {shots} shots")
    return b & 1


# ------------------------------------------------------------------ one model
def model_row(path: Path, geodesic: bool = True) -> dict:
    dem = stim.DetectorErrorModel.from_file(path).flattened()
    ps = np.array([i.args_copy()[0] for i in dem if i.type == "error"])
    row = dict(file=str(path.relative_to(DATA)), mechanisms=int(ps.size), mean_p=float(ps.mean()), max_p=float(ps.max()),
               sum_p=float(ps.sum()), detectors=dem.num_detectors)
    try:
        G = dem_graph(dem)
    except (ValueError, NotImplementedError) as e:
        row.update(accepted=False, refusal=f"{type(e).__name__}: {e}")
        return row
    row.update(accepted=True, stats=G.stats)
    try:
        r = peierls_bound(dem, G=G)
        row.update(bound_s1=r["bound"], lam_s1=r["lam"], rho_upper_s1=r["rho_upper"], delta=r["delta"],
                   bound_s1_lam_half=bound_at_scale(dem, 1.0, LAM))
        m = certifiability_margin(dem, s_max=S_MAX, rtol=RTOL, lam=LAM)
        row.update(margin=m, s_star=m["s_star"], s_star_label=">4" if m["capped"] else f"{m['s_star']:.4g}")
        if geodesic:
            try:
                g = certifiability_margin(dem, s_max=S_MAX, rtol=1e-2, method="4.32")
                row.update(margin_4_32=g, s_star_4_32=g["s_star"])
            except NotImplementedError as e:
                row.update(s_star_4_32=None, refusal_4_32=f"{type(e).__name__}: {e}")
    except (ValueError, NotImplementedError) as e:
        row.update(accepted=False, refusal=f"{type(e).__name__}: {e}")
    return row


# ------------------------------------------------------------------ experiments
def sycamore_jobs() -> list:
    jobs = []
    for d in sorted(DATA.iterdir()):
        m = SYC_RE.match(d.name)
        if m and d.is_dir():
            jobs.append(("sycamore", str(d)))
    return jobs


def willow_jobs() -> list:
    return [("willow", str(p.parent)) for p in sorted(WILLOW.glob("*/*/r*/metadata.json"))]


def run_sycamore(path: str) -> dict:
    d = Path(path)
    m = SYC_RE.match(d.name)
    props = dict(line.split(": ", 1) for line in (d / "properties.yml").read_text().splitlines() if ": " in line)
    rounds, shots = int(props["rounds"]), int(props["shots"])
    out = dict(device="sycamore", name=d.name, code=m.group(1), basis=m.group(2), d=int(m.group(3)), rounds=rounds,
               patch=f"{m.group(5)}_{m.group(6)}", shots=shots, type=props["type"])
    out["models"] = {"circuit": model_row(d / "circuit_detector_error_model.dem"),
                     "pij_even_for_odd": model_row(d / "pij_from_even_for_odd.dem"),
                     "pij_odd_for_even": model_row(d / "pij_from_odd_for_even.dem")}
    s2 = [out["models"][k].get("s_star") for k in ("pij_even_for_odd", "pij_odd_for_even")]
    g2 = [out["models"][k].get("s_star_4_32") for k in ("pij_even_for_odd", "pij_odd_for_even")]
    out["s_star"] = dict(circuit=out["models"]["circuit"].get("s_star"),
                         pij=float(np.mean(s2)) if all(x is not None for x in s2) else None, pij_both=s2,
                         circuit_4_32=out["models"]["circuit"].get("s_star_4_32"),
                         pij_4_32=float(np.mean(g2)) if all(x is not None for x in g2) else None)
    actual = read_01(d / "obs_flips_actual.01")
    assert actual.size == shots
    out["decoders"] = {}
    for f in sorted(d.glob("obs_flips_predicted_by_*.01")):
        name = f.stem.removeprefix("obs_flips_predicted_by_")
        out["decoders"][name] = decoder_row(actual, read_01(f), rounds)
    return out


def run_willow(path: str) -> dict:
    d = Path(path)
    meta = json.loads((d / "metadata.json").read_text())
    rounds, shots = int(meta["rounds"]), int(meta["shots"])
    out = dict(device="willow", name=str(d.relative_to(WILLOW)), code="surface_code", basis=meta["basis"],
               d=int(meta["distance"]), rounds=rounds, patch=d.parents[1].name, shots=shots,
               type="surface_code_memory_experiment")
    out["models"] = {}
    for prior in ("si1000", "rl_optimized"):
        f = d / "decoding_results" / f"correlated_matching_decoder_with_{prior}_prior" / "error_model.dem"
        out["models"][prior] = model_row(f, geodesic=rounds <= 25)   # Theorem 4.32 is costly on long experiments
    out["s_star"] = {k: v.get("s_star") for k, v in out["models"].items()}
    out["s_star"].update({f"{k}_4_32": v.get("s_star_4_32") for k, v in out["models"].items()})
    actual = read_b8_one_bit(d / "obs_flips_actual.b8", shots)
    out["decoders"] = {}
    for f in sorted(d.glob("decoding_results/*/obs_flips_predicted.b8")):
        out["decoders"][f.parent.name] = decoder_row(actual, read_b8_one_bit(f, shots), rounds)
    return out


def run(job):
    kind, path = job
    t = time.time()
    try:
        r = run_sycamore(path) if kind == "sycamore" else run_willow(path)
    except Exception as e:  # recorded, never silently dropped
        r = dict(device=kind, name=path, error=f"{type(e).__name__}: {e}")
    r["seconds"] = round(time.time() - t, 2)
    return r


# ------------------------------------------------------------------ summary
def summarize(rows: list) -> list:
    """Per device / code / distance / basis (and rounds bucket): median and range of s* per model type, and the observed
    per-round error of every decoder (median over the group)."""
    groups = collections.defaultdict(list)
    for r in rows:
        if "error" in r:
            continue
        groups[(r["device"], r["code"], r["d"], r["basis"])].append(r)
        if r["device"] == "willow":
            groups[(r["device"], r["code"], r["d"], r["basis"], r["rounds"])].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        g = dict(device=key[0], code=key[1], d=key[2], basis=key[3], rounds=key[4] if len(key) > 4 else "all",
                 experiments=len(rs), patches=len({r["patch"] for r in rs}), rounds_list=sorted({r["rounds"] for r in rs}))
        g["s_star"] = {}
        for t in rs[0]["s_star"]:
            if t.startswith("pij_both"):
                continue
            v = [r["s_star"][t] for r in rs if r["s_star"][t] is not None]
            if v:
                g["s_star"][t] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v),
                                      capped=sum(1 for x in v if x >= S_MAX))
        g["refused"] = sorted({f"{r['name']}:{k}: {m['refusal']}" for r in rs for k, m in r["models"].items()
                               if not m.get("accepted")})
        g["eps_round"] = {}
        for dec in sorted({k for r in rs for k in r["decoders"]}):
            v = [r["decoders"][dec]["eps_round"] for r in rs if dec in r["decoders"]]
            g["eps_round"][dec] = dict(median=float(np.nanmedian(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v)))
        out.append(g)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="a few experiments; no output file")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO,
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    out: dict = dict(meta=dict(script="experiments/hardware/certifiability_gap.py", args=vars(a), git_commit=commit,
                               git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               versions=dict(stim=stim.__version__, pymatching=pymatching.__version__,
                                             numpy=np.__version__, scipy=scipy.__version__),
                               theorem="theory/sec4-certification.tex, Theorem 4.28",
                               margin=dict(s_max=S_MAX, rtol=RTOL, lam=LAM, scaling="p -> min(1/2, s p), pymatching "
                                           "weights of the scaled model, bound finite iff the walk sum converges"),
                               data=dict(sycamore="Zenodo 10.5281/zenodo.6804040 (google_qec3v5_experiment_data.zip)",
                                         willow="Zenodo 10.5281/zenodo.13273331 (google_105Q_surface_code_d3_d5_d7.zip; "
                                                "partial extraction: r10 and r110, metadata, observable flips, "
                                                "predictions, correlated-matching error models)")))
    jobs = sycamore_jobs() + willow_jobs()
    if not jobs:
        print("no data under", DATA)
        return 1
    if a.quick:
        jobs = [j for j in jobs if "bZ_d3_r05" in j[1]][:3] + [j for j in jobs if j[1].endswith("d3_at_q6_7/Z/r10")]
    print(f"{len(jobs)} experiments, {a.procs} processes", flush=True)
    rows = []
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=8) as pool:
        # largest models first, so that the long jobs do not end up last
        jobs.sort(key=lambda j: -sum(f.stat().st_size for f in Path(j[1]).rglob("*.dem")))
        for r in pool.imap_unordered(run, jobs):
            rows.append(r)
            if "error" in r:
                print(f"  ERROR {r['name']}: {r['error']}", flush=True)
            else:
                s = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r["s_star"].items()
                             if k != "pij_both")
                print(f"  [{len(rows):3d}/{len(jobs)}] {r['device']:8s} {r['name']:45s} s*: {s}  ({r['seconds']:.1f}s)",
                      flush=True)
    rows.sort(key=lambda r: (r["device"], r["name"]))
    out["experiments"] = rows
    out["summary"] = summarize(rows)
    for g in out["summary"]:
        ss = "  ".join(f"{t}: {v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}]" for t, v in g["s_star"].items())
        ep = "  ".join(f"{k}: {v['median']:.2e}" for k, v in g["eps_round"].items())
        print(f"{g['device']:8s} {g['code']:15s} d={g['d']:2d} {g['basis']} r={g['rounds']}: n={g['experiments']:3d}  s* {ss}"
              f"\n{'':40s}eps_r {ep}" + (f"\n{'':40s}refused {g['refused']}" if g["refused"] else ""))
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    out["meta"]["errors"] = [r["name"] for r in rows if "error" in r]
    if not a.quick:
        path = REPO / "results" / "hardware_gap.json"
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print("wrote", path.relative_to(REPO))
    print(f"done ({time.time() - t0:.0f}s)")
    return 1 if out["meta"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
