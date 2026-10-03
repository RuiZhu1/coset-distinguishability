"""Circuit-level Peierls bound for MWPM (theory Theorem 4.28, Propositions 4.29 and 4.30): numbers for the stim rotated memory
circuit (rounds = d), and a comparison with simulated MWPM failure rates.

Noise models (stim's generated circuits):
  circuit           after_clifford_depolarization = after_reset_flip_probability = before_measure_flip_probability
                    = before_round_data_depolarization = p          (uniform circuit-level noise)
  phenomenological  before_round_data_depolarization = before_measure_flip_probability = p, noiseless gates

Parts:
  (1) the bound of Theorem 4.28 (non-backtracking walk sum, lam optimized, pymatching's weights) for d <= 15;
  (2) the lattice form of Proposition 4.30: edge factors of the finite graph are at most an envelope that does not depend on
      d (checked for 5 <= d <= 15); the tilted non-backtracking matrix of the envelope lattice gives
          p_L(d) <= N0(d) bb^2 s(theta) R(theta) / (1 - rho(theta)) * exp(-theta (d - 2))   for every theta with rho(theta) < 1,
      hence the certified exponent theta*(p) (largest root of rho(theta) = 1) and the range rho(0) < 1;
  (3) simulated pymatching failure rates at small d with Clopper-Pearson limits: the bound must not be below the lower limit.

    python experiments/certification/circuit_peierls.py [--quick]

Writes results/circuit_peierls.json (not with --quick).  Exit status non-zero if a check fails.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
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
from lcd.analysis.circuit_peierls import dem_graph, edge_factors, peierls_bound, pymatching_weights  # noqa: E402

MODELS = {
    "circuit": lambda p: dict(after_clifford_depolarization=p, after_reset_flip_probability=p,
                              before_measure_flip_probability=p, before_round_data_depolarization=p),
    "phenomenological": lambda p: dict(before_measure_flip_probability=p, before_round_data_depolarization=p),
}
PS = {"circuit": [1e-4, 2e-4, 5e-4, 1e-3, 1.5e-3, 2e-3], "phenomenological": [1e-3, 2e-3, 5e-3, 1e-2]}
DS = [3, 5, 7, 9, 11, 13, 15]
SIM = {"circuit": [(3, 1e-3, 400_000), (3, 3e-3, 200_000), (5, 5e-4, 4_000_000), (5, 1e-3, 2_000_000),
                   (5, 2e-3, 1_000_000), (7, 1e-3, 10_000_000), (7, 2e-3, 1_000_000)],
       "phenomenological": [(3, 5e-3, 200_000), (5, 5e-3, 1_000_000), (5, 1e-2, 400_000), (7, 1e-2, 1_000_000)]}

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def make_circuit(model: str, d: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, **MODELS[model](p))


# ------------------------------------------------------------------ (2) the envelope lattice
def finite_data(model: str, d: int, p: float, lam: float = 0.5) -> dict:
    """Edge factors of the finite graph grouped by offset, the boundary factors, the two sides, and the central star."""
    c = make_circuit(model, d, p)
    dem = c.detector_error_model(decompose_errors=True)
    co = c.get_detector_coordinates()
    G = dem_graph(dem)
    b = edge_factors(G, pymatching_weights(dem, G), lam)
    env: dict = collections.defaultdict(float)
    for i, e in enumerate(G.edges):
        if len(e) == 2:
            v = tuple(int(x) for x in np.array(co[e[1]]) - np.array(co[e[0]]))
            v = max(v, tuple(-x for x in v))
            env[v] = max(env[v], float(b[i]))
    ys = {s: sorted({co[k][1] for k, t in G.side.items() if t == s}) for s in (0, 1)}
    # star of the detector nearest the centre (translation invariance is checked on three more central detectors)
    cen = np.array([d, d, d // 2])
    nodes = sorted({x for e in G.edges if len(e) == 2 for x in e}, key=lambda k: float(np.sum((np.array(co[k]) - cen) ** 2)))
    inc = collections.defaultdict(list)
    for i, e in enumerate(G.edges):
        for x in e:
            inc[x].append(i)

    def star(k):
        out = {}
        for i in inc[k]:
            e = G.edges[i]
            if len(e) == 2:
                o = e[1] if e[0] == k else e[0]
                out[tuple(int(x) for x in np.array(co[o]) - np.array(co[k]))] = float(b[i])
        return out

    s0 = star(nodes[0])
    invariant = all(star(k).keys() == s0.keys() and max(abs(star(k)[v] - s0[v]) for v in s0) < 1e-12 for k in nodes[1:4])
    return dict(env=dict(env), bmax=float(max(b[i] for i, e in enumerate(G.edges) if len(e) == 1)),
                n0=sum(1 for t in G.side.values() if t == 0), n1=sum(1 for t in G.side.values() if t == 1), ys=ys,
                star=s0, star_invariant=invariant, stats=G.stats)


def tilted(weights: dict, theta: float, symmetric: bool = True) -> tuple[float, float, float]:
    """(rho, s, R) of the tilted non-backtracking matrix of the lattice with these offset weights; s = sum of the tilted
    weights of the arcs at a vertex, R = max/min of the right Perron vector. Tilt exp(-theta * row step), row step dy/2."""
    offs = []
    for v, x in weights.items():
        offs.append((v, x))
        if symmetric:
            offs.append((tuple(-y for y in v), x))
    wt = np.array([x * np.exp(-theta * (v[1] // 2)) for v, x in offs])
    n = len(offs)
    B = np.zeros((n, n))
    for i, (v, _) in enumerate(offs):
        for j, (v2, _) in enumerate(offs):
            if v2 != tuple(-y for y in v):
                B[i, j] = wt[j]
    ev, vec = np.linalg.eig(B)
    k = int(np.argmax(abs(ev)))
    r = np.abs(np.real(vec[:, k]))
    return float(abs(ev[k])), float(wt.sum()), float(r.max() / r.min())


def theta_star(weights: dict) -> float:
    if tilted(weights, 0.0)[0] >= 1:
        return 0.0
    lo, hi = 0.0, 1.0
    while tilted(weights, hi)[0] < 1:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tilted(weights, mid)[0] < 1 else (lo, mid)
    return lo


def closed_form(F: dict, d: int, theta: float) -> float:
    rho, s, R = tilted(F["env"], theta)
    if rho >= 1:
        return np.inf
    return F["n0"] * F["bmax"] ** 2 * s * R / (1 - rho) * np.exp(-theta * (d - 2))


def lattice_part(model: str, p: float, ds=(5, 7, 9, 11, 13, 15)) -> dict:
    Fs = {d: finite_data(model, d, p) for d in ds}
    ref = Fs[ds[-1]]
    same_env = all(F["env"].keys() == ref["env"].keys() and max(abs(F["env"][v] - ref["env"][v]) for v in ref["env"]) < 1e-12
                   and abs(F["bmax"] - ref["bmax"]) < 1e-12 for F in Fs.values())
    # (H1) the offsets of Proposition 4.30 (circuit level; a subset for phenomenological noise), each changing y by <= 2;
    # (H2) the two sides are the rows y = 2 and y = 2d - 2 (in either order), each with (d + 1)^2 / 2 detectors
    offsets = {(2, 2, 0), (2, -2, 0), (0, 0, 1), (2, 2, -1), (2, -2, -1), (4, 0, -1)}
    geometry = all(sorted([F["ys"][0], F["ys"][1]]) == [[2.0], [2.0 * d - 2]]
                   and F["n0"] == F["n1"] == (d + 1) ** 2 // 2 and set(F["env"]) <= offsets for d, F in Fs.items())
    bulk = {v: x for v, x in ref["star"].items()}
    th = theta_star(ref["env"])
    rows = []
    for d, F in Fs.items():
        best = min((closed_form(F, d, t), t) for t in np.linspace(0.0, max(th, 1e-9), 201))
        rows.append(dict(d=d, closed_form=best[0], theta=best[1]))
    return dict(p=p, envelope_d_independent=bool(same_env), geometry=bool(geometry),
                bulk_star_translation_invariant=bool(ref["star_invariant"]),
                rho_envelope=tilted(ref["env"], 0.0)[0], rho_bulk=tilted(bulk, 0.0, symmetric=False)[0],
                theta_star=th, Lambda_star=float(np.exp(2 * th)), bmax=ref["bmax"],
                envelope={str(v): x for v, x in ref["env"].items()}, closed=rows)


def p_star(model: str, which: str, lo: float, hi: float, d: int = 9) -> float:
    """Noise level where the lattice growth rate (envelope or bulk) reaches 1."""
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        F = finite_data(model, d, mid)
        r = tilted(F["env"], 0.0)[0] if which == "envelope" else tilted(F["star"], 0.0, symmetric=False)[0]
        lo, hi = (mid, hi) if r < 1 else (lo, mid)
    return float(np.sqrt(lo * hi))


# ------------------------------------------------------------------ (3) simulation
def simulate(model: str, d: int, p: float, shots: int, seed: int = 2026) -> tuple[int, int]:
    c = make_circuit(model, d, p)
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    sampler = c.compile_detector_sampler(seed=seed)
    fails = done = 0
    while done < shots:
        n = min(200_000, shots - done)
        det, obs = sampler.sample(n, separate_observables=True)
        fails += int(np.sum(np.any(m.decode_batch(det) != obs, axis=1)))
        done += n
    return fails, shots


def cp(k: int, n: int, level: float = 0.99) -> tuple[float, float]:
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small distances and fewer shots; no output file")
    a = ap.parse_args()
    t0 = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO,
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    out: dict = dict(meta=dict(script="experiments/certification/circuit_peierls.py", args=vars(a), git_commit=commit,
                               git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               versions=dict(stim=stim.__version__, pymatching=pymatching.__version__,
                                             numpy=np.__version__, scipy=scipy.__version__),
                               theorem="theory/sec4-certification.tex, Theorem 4.28, Propositions 4.29, 4.30"))
    ds = [3, 5, 7] if a.quick else DS
    for model in MODELS:
        print(f"== {model}", flush=True)
        table = []
        for p in PS[model]:
            for d in ds:
                r = peierls_bound(make_circuit(model, d, p).detector_error_model(decompose_errors=True))
                table.append(dict(d=d, p=p, **{k: r[k] for k in ("bound", "lam", "rho_upper")}, **r["stats"]))
                print(f"  d={d:2d} p={p:.1e}: bound {r['bound']:.3e} (lam {r['lam']:.3f}, rho <= {r['rho_upper']:.4f}); "
                      f"split multi {r['stats']['split_multi']}, exclusive multi {r['stats']['exclusive_multi']}", flush=True)
        check(f"{model}: every mechanism has one edge in the observable graph after re-decomposition",
              all(r["split_multi"] == 0 and r["exclusive_multi"] == 0 for r in table))
        lat = []
        for p in PS[model]:
            L = lattice_part(model, p, ds=(5, 7) if a.quick else (5, 7, 9, 11, 13, 15))
            lat.append(L)
            check(f"{model} p={p:.0e}: (H1)-(H2): offsets, envelope and boundary factor independent of d, sides at y = 2 and 2d - 2",
                  L["envelope_d_independent"] and L["geometry"])
            for row in L["closed"]:
                exact = next(r["bound"] for r in table if r["p"] == p and r["d"] == row["d"]) if row["d"] in ds else None
                if exact is not None and np.isfinite(row["closed_form"]):
                    check(f"{model} p={p:.0e} d={row['d']}: closed form >= walk sum", row["closed_form"] >= exact * (1 - 1e-9),
                          f"({row['closed_form']:.2e} >= {exact:.2e})")
            print(f"  p={p:.1e}: rho(envelope) {L['rho_envelope']:.4f}, rho(bulk) {L['rho_bulk']:.4f}, theta* {L['theta_star']:.4f} "
                  f"(Lambda* = exp(2 theta*) = {L['Lambda_star']:.2f})", flush=True)
        ranges = {} if a.quick else dict(
            envelope=p_star(model, "envelope", 1e-4 if model == "circuit" else 1e-3, 1e-2 if model == "circuit" else 5e-2),
            bulk=p_star(model, "bulk", 1e-4 if model == "circuit" else 1e-3, 1e-2 if model == "circuit" else 5e-2))
        if ranges:
            print(f"  range of the lattice bound: p < {ranges['envelope']:.3e} (envelope), bulk growth rate 1 at p = {ranges['bulk']:.3e}")
        sims = []
        for d, p, shots in SIM[model]:
            if a.quick and d > 5:
                continue
            k, n = simulate(model, d, p, shots // (10 if a.quick else 1))
            lo, hi = cp(k, n)
            bnd = next((r["bound"] for r in table if r["d"] == d and r["p"] == p), None)
            if bnd is None:
                bnd = peierls_bound(make_circuit(model, d, p).detector_error_model(decompose_errors=True))["bound"]
            sims.append(dict(d=d, p=p, fails=k, shots=n, rate=k / n, cp99=[lo, hi], bound=bnd))
            check(f"{model} d={d} p={p:.0e}: bound >= simulated MWPM rate (99% lower limit)", bnd >= lo,
                  f"(MWPM {k}/{n} = {k / n:.2e}, bound {bnd:.2e}, ratio {bnd / max(k / n, 1e-300):.0f})")
        out[model] = dict(table=table, lattice=lat, ranges=ranges, simulation=sims)
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    out["meta"]["failures"] = failures
    if not a.quick:
        path = REPO / "results" / "circuit_peierls.json"
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print("wrote", path.relative_to(REPO))
    print(f"{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)} ({time.time() - t0:.0f}s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
