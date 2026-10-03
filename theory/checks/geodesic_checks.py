"""Checks of the geodesic refinement of the circuit-level Peierls bound (theory sec 4.9, Theorem 4.32).
Light: a few minutes, < 1 GB.

  G1  Lemma (deterministic): on sampled faults, every maximal run of the correction inside X + C is a geodesic of the
      integer weights pymatching minimizes (stim rotated memory circuit, d = 3, 5, p = 6e-3, 1e-2). This also tests the
      emulation of pymatching's weight discretization.
  G2  Relaxation (exact enumeration): on small detector error models, the computable alternating-run sum is >= the
      exact sum over odd simple cycles and subsets S whose complement runs are geodesic (wrap-around run through the
      boundary vertex required to be geodesic as a whole), which is <= the sum of Theorem 4.28(a).
  G3  Against simulation: the bound is >= the 99% upper... more precisely, >= the lower likelihood limit of the sampled
      pymatching failure rate (sinter), for stim's rotated memory circuit, d = 3, 5, 7, several p.

Run:  python theory/checks/geodesic_checks.py      Writes results/geodesic_checks.json.  Exit status non-zero on failure.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.circuit_peierls import _parse, dem_graph, peierls_bound  # noqa: E402
from lcd.analysis.circuit_peierls_geodesic import _Geo, geodesic_bound, pymatching_integer_weights  # noqa: E402

failures: list[str] = []
OUT: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def surface(d: int, p: float, rounds: int | None = None) -> stim.DetectorErrorModel:
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds or d,
                               after_clifford_depolarization=p, after_reset_flip_probability=p,
                               before_measure_flip_probability=p, before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True)


# ------------------------------------------------------------------ G1
def g1() -> None:
    print("G1  correction runs are geodesics of pymatching's integer weights (sampled faults)")
    rng = np.random.default_rng(11)
    rows = []
    for d, p, shots in ((3, 6e-3, 4000), (5, 6e-3, 3000), (5, 1e-2, 1500)):
        dem = surface(d, p)
        G = dem_graph(dem)
        wt, _ = pymatching_integer_weights(dem, G)
        geo = _Geo(G, wt)
        eid = G.index
        m = pymatching.Matching.from_detector_error_model(dem)
        obs_dets = {x for e in G.edges for x in e}
        table = []
        for pm, comps in _parse(dem):
            S, L = set(), 0
            for e, o in comps:
                S ^= set(e)
                L ^= o
            Sz = tuple(sorted(x for x in S if x in obs_dets))
            table.append((pm, sorted(S), L, eid.get(Sz) if Sz else None))
        probs = np.array([t[0] for t in table])

        def vtx(e):
            return (geo.vid[e[0]], geo.b) if len(e) == 1 else (geo.vid[e[0]], geo.vid[e[1]])

        fails = runs = bad = 0
        for _ in range(shots):
            f = np.flatnonzero(rng.random(len(probs)) < probs)
            syn = np.zeros(dem.num_detectors, np.uint8)
            X = np.zeros(len(G.edges), np.uint8)
            L = 0
            for k in f:
                _, S, l, edge = table[k]
                syn[S] ^= 1
                L ^= l
                if edge is not None:
                    X[edge] ^= 1
            C = np.zeros(len(G.edges), np.uint8)
            for u, v in m.decode_to_edges_array(syn):
                key = (int(u),) if v < 0 else ((int(v),) if u < 0 else tuple(sorted((int(u), int(v)))))
                if key in eid:
                    C[eid[key]] ^= 1
            if int(m.decode(syn)[0]) == L:
                continue
            fails += 1
            cadj = collections.defaultdict(list)
            for i in np.flatnonzero(C & ~X & 1):
                a, b = vtx(G.edges[i])
                cadj[a].append((b, i))
                cadj[b].append((a, i))
            seen = set()
            for start in list(cadj):
                if len(cadj[start]) == 2 and start != geo.b:
                    continue
                for nb, i in cadj[start]:
                    if i in seen:
                        continue
                    seen.add(i)
                    w, cur, ci = wt[i], nb, i
                    while len(cadj[cur]) == 2 and cur != geo.b:
                        (n1, i1), (n2, i2) = cadj[cur]
                        nxt, ni = (n2, i2) if i1 == ci else (n1, i1)
                        if ni in seen:
                            break
                        seen.add(ni)
                        w += wt[ni]
                        ci, cur = ni, nxt
                    runs += 1
                    bad += not abs(w - geo.D[start, cur]) < 0.5
        rows.append(dict(d=d, p=p, shots=shots, failures=fails, runs=runs, non_geodesic=bad))
        check(f"G1 d={d} p={p:g}: every correction run is a geodesic", bad == 0 and runs > 0,
              f"({fails} failures, {runs} runs)")
    OUT["G1"] = rows


# ------------------------------------------------------------------ G2
def exact_restricted(dem, lam: float) -> tuple[float, float, float, int]:
    """(relaxed bound, exact restricted sum, Theorem 4.28(a) sum, number of odd simple cycles) at weights w~."""
    G = dem_graph(dem)
    wt, kappa = pymatching_integer_weights(dem, G)
    geo = _Geo(G, wt)
    what = wt / (2 * kappa)
    x = G.q * np.exp(lam * what)
    c = (1 - G.q) * np.exp(-lam * what)
    nbr = collections.defaultdict(list)
    for k in range(len(geo.E)):
        u, v, e = geo.U[k], geo.W[k], geo.E[k]
        nbr[u].append((v, e))
        nbr[v].append((u, e))
    b = geo.b
    cycles = []

    def dfs(path_v, path_e):
        u = path_v[-1]
        for v, e in nbr[u]:
            if v == b and len(path_v) >= 2 and geo.side[u] == 1:
                cycles.append((path_v + [b], path_e + [e]))
            elif v != b and v not in path_v:
                dfs(path_v + [v], path_e + [e])

    for u, e in nbr[b]:
        if geo.side[u] == 0:
            dfs([b, u], [e])
    exact = old = 0.0
    for verts, es in cycles:
        ell = len(es)
        old += np.prod(x[es] + c[es])
        for mask in range(1 << ell):
            inS = [(mask >> i) & 1 for i in range(ell)]
            if all(inS):
                exact += np.prod(x[es])
                continue
            # rotate so that position 0 starts a run of the complement (or an S edge)
            ok = True
            i = 0
            # find runs of complement cyclically
            start = next(k for k in range(ell) if inS[k] and not inS[(k + 1) % ell]) if any(inS) else None
            if start is None:        # all complement: the whole cycle would be a run; weight w~(C) <= 0 impossible
                continue
            k = (start + 1) % ell
            steps = 0
            while steps < ell:
                if inS[k]:
                    k = (k + 1) % ell
                    steps += 1
                    continue
                a = verts[k]                         # run starts at vertex verts[k] (edge es[k] goes verts[k] -> verts[k+1])
                wsum = 0.0
                while not inS[k] and steps < ell:
                    wsum += wt[es[k]]
                    k = (k + 1) % ell
                    steps += 1
                z = verts[k] if k != 0 else verts[0]
                if not abs(wsum - geo.D[a, z]) < 0.5:
                    ok = False
                    break
            if ok:
                exact += np.prod([x[es[t]] if inS[t] else c[es[t]] for t in range(ell)])
            i += 1
    relaxed = geodesic_bound(dem, lams=[lam])["bound"]
    return relaxed, exact, old, len(cycles)


def g2() -> None:
    print("G2  relaxed sum >= exact restricted sum <= Theorem 4.28(a) sum (exact enumeration)")
    rows = []
    cases = [("repetition d=3 r=2", stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2,
                                                             before_round_data_depolarization=0.02,
                                                             before_measure_flip_probability=0.02)),
             ("repetition d=5 r=1", stim.Circuit.generated("repetition_code:memory", distance=5, rounds=1,
                                                             before_round_data_depolarization=0.01,
                                                             before_measure_flip_probability=0.03)),
             ("surface d=3 r=1", stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1,
                                                         before_round_data_depolarization=0.01,
                                                         before_measure_flip_probability=0.01))]
    for name, circ in cases:
        dem = circ.detector_error_model(decompose_errors=True)
        for lam in (0.3, 0.5):
            relaxed, exact, old, n = exact_restricted(dem, lam)
            rows.append(dict(case=name, lam=lam, cycles=n, relaxed=relaxed, exact=exact, theorem_4_28a=old))
            check(f"G2 {name} lam={lam}: exact <= relaxed and exact <= 4.28(a)", exact <= relaxed * (1 + 1e-12)
                  and exact <= old * (1 + 1e-12),
                  f"({n} cycles; exact {exact:.4e}, relaxed {relaxed:.4e}, 4.28(a) {old:.4e})")
    OUT["G2"] = rows


# ------------------------------------------------------------------ G3
def g3() -> None:
    print("G3  bound >= sampled pymatching failure rate (sinter)")
    import sinter
    rows = []
    tasks = []
    for d, p in ((3, 1e-3), (3, 3e-3), (5, 2e-3), (5, 3e-3), (7, 3e-3)):
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
                                   after_clifford_depolarization=p, after_reset_flip_probability=p,
                                   before_measure_flip_probability=p, before_round_data_depolarization=p)
        tasks.append(sinter.Task(circuit=c, json_metadata=dict(d=d, p=p)))
    stats = sinter.collect(num_workers=2, tasks=tasks, decoders=["pymatching"], max_shots=400_000, max_errors=400)
    for s in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        d, p = s.json_metadata["d"], s.json_metadata["p"]
        fit = sinter.fit_binomial(num_shots=s.shots, num_hits=s.errors, max_likelihood_factor=1e3)
        bnd = geodesic_bound(surface(d, p))["bound"]
        old = peierls_bound(surface(d, p))["bound"]
        rows.append(dict(d=d, p=p, shots=s.shots, errors=s.errors, rate=s.errors / s.shots, low=fit.low, high=fit.high,
                         geodesic=bnd, theorem_4_28=old))
        check(f"G3 d={d} p={p:g}: geodesic bound >= sampled rate", bnd >= fit.low,
              f"(rate {s.errors / s.shots:.2e}, bound {bnd:.2e} = {bnd / max(s.errors / s.shots, 1e-300):.0f}x; "
              f"Theorem 4.28 {old:.2e})")
    OUT["G3"] = rows


def main() -> int:
    for f in (g1, g2, g3):
        f()
    OUT["failures"] = failures
    (REPO / "results" / "geodesic_checks.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
    print("ALL PASS" if not failures else f"FAILED: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
