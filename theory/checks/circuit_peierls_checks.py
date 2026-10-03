"""Checks of the circuit-level Peierls bound (theory sec 4.9: Definition 4.27, Theorem 4.28, Propositions 4.29, 4.30).
Light: about two minutes, < 1 GB.

  N1  Definition 4.27 and Proposition 4.29 (structure): for stim's rotated memory circuit (uniform circuit noise and
      phenomenological noise, d = 3, 5, 7) the observable flags are consistent, every cycle avoiding the boundary vertex is
      even, and after re-decomposition every mechanism has exactly one edge in the observable graph
  N2  Theorem 4.28, step 1 (deterministic): on sampled faults (d = 3, 5) the re-decomposed error X has the sampled
      syndrome and observable; pymatching fails iff X + C has odd observable parity; every cycle of a decomposition of
      X + C into edge-disjoint simple cycles has w(gamma & C) <= w(gamma & X) (up to pymatching's weight rounding)
  N3  Theorem 4.28, steps 2-3: P[w(gamma & X) >= w(gamma)/2] <= prod beta_e(lam) by exact enumeration on random short
      cycles; the sum over odd simple cycles (all simple paths between the sides, enumerated) is <= the non-backtracking
      walk sum (d = 3, two rounds); the burst factor of part (d) by exact enumeration on a short path
  N4  Theorem 4.28 against simulation: the bound is >= the 99% lower limit of the simulated MWPM failure rate (d = 3, 5)
  N5  Proposition 4.30: the envelope and the boundary factor do not depend on d (d = 5, 7, 9), the closed form is >= the
      walk sum, and theta*(p = 1e-3) of the circuit model is about 0.67

Run:  python theory/checks/circuit_peierls_checks.py        Writes results/circuit_peierls_checks.json.  Exit status non-zero on failure.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pymatching
import stim

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "certification"))
from lcd.analysis.circuit_peierls import (_arcs, _parse, dem_graph, edge_factors, peierls_bound,  # noqa: E402
                                          pymatching_weights, walk_sum)
import circuit_peierls as cpx  # noqa: E402

failures: list[str] = []
OUT: dict = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def dem_of(model: str, d: int, p: float, rounds: int | None = None) -> stim.DetectorErrorModel:
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds or d, **cpx.MODELS[model](p))
    return c.detector_error_model(decompose_errors=True)


# ------------------------------------------------------------------ N1
def n1() -> None:
    print("N1  structure of the observable graph")
    rows = []
    for model in cpx.MODELS:
        for d in (3, 5, 7):
            try:
                G = dem_graph(dem_of(model, d, 1e-3))
                ok = G.stats["split_multi"] == 0 and G.stats["exclusive_multi"] == 0
                rows.append(dict(model=model, d=d, **G.stats))
                check(f"N1 {model} d={d}: consistent flags, balanced, one edge per mechanism", ok,
                      f"({G.stats['redecomposed']} of {G.stats['mechanisms']} mechanisms re-decomposed)")
            except (ValueError, NotImplementedError) as exc:
                check(f"N1 {model} d={d}", False, str(exc))
    OUT["N1"] = rows


# ------------------------------------------------------------------ N2
def n2() -> None:
    print("N2  failure gives heavy odd cycles (sampled faults, pymatching)")
    rng = np.random.default_rng(7)
    res = []
    for d, p, shots in ((3, 6e-3, 3000), (5, 6e-3, 1500)):
        dem = dem_of("circuit", d, p)
        G = dem_graph(dem)
        w = pymatching_weights(dem, G)
        eid = G.index
        m = pymatching.Matching.from_detector_error_model(dem)
        mechs = _parse(dem)
        nd = dem.num_detectors
        obs_dets = {x for e in G.edges for x in e}
        # per mechanism: full detector set, observable flip, and its edge in the observable graph (re-decomposition)
        table = []
        for pm, comps in mechs:
            S, L = set(), 0
            for e, o in comps:
                S ^= set(e)
                L ^= o
            Sz = tuple(sorted(x for x in S if x in obs_dets))
            edge = eid.get(Sz) if Sz else None
            if Sz and (edge is None or G.flag[edge] != L):
                edge = -1  # not representable as one edge (does not happen, N1)
            table.append((pm, sorted(S), L, edge))
        probs = np.array([t[0] for t in table])
        bad_repr = sum(t[3] == -1 for t in table)
        fails = heavy_ok = synd_ok = 0
        worst = 0.0
        for _ in range(shots):
            f = np.flatnonzero(rng.random(len(probs)) < probs)
            syn = np.zeros(nd, np.uint8)
            L = 0
            X = np.zeros(len(G.edges), np.uint8)
            for k in f:
                _, S, l, edge = table[k]
                syn[S] ^= 1
                L ^= l
                if edge is not None and edge >= 0:
                    X[edge] ^= 1
            # syndrome on the observable graph from X
            bx = np.zeros(nd, np.uint8)
            for i in np.flatnonzero(X):
                for x in G.edges[i]:
                    bx[x] ^= 1
            obs_idx = np.array(sorted(obs_dets))
            synd_ok += bool((bx[obs_idx] == syn[obs_idx]).all() and int(G.flag[X == 1].sum() % 2) == L)
            pairs = m.decode_to_edges_array(syn)
            C = np.zeros(len(G.edges), np.uint8)
            for u, v in pairs:
                key = (int(u),) if v < 0 else ((int(v),) if u < 0 else tuple(sorted((int(u), int(v)))))
                if key in eid:
                    C[eid[key]] ^= 1
            pred = int(m.decode(syn)[0])
            fail = pred != L
            Y = X ^ C
            fails += fail
            if fail != bool(G.flag[Y == 1].sum() % 2):
                continue
            # decompose Y into edge-disjoint simple cycles (boundary vertex = -1) and test minimality on each
            adj = collections.defaultdict(set)
            for i in np.flatnonzero(Y):
                e = G.edges[i]
                u, v = (e[0], -1) if len(e) == 1 else e
                adj[u].add((v, i))
                adj[v].add((u, i))
            ok = True
            while any(adj.values()):
                start = next(x for x in adj if adj[x])
                # walk along unused edges; the path is verts[0] -e[0]- verts[1] ...; a repeated vertex closes a cycle
                verts, path, pos, x = [start], [], {start: 0}, start
                while True:
                    y, i = next(iter(adj[x]))
                    adj[x].discard((y, i))
                    adj[y].discard((x, i))
                    path.append(i)
                    if y in pos:
                        j = pos[y]
                        cyc = path[j:]
                        for v in verts[j + 1:]:
                            del pos[v]
                        del path[j:], verts[j + 1:]
                        wc = sum(w[i] for i in cyc if C[i])
                        wx = sum(w[i] for i in cyc if X[i])
                        worst = max(worst, (wc - wx) / sum(w[i] for i in cyc))
                        ok &= wc <= wx + 2e-3 * len(cyc)
                        if not path:
                            break
                        x = y
                        continue
                    pos[y] = len(verts)
                    verts.append(y)
                    x = y
            heavy_ok += ok
        res.append(dict(d=d, p=p, shots=shots, fails=fails, not_one_edge=bad_repr))
        check(f"N2 d={d} p={p:.0e}: every mechanism is one observable-graph edge", bad_repr == 0)
        check(f"N2 d={d} p={p:.0e}: X has the sampled syndrome and observable", synd_ok == shots, f"({synd_ok}/{shots})")
        check(f"N2 d={d} p={p:.0e}: failure iff odd X + C, and every cycle of X + C is heavy", heavy_ok == shots,
              f"({fails} failures in {shots} shots; largest (w(C)-w(X))/w(cycle) = {worst:.1e})")
    OUT["N2"] = res


# ------------------------------------------------------------------ N3
def n3() -> None:
    print("N3  Chernoff step, simple cycles vs non-backtracking walks, burst factor")
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(300):
        k = int(rng.integers(2, 12))
        q = rng.uniform(1e-3, 0.3, k)
        w = np.log((1 - rng.uniform(1e-3, 0.3, k)) / rng.uniform(1e-3, 0.3, k)).clip(0)
        lam = rng.uniform(0.05, 1.0)
        exact = 0.0
        for xs in itertools.product((0, 1), repeat=k):
            xs = np.array(xs)
            if (w * xs).sum() >= w.sum() / 2:
                exact += np.prod(np.where(xs == 1, q, 1 - q))
        bound = np.prod(np.exp(-lam * w) * (1 - q + q * np.exp(2 * lam * w)))
        worst = max(worst, exact / bound)
    check("N3 P[w(gamma & X) >= w(gamma)/2] <= prod beta_e(lam), 300 random cycles", worst <= 1 + 1e-12, f"(max ratio {worst:.3f})")

    dem = dem_of("circuit", 3, 2e-3, rounds=2)
    G = dem_graph(dem)
    beta = edge_factors(G, pymatching_weights(dem, G), 0.5)
    walk, _ = walk_sum(G, beta, _arcs(G))
    adj = collections.defaultdict(list)
    for i, e in enumerate(G.edges):
        if len(e) == 2:
            adj[e[0]].append((e[1], i))
            adj[e[1]].append((e[0], i))
    bedge = {e[0]: i for i, e in enumerate(G.edges) if len(e) == 1}
    total, npaths = 0.0, 0

    def dfs(x, visited, weight):
        nonlocal total, npaths
        if G.side.get(x) == 1:
            total += weight * beta[bedge[x]]
            npaths += 1
        for y, i in adj[x]:
            if y not in visited:
                visited.add(y)
                dfs(y, visited, weight * beta[i])
                visited.discard(y)

    for u, s in G.side.items():
        if s == 0:
            dfs(u, {u}, beta[bedge[u]])
    check("N3 sum over odd simple cycles <= non-backtracking walk sum (d = 3, two rounds)", total <= walk * (1 + 1e-12),
          f"({npaths} simple paths: {total:.4e} <= {walk:.4e}, ratio {walk / total:.2f})")
    OUT["N3"] = dict(chernoff_max_ratio=worst, simple_paths=npaths, simple_sum=total, walk_sum=walk)

    # burst factor (part (d)): path of 6 edges, regions of 2 consecutive edges, each active with prob rho_z;
    # an active region makes its edges fair coins; compare with prod beta_e * exp(|gamma| rho_bar (kappa^V - 1)/V)
    worst_b = 0.0
    for _ in range(100):
        k, V = 6, 2
        q = rng.uniform(1e-3, 0.2, k)
        w = np.log((1 - q) / q)
        rz = rng.uniform(0, 0.05, k - 1)
        lam = rng.uniform(0.2, 0.6)
        a = np.exp(2 * lam * w)
        beta_e = np.exp(-lam * w) * (1 - q + q * a)
        kappa = (np.exp(-lam * w) * (1 + a) / 2) / beta_e
        rho_bar = max((rz[i - 1] if i > 0 else 0.0) + (rz[i] if i < k - 1 else 0.0) for i in range(k))
        with np.errstate(over="ignore"):
            bound = np.prod(beta_e) * np.exp(k * rho_bar * (kappa.max() ** V - 1) / V)
        exact = 0.0
        for act in itertools.product((0, 1), repeat=k - 1):
            pa = np.prod(np.where(np.array(act) == 1, rz, 1 - rz))
            hit = np.zeros(k, bool)
            for z, on in enumerate(act):
                if on:
                    hit[z] = hit[z + 1] = True
                    # edges z, z+1
            qq = np.where(hit, 0.5, q)
            for xs in itertools.product((0, 1), repeat=k):
                xs = np.array(xs)
                if (w * xs).sum() >= w.sum() / 2:
                    exact += pa * np.prod(np.where(xs == 1, qq, 1 - qq))
        worst_b = max(worst_b, exact / bound)
    check("N3 burst factor of Theorem 4.28(d) (exact enumeration, 100 random paths)", worst_b <= 1 + 1e-12,
          f"(max ratio {worst_b:.3f})")
    OUT["N3"]["burst_max_ratio"] = worst_b


# ------------------------------------------------------------------ N4
def n4() -> None:
    print("N4  bound against simulated MWPM failure rates")
    rows = []
    for model, d, p, shots in (("circuit", 3, 1e-3, 200_000), ("circuit", 3, 3e-3, 100_000), ("circuit", 5, 2e-3, 300_000),
                               ("phenomenological", 3, 5e-3, 100_000), ("phenomenological", 5, 1e-2, 200_000)):
        k, n = cpx.simulate(model, d, p, shots)
        lo, _ = cpx.cp(k, n)
        b = peierls_bound(dem_of(model, d, p))["bound"]
        rows.append(dict(model=model, d=d, p=p, fails=k, shots=n, bound=b))
        check(f"N4 {model} d={d} p={p:.0e}: bound >= 99% lower limit", b >= lo, f"(MWPM {k / n:.2e}, bound {b:.2e})")
    OUT["N4"] = rows


# ------------------------------------------------------------------ N5
def n5() -> None:
    print("N5  lattice form for every d")
    L = cpx.lattice_part("circuit", 1e-3, ds=(5, 7, 9))
    check("N5 (H1)-(H2) for d = 5, 7, 9: the twelve offsets, envelope and boundary factor independent of d, sides at "
          "y = 2 and 2d - 2 with (d + 1)^2 / 2 detectors each", L["envelope_d_independent"] and L["geometry"]
          and len(L["envelope"]) == 6)
    for row in L["closed"]:
        exact = peierls_bound(dem_of("circuit", row["d"], 1e-3), lams=[0.5])["bound"]
        check(f"N5 d={row['d']}: closed form >= walk sum at lam = 1/2", row["closed_form"] >= exact,
              f"({row['closed_form']:.2e} >= {exact:.2e})")
    check("N5 theta*(1e-3) of the circuit model in [0.66, 0.68]", 0.66 <= L["theta_star"] <= 0.68, f"({L['theta_star']:.4f})")
    OUT["N5"] = {k: L[k] for k in ("rho_envelope", "rho_bulk", "theta_star", "Lambda_star", "closed")}


def main() -> int:
    for f in (n1, n2, n3, n4, n5):
        f()
    OUT["failures"] = failures
    (REPO / "results" / "circuit_peierls_checks.json").write_text(json.dumps(OUT, indent=1, default=float) + "\n")
    print("ALL PASS" if not failures else "FAILURES: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
