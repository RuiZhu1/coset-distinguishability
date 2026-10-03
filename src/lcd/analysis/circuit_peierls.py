"""Circuit-level Peierls bound for minimum-weight matching on a detector error model (theory, Theorem 4.28).

Input: a stim detector error model with one logical observable, decomposed into graphlike components
(``circuit.detector_error_model(decompose_errors=True)``), and the decoder's edge weights (by default those of
``pymatching.Matching.from_detector_error_model``). Output: an upper bound on the logical failure probability of an
exact minimum-weight decoder with those weights, hence of the maximum-likelihood decoder.

The bound (Theorem 4.28):
  * the observable graph is the part of the decoding graph whose components (without the boundary vertex b) carry an
    edge that flips the observable; every mechanism is written as a set of its edges with the same detectors and the same
    observable flip, as one edge whenever such an edge exists ("re-decomposition");
  * if every cycle avoiding b has even observable parity (checked; the graph is "balanced"), the boundary detectors
    split into two sides and the odd cycles are the paths between the sides closed through b;
  * for lam >= 0 an odd simple cycle gamma contributes at most prod_{e in gamma} beta_e(lam), with
        beta_e(lam) = exp(-lam w_e) (1 - q_e + q_e exp(2 lam w_e)) * prod_{split m on e} (1 + p_m^(1/k_m)(exp(2 lam w_e) - 1)),
    q_e the flip probability of edge e from the mechanisms that have no other edge on any odd cycle with e, and the
    remaining (split) mechanisms m with k_m edges shared out by the splitting lemma;
  * pymatching minimizes rounded weights; every beta_e is multiplied by exp(lam delta), delta the rounding slack (4.28(e));
  * summing over non-backtracking walks from one side to the other gives start^T (I - B)^{-1} end, finite when the
    spectral radius of the weighted non-backtracking matrix B is < 1.

    from lcd.analysis.circuit_peierls import peierls_bound
    res = peierls_bound(circuit.detector_error_model(decompose_errors=True))
    res["bound"], res["lam"], res["rho_upper"]
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Edge = tuple  # (u,) for an edge to the boundary vertex, (u, v) with u < v otherwise


@dataclass
class DemGraph:
    edges: list                     # edges of the observable graph
    flag: np.ndarray                # observable flag of each edge (0/1)
    q: np.ndarray                   # flip probability of each edge from its exclusive mechanisms
    split: list                     # per edge: the shares p_m^(1/k_m) of mechanisms that are split
    side: dict                      # boundary detector -> side 0/1 (odd cycles join the two sides)
    stats: dict = field(default_factory=dict)

    @property
    def index(self) -> dict:
        return {e: i for i, e in enumerate(self.edges)}


def _parse(dem) -> list:
    """[(p, [(detectors, observable parity), ...]), ...] with one entry per graphlike component."""
    if dem.num_observables != 1:
        raise ValueError("one logical observable expected")
    import stim
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps, dets, o = [], [], 0
        for t in inst.targets_copy() + [stim.DemTarget.separator()]:
            if t.is_separator():
                comps.append((tuple(sorted(dets)), o))
                dets, o = [], 0
            elif t.is_logical_observable_id():
                o ^= 1
            else:
                dets.append(t.val)
        p = inst.args_copy()[0]
        if not 0 <= p <= 0.5:
            raise ValueError(f"mechanism probability {p} outside [0, 1/2]")
        out.append((p, comps))
    return out


def dem_graph(dem) -> DemGraph:
    """The observable graph of a decomposed detector error model, with re-decomposed mechanisms (Definition 4.27)."""
    mechs = _parse(dem)
    flag: dict = {}
    for _, comps in mechs:
        for e, o in comps:
            if len(e) not in (1, 2):
                raise ValueError(f"component {e} is not graphlike (use decompose_errors=True)")
            if flag.setdefault(e, o) != o:
                raise ValueError(f"edge {e} carries both observable flags")
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in flag:
        find(e[0])
        if len(e) == 2:
            parent[find(e[0])] = find(e[1])
    roots = {find(e[0]) for e, o in flag.items() if o}
    in_obs = lambda e: find(e[0]) in roots  # noqa: E731
    edges = sorted(e for e in flag if in_obs(e))
    eid = {e: i for i, e in enumerate(edges)}

    # balance and sides
    adj = collections.defaultdict(list)
    for e in edges:
        if len(e) == 2:
            adj[e[0]].append((e[1], flag[e]))
            adj[e[1]].append((e[0], flag[e]))
    phi: dict = {}
    for s in {x for e in edges for x in e}:
        if s in phi:
            continue
        phi[s], stack = 0, [s]
        while stack:
            x = stack.pop()
            for y, o in adj[x]:
                if y not in phi:
                    phi[y] = phi[x] ^ o
                    stack.append(y)
                elif phi[y] != phi[x] ^ o:
                    raise NotImplementedError("an odd cycle avoids the boundary vertex (unbalanced graph)")
    side = {e[0]: flag[e] ^ phi[e[0]] for e in edges if len(e) == 1}

    # re-decomposition
    lists = collections.defaultdict(list)
    split = [[] for _ in edges]
    n_re = n_excl = n_split = 0
    for p, comps in mechs:
        zc = [(e, o) for e, o in comps if in_obs(e)]
        if not zc:
            continue
        S, L = set(), 0
        for e, o in zc:
            S ^= set(e)
            L ^= o
        key = tuple(sorted(S))
        if not S:
            if L:
                raise ValueError("a mechanism flips the observable without any detector")
            continue
        if key in eid and flag[key] == L:
            lists[eid[key]].append(p)
            n_re += len(zc) > 1
            continue
        if len(zc) == 1:
            lists[eid[zc[0][0]]].append(p)
            continue
        es = [eid[e] for e, _ in zc]
        if all(len(e) == 1 for e, _ in zc) and len({side[e[0]] for e, _ in zc}) == 1:
            for i in es:                       # boundary edges on one side: never two on an odd simple cycle
                lists[i].append(p)
            n_excl += 1
        else:
            for i in es:
                split[i].append(p ** (1 / len(es)))
            n_split += 1
    q = np.zeros(len(edges))
    for i, ps in lists.items():
        q[i] = 0.5 * (1 - np.prod([1 - 2 * x for x in ps]))
    stats = dict(mechanisms=len(mechs), edges=len(edges), boundary_edges=sum(len(e) == 1 for e in edges),
                 redecomposed=n_re, exclusive_multi=n_excl, split_multi=n_split)
    return DemGraph(edges=edges, flag=np.array([flag[e] for e in edges], np.int8), q=q, split=split, side=side,
                    stats=stats)


def pymatching_weights(dem, G: DemGraph) -> np.ndarray:
    """Weights of the observable-graph edges in pymatching's matching graph built from the same model."""
    import pymatching
    m = pymatching.Matching.from_detector_error_model(dem)
    eid = G.index
    w = np.full(len(G.edges), np.nan)
    for u, v, attr in m.edges():
        key = (u,) if v is None else tuple(sorted((u, v)))
        if key in eid:
            w[eid[key]] = attr["weight"]
    if np.isnan(w).any() or (w < 0).any():
        raise ValueError("an observable-graph edge has no nonnegative pymatching weight")
    return w


# pymatching 2.x (sparse_blossom/driver/user_graph.h, user_graph.cc) minimizes the integer weights
# 2 round(kappa w), kappa = (NUM_DISTINCT_WEIGHTS - 1) / max|w| over all edges (kappa = 1 if every weight is an integer).
PYMATCHING_NUM_DISTINCT_WEIGHTS = 1 << 24


def pymatching_rounding_slack(dem) -> float:
    """delta of Theorem 4.28(e): pymatching minimizes weights within delta per edge of its float weights (up to scale)."""
    import pymatching
    if int(pymatching.__version__.split(".")[0]) != 2:
        raise RuntimeError(f"weight discretization checked for pymatching 2.x only, found {pymatching.__version__}")
    ws = np.array([attr["weight"] for _, _, attr in pymatching.Matching.from_detector_error_model(dem).edges()])
    if ws.size == 0 or np.all(np.round(ws) == ws):
        return 0.0
    return float(np.abs(ws).max()) / (2 * (PYMATCHING_NUM_DISTINCT_WEIGHTS - 1))


def edge_factors(G: DemGraph, w: np.ndarray, lam: float, delta: float = 0.0) -> np.ndarray:
    """beta_e(lam) of Theorem 4.28, times exp(lam delta) for a decoder that minimizes weights within delta of w (4.28(e))."""
    a = np.exp(2 * lam * w)
    beta = np.exp(lam * (delta - w)) * (1 - G.q + G.q * a)
    for i, shares in enumerate(G.split):
        for s in shares:
            beta[i] *= 1 + s * (a[i] - 1)
    return beta


@dataclass
class _Arcs:
    arc_edge: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    start_bdry: np.ndarray          # boundary edge index of the arc's tail if on side 0, else -1
    end_bdry: np.ndarray            # boundary edge index of the arc's head if on side 1, else -1


def _arcs(G: DemGraph) -> _Arcs:
    arcs, arc_edge = [], []
    for i, e in enumerate(G.edges):
        if len(e) == 2:
            arcs += [(e[0], e[1]), (e[1], e[0])]
            arc_edge += [i, i]
    out = collections.defaultdict(list)
    for a, (x, _) in enumerate(arcs):
        out[x].append(a)
    rows, cols = [], []
    for a, (x, y) in enumerate(arcs):
        for b in out[y]:
            if arcs[b][1] != x:
                rows.append(a)
                cols.append(b)
    bedge = {e[0]: i for i, e in enumerate(G.edges) if len(e) == 1}
    sb = np.array([bedge[x] if G.side.get(x) == 0 else -1 for x, _ in arcs], dtype=np.int64)
    eb = np.array([bedge[y] if G.side.get(y) == 1 else -1 for _, y in arcs], dtype=np.int64)
    return _Arcs(np.array(arc_edge, np.int64), np.array(rows, np.int64), np.array(cols, np.int64), sb, eb)


def walk_sum(G: DemGraph, beta: np.ndarray, arcs: _Arcs | None = None) -> tuple[float, float]:
    """(bound, rho_upper): the sum over non-backtracking walks between the sides, closed by their boundary edges.

    rho_upper = 1 - 1/max z with z = (I - B)^{-1} 1 is a Collatz-Wielandt upper bound on the spectral radius of B; if z
    is not positive the radius is >= 1 and the bound is infinite.
    """
    A = arcs if arcs is not None else _arcs(G)
    n = len(A.arc_edge)
    if n == 0:
        return 0.0, 0.0
    B = sp.csc_matrix((beta[A.arc_edge[A.cols]], (A.rows, A.cols)), shape=(n, n))
    lu = spla.splu((sp.identity(n, format="csc") - B).tocsc())
    z = lu.solve(np.ones(n))
    if not np.all(z > 0):
        return np.inf, np.nan
    rho_upper = 1 - 1 / z.max()
    start = np.where(A.start_bdry >= 0, beta[np.maximum(A.start_bdry, 0)] * beta[A.arc_edge], 0.0)
    end = np.where(A.end_bdry >= 0, beta[np.maximum(A.end_bdry, 0)], 0.0)
    return float(start @ lu.solve(end)), float(rho_upper)


def peierls_bound(dem, weights: np.ndarray | None = None, lams=None, G: DemGraph | None = None,
                  delta: float | None = None) -> dict:
    """Minimize the bound of Theorem 4.28 over lam (the bound is log-convex in lam: coarse grid, then golden section).

    Default: pymatching's weights and its rounding slack delta (Theorem 4.28(e)). With explicit ``weights`` the decoder is
    assumed to minimize exactly those weights (delta = 0) unless ``delta`` is given.
    """
    G = G if G is not None else dem_graph(dem)
    if weights is None:
        w = pymatching_weights(dem, G)
        delta = pymatching_rounding_slack(dem) if delta is None else delta
    else:
        w, delta = weights, (0.0 if delta is None else delta)
    A = _arcs(G)

    def f(lam):
        return walk_sum(G, edge_factors(G, w, lam, delta), A)

    grid = list(lams) if lams is not None else [0.1 * k for k in range(1, 10)]
    vals = [f(l)[0] for l in grid]
    k = int(np.argmin(vals))
    if lams is None and np.isfinite(vals[k]):
        lo, hi = grid[max(k - 1, 0)] if k > 0 else 0.01, grid[min(k + 1, len(grid) - 1)] if k < len(grid) - 1 else 0.99
        g = (sqrt(5) - 1) / 2
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f1, f2 = f(x1)[0], f(x2)[0]
        for _ in range(25):
            if f1 <= f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = f(x1)[0]
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = f(x2)[0]
        lam = x1 if f1 <= f2 else x2
    else:
        lam = grid[k]
    val, rho = f(lam)
    return dict(bound=val, lam=float(lam), rho_upper=rho, delta=delta, stats=G.stats)
