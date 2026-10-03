"""Geodesic refinement of the circuit-level Peierls bound (draft; theory note in preparation).

Setting of Theorem 4.28 with every mechanism exclusive (no split mechanisms), which holds for stim's rotated memory
circuits after re-decomposition and for Google's published hardware DEMs. On failure there is an odd simple cycle
gamma through b in X + C, where X is the error and C the decoder's correction (Theorem 4.28, proof). New ingredient:

  every maximal run of gamma inside C is a geodesic of the decoder's weights between its end points.

Indeed, if sigma is such a run from u to v and sigma' is any u-v path, C + sigma + sigma' has the same syndrome, so
minimality gives w~(sigma) <= w~(sigma'). Here w~ are the integer weights pymatching actually minimizes,
2 round(kappa w) (Theorem 4.28(e)); using them directly removes the rounding slack.

With S = gamma ∩ X and w^ = w~ / (2 kappa) (proportional to w~, close to w), exclusivity gives
  P[S ⊆ X, gamma \\ S ∩ X = ∅] = prod_{e in S} q_e prod_{e in gamma \\ S} (1 - q_e),
and the Chernoff step for w~(gamma \\ S) <= w~(S) gives, for lam >= 0,
  P[fail] <= sum over (gamma, S) with C-runs geodesic of prod_{S} x_e prod_{gamma \\ S} c_e,
  x_e = q_e exp(lam w^_e),   c_e = (1 - q_e) exp(-lam w^_e)      (x_e + c_e = beta_e of Theorem 4.28).
The sum over all subsets S (Theorem 4.28) is thereby restricted to those whose complement in gamma is a union of
geodesics. Relaxation to a computable sum: cut gamma at b, split it into alternating maximal runs; X-runs are
non-backtracking walks (factor x_e per edge), C-runs are geodesics (factor c_e per edge; sub-paths of geodesics are
geodesic, so the run through b may be cut there). Summing over all alternating sequences (not only simple cycles)
gives an upper bound, evaluated with vertex-to-vertex kernels K_X (non-backtracking walk sums) and K_C (geodesic sums).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra

from lcd.analysis.circuit_peierls import PYMATCHING_NUM_DISTINCT_WEIGHTS, DemGraph, dem_graph, pymatching_weights


def pymatching_integer_weights(dem, G: DemGraph) -> tuple[np.ndarray, float]:
    """(w~, kappa): the integer weights 2 round(kappa w) pymatching 2 minimizes on the edges of G (Theorem 4.28(e))."""
    import pymatching
    if int(pymatching.__version__.split(".")[0]) != 2:
        raise RuntimeError(f"weight discretization checked for pymatching 2.x only, found {pymatching.__version__}")
    w = pymatching_weights(dem, G)
    allw = np.array([attr["weight"] for _, _, attr in pymatching.Matching.from_detector_error_model(dem).edges()])
    kappa = 1.0 if np.all(np.round(allw) == allw) else (PYMATCHING_NUM_DISTINCT_WEIGHTS - 1) / float(np.abs(allw).max())
    wt = 2 * np.floor(w * kappa + 0.5)          # C++ std::round = half away from zero; w >= 0 here
    if (wt <= 0).any():
        raise NotImplementedError("zero-weight edge in the observable graph")
    return wt, kappa


class _Geo:
    """Graph data independent of lam: vertices, arcs, integer distances, geodesic-DAG masks."""

    def __init__(self, G: DemGraph, wt: np.ndarray):
        if any(G.split[i] for i in range(len(G.edges))):
            raise NotImplementedError("split (non-exclusive) mechanisms: use circuit_peierls.peierls_bound")
        dets = sorted({x for e in G.edges for x in e})
        self.vid = {x: i for i, x in enumerate(dets)}
        V = self.V = len(dets)
        self.b = V                                   # boundary vertex index
        U, W, E = [], [], []
        for i, e in enumerate(G.edges):
            u = self.vid[e[0]]
            v = self.b if len(e) == 1 else self.vid[e[1]]
            U.append(u); W.append(v); E.append(i)
        self.U, self.W, self.E = np.array(U), np.array(W), np.array(E)
        self.wt = wt
        A = sp.coo_matrix((np.r_[wt, wt], (np.r_[self.U, self.W], np.r_[self.W, self.U])), shape=(V + 1, V + 1)).tocsr()
        self.D = dijkstra(A, directed=False)          # exact: integer weights, sums < 2^53
        # directed edges p -> v (both orientations), interior only (b excluded as intermediate)
        inner = self.W != self.b
        self.dp = np.r_[self.U[inner], self.W[inner]]
        self.dv = np.r_[self.W[inner], self.U[inner]]
        self.de = np.r_[self.E[inner], self.E[inner]]
        # boundary edges: vertex -> edge index, side
        self.bedge = {self.U[k]: self.E[k] for k in range(len(self.E)) if self.W[k] == self.b}
        self.side = np.full(V, -1)
        for x, s in G.side.items():
            self.side[self.vid[x]] = s
        # non-backtracking arcs of G_o - b
        tails, heads, aedge = self.dp, self.dv, self.de
        out = [[] for _ in range(V)]
        for a, t in enumerate(tails):
            out[t].append(a)
        rows, cols = [], []
        for a in range(len(tails)):
            for c in out[heads[a]]:
                if heads[c] != tails[a]:
                    rows.append(a); cols.append(c)
        self.nb_rows, self.nb_cols = np.array(rows, np.int64), np.array(cols, np.int64)
        self.arc_tail, self.arc_head, self.arc_edge = tails, heads, aedge

    def geodesic_sums(self, c: np.ndarray, sources: np.ndarray, first: dict | None = None) -> np.ndarray:
        """F[k, v] = sum over interior geodesics from sources[k] to v of prod c_e (F[k, source] = 1).

        With ``first`` = {u: factor} the paths start at b, take the edge b-u (u in first) as their first edge, and
        must be geodesic from b; then ``sources`` must be [b] and F[0, v] excludes the empty path."""
        V = self.V
        if first is None and len(sources) > 256:                         # bound the memory: chunks of sources
            return np.vstack([self.geodesic_sums(c, sources[k:k + 256]) for k in range(0, len(sources), 256)])
        Ds = self.D[sources][:, :V]                                      # distances from each source
        # mask[k, j]: directed edge j lies on a geodesic from source k
        mask = np.isclose(Ds[:, self.dp] + self.wt[self.de], Ds[:, self.dv], rtol=0, atol=0.5)
        coef = mask * c[self.de][None, :]
        F = np.zeros((len(sources), V))
        if first is None:
            F[np.arange(len(sources)), sources] = 1.0
        else:
            for u, f in first.items():
                if np.isclose(self.D[self.b, u], self.wt[self.bedge[u]], atol=0.5):
                    F[0, u] += f
        total = F.copy()
        frontier = F
        for _ in range(4 * V + 4):
            nxt = np.zeros_like(F)
            np.add.at(nxt.T, self.dv, (frontier[:, self.dp] * coef).T)
            if not nxt.any():
                break
            total += nxt
            frontier = nxt
        else:
            raise RuntimeError("geodesic DAG did not terminate")
        return total

    def kernels(self, lam: float, q: np.ndarray, what: np.ndarray):
        x = q * np.exp(lam * what)
        c = (1 - q) * np.exp(-lam * what)
        V = self.V
        # X-runs: non-backtracking walks of length >= 1 between interior vertices
        n = len(self.arc_tail)
        Bx = sp.csc_matrix((x[self.arc_edge[self.nb_cols]], (self.nb_rows, self.nb_cols)), shape=(n, n))
        lu = spla.splu((sp.identity(n, format="csc") - Bx).tocsc())
        z = lu.solve(np.ones(n))
        if not np.all(z > 0):
            return None
        Out = sp.csc_matrix((np.ones(n), (np.arange(n), self.arc_head)), shape=(n, V)).toarray()
        In = sp.csr_matrix((x[self.arc_edge], (self.arc_tail, np.arange(n))), shape=(V, n))
        KX = In @ lu.solve(Out)                                          # V x V
        # C-runs: geodesic sums between distinct interior vertices
        KC = self.geodesic_sums(c, np.arange(V)) - np.eye(V)
        # starts and ends at b
        side0 = {u: None for u in range(V) if self.side[u] == 0 and u in self.bedge}
        side1 = {u: None for u in range(V) if self.side[u] == 1 and u in self.bedge}
        s0 = np.zeros(V); s1 = np.zeros(V)
        for u in side0:
            s0[u] = x[self.bedge[u]]
        for u in side1:
            s1[u] = x[self.bedge[u]]
        sX = s0 + s0 @ KX                                                # X-run from b ending at v
        eX = s1 + KX @ s1                                                # X-run from v ending at b
        sC = self.geodesic_sums(c, np.array([self.b]), {u: c[self.bedge[u]] for u in side0})[0]
        eC = self.geodesic_sums(c, np.array([self.b]), {u: c[self.bedge[u]] for u in side1})[0]
        return KX, KC, sX, sC, eX, eC, s1


def geodesic_bound(dem, lams=None, G: DemGraph | None = None) -> dict:
    """Upper bound on the per-shot failure probability of pymatching (hence of ML) with the geodesic refinement."""
    G = G if G is not None else dem_graph(dem)
    wt, kappa = pymatching_integer_weights(dem, G)
    geo = _Geo(G, wt)
    what = wt / (2 * kappa)
    V = geo.V

    def f(lam):
        k = geo.kernels(lam, G.q, what)
        if k is None:
            return np.inf, np.nan
        KX, KC, sX, sC, eX, eC, s1 = k
        M = np.zeros((2 * V, 2 * V))
        M[:V, V:] = KC                                                  # X-ended -> C-ended
        M[V:, :V] = KX                                                  # C-ended -> X-ended
        lu = sla.lu_factor(np.eye(2 * V) - M)
        zz = sla.lu_solve(lu, np.ones(2 * V))
        if not np.all(zz > 0):
            return np.inf, np.nan
        rho_upper = 1 - 1 / zz.max()
        start = np.r_[sX, sC]
        end = np.r_[eC, eX]
        total = float(sX @ s1) + float(start @ sla.lu_solve(lu, end))
        return total, rho_upper

    grid = list(lams) if lams is not None else [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    vals = [f(l) for l in grid]
    k = int(np.argmin([v[0] for v in vals]))
    return dict(bound=vals[k][0], lam=float(grid[k]), rho_upper=vals[k][1], kappa=kappa, stats=G.stats)


def best_bound(dem, G: DemGraph | None = None) -> dict:
    """The smaller of Theorem 4.28 and Theorem 4.32 (both are upper bounds); ``method`` says which one."""
    from lcd.analysis.circuit_peierls import peierls_bound
    G = G if G is not None else dem_graph(dem)
    r = dict(peierls_bound(dem, G=G), method="4.28")
    try:
        g = geodesic_bound(dem, G=G)
    except NotImplementedError:
        return r
    if g["bound"] < r["bound"]:
        r.update(bound=g["bound"], lam=g["lam"], rho_upper=g["rho_upper"], method="4.32")
    return r
