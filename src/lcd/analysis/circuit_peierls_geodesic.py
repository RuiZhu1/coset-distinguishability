"""Geodesic and gap refinements of the circuit-level Peierls bound (theory Theorems 4.32, 4.34).

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

    def kernels(self, lam: float, q: np.ndarray, what: np.ndarray, q_lower: np.ndarray | None = None):
        """With ``q_lower``, x_e uses q (an upper limit) and c_e uses 1 - q_lower (Theorem 4.32 proof, step (b):
        prod_S q_e prod_rest (1 - q_e) <= prod_S q_hi prod_rest (1 - q_lo))."""
        ql = q if q_lower is None else q_lower
        x = q * np.exp(lam * what)
        c = (1 - ql) * np.exp(-lam * what)
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


def geodesic_bound(dem, lams=None, G: DemGraph | None = None, q_upper: np.ndarray | None = None,
                   q_lower: np.ndarray | None = None) -> dict:
    """Upper bound on the per-shot failure probability of pymatching (hence of ML) with the geodesic refinement.

    The decoder's weights always come from ``dem`` (pymatching's integer weights). The edge probabilities default to
    G.q; with ``q_upper`` / ``q_lower`` (arrays on G.edges, q_lower <= q <= q_upper for the true q) the bound holds for
    every independent-edge model with q in the box: x_e uses q_upper and c_e uses q_lower (not monotone in q otherwise).
    """
    G = G if G is not None else dem_graph(dem)
    if q_upper is None and q_lower is not None:
        raise ValueError("q_lower needs q_upper")
    qu = G.q if q_upper is None else np.asarray(q_upper, float)
    # q_upper alone: q_lower = 0 is the conservative choice
    ql = G.q if q_upper is None else (np.zeros_like(qu) if q_lower is None else np.asarray(q_lower, float))
    if qu.shape != G.q.shape or ql.shape != G.q.shape:
        raise ValueError("q_upper / q_lower must be arrays on G.edges")
    if np.any(ql > qu) or np.any(ql < 0) or np.any(qu > 0.5):
        raise ValueError("need 0 <= q_lower <= q_upper <= 1/2")
    wt, kappa = pymatching_integer_weights(dem, G)
    geo = _Geo(G, wt)
    what = wt / (2 * kappa)
    V = geo.V

    def f(lam):
        k = geo.kernels(lam, qu, what, ql)
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
    """The smallest of Theorems 4.28, 4.32 and 4.34 (all are upper bounds); ``method`` says which one."""
    from lcd.analysis.circuit_peierls import peierls_bound
    G = G if G is not None else dem_graph(dem)
    r = dict(peierls_bound(dem, G=G), method="4.28")
    for name, fn in (("4.32", geodesic_bound), ("4.34", gap_bound_nb)):
        try:
            g = fn(dem, G=G)
        except NotImplementedError:
            return r
        if g["bound"] < r["bound"]:
            r.update(bound=g["bound"], lam=g["lam"], rho_upper=g["rho_upper"], method=name)
    return r


def gap_bound(dem, lams=None, G: DemGraph | None = None, q_upper=None, q_lower=None) -> dict:
    """Gap refinement with vertex-level junctions (Theorem 4.34 without its non-backtracking junctions; kept for the checks): the Chernoff factor is charged to the distances across the error gaps.

    On a failure cycle with correction runs sigma_1..sigma_k (cyclic order) and error runs pi_j between them, with end
    points z_j (end of sigma_j) and y_{j+1} (start of sigma_{j+1}), replacing every sigma_i by geodesics z_j -> y_{j+1}
    keeps the syndrome, so minimality gives sum_i w~(sigma_i) <= sum_j d~(z_j, y_{j+1}) (<= sum_j w~(pi_j)). The Chernoff
    factor exp(lam (sum_j d^(z_j, y_{j+1}) - sum_i w^(sigma_i))) is therefore attached to the gaps, and an error run
    pi from z to y costs prod_{e in pi} q_e * exp(lam d^(z, y)); a cycle without correction runs costs prod q_e. Through
    the boundary vertex, d(z, y) <= d(z, b) + d(b, y) is used. C-runs are geodesics with factor (1 - q_e) exp(-lam w^_e)
    as in Theorem 4.32.
    """
    G = G if G is not None else dem_graph(dem)
    wt, kappa = pymatching_integer_weights(dem, G)
    geo = _Geo(G, wt)
    what = wt / (2 * kappa)
    qu = G.q if q_upper is None else np.asarray(q_upper)
    ql = G.q if q_lower is None else np.asarray(q_lower)
    V, b = geo.V, geo.b
    Dh = geo.D / (2 * kappa)                                          # distances in w^ units
    n = len(geo.arc_tail)
    Bq = sp.csc_matrix((qu[geo.arc_edge[geo.nb_cols]], (geo.nb_rows, geo.nb_cols)), shape=(n, n))
    lu = spla.splu((sp.identity(n, format="csc") - Bq).tocsc())
    if not np.all(lu.solve(np.ones(n)) > 0):
        return dict(bound=np.inf, lam=None, rho_upper=np.nan, kappa=kappa, stats=G.stats)
    Out = sp.csc_matrix((np.ones(n), (np.arange(n), geo.arc_head)), shape=(n, V)).toarray()
    In = sp.csr_matrix((qu[geo.arc_edge], (geo.arc_tail, np.arange(n))), shape=(V, n))
    W = In @ lu.solve(Out)                                              # sum of prod q over NB walks, length >= 1
    I = np.eye(V)
    s0 = np.zeros(V); s1 = np.zeros(V)
    for u, e in geo.bedge.items():
        if geo.side[u] == 0:
            s0[u] = qu[e]
        elif geo.side[u] == 1:
            s1[u] = qu[e]
    WX0 = s0 @ (I + W)                                                  # error run b -> y (prod q only)
    WX1 = (I + W) @ s1                                                  # error run z -> b
    pure = float(WX0 @ s1)                                              # cycles with no correction run

    def f(lam):
        c = (1 - ql) * np.exp(-lam * what)
        KX = np.exp(lam * Dh[:V, :V]) * W
        np.fill_diagonal(KX, 0.0)                                       # z == y: gap of a closed walk; see below
        KX += np.diag(np.diag(W))                                       # d(z, z) = 0: factor 1
        KC = geo.geodesic_sums(c, np.arange(V)) - I
        sX = WX0 * np.exp(lam * Dh[b, :V])
        eX = WX1 * np.exp(lam * Dh[:V, b])
        side0 = {u: c[e] for u, e in geo.bedge.items() if geo.side[u] == 0}
        side1 = {u: c[e] for u, e in geo.bedge.items() if geo.side[u] == 1}
        sC = geo.geodesic_sums(c, np.array([b]), side0)[0]
        eC = geo.geodesic_sums(c, np.array([b]), side1)[0]
        M = np.zeros((2 * V, 2 * V))
        M[:V, V:] = KC
        M[V:, :V] = KX
        luM = sla.lu_factor(np.eye(2 * V) - M)
        zz = sla.lu_solve(luM, np.ones(2 * V))
        if not np.all(zz > 0):
            return np.inf, np.nan
        return pure + float(np.r_[sX, sC] @ sla.lu_solve(luM, np.r_[eC, eX])), 1 - 1 / zz.max()

    grid = list(lams) if lams is not None else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    vals = [f(l) for l in grid]
    k = int(np.argmin([v[0] for v in vals]))
    return dict(bound=vals[k][0], lam=float(grid[k]), rho_upper=vals[k][1], kappa=kappa, stats=G.stats)


def gap_bound_nb(dem, lams=None, G: DemGraph | None = None, q_upper=None, q_lower=None) -> dict:
    """Theorem 4.34: gap refinement with non-backtracking junctions.

    Same terms as ``gap_bound``, but the alternating sequence is tracked on arcs (directed edges of G_o - b): a run is
    entered by its first arc and left by its last arc, and at a junction the next run may not start with the reverse of
    the previous run's last arc (a simple cycle never does). Runs consisting of a boundary edge only are handled as vertex
    states. Arc kernels: KXa[f, l] = q_f P[f, l] exp(lam d^(tail f, head l)) with P = (I - B_q)^{-1};
    KCa[f, l] = c_f c_l F[head f, tail l] if w~_f + d~(head f, tail l) + w~_l = d~(tail f, head l) (f != l), c_f if f = l
    is a geodesic edge, with F the interior geodesic sums.
    """
    G = G if G is not None else dem_graph(dem)
    wt, kappa = pymatching_integer_weights(dem, G)
    geo = _Geo(G, wt)
    what = wt / (2 * kappa)
    qu = G.q if q_upper is None else np.asarray(q_upper)
    ql = G.q if q_lower is None else np.asarray(q_lower)
    V, b = geo.V, geo.b
    D = geo.D
    Dh = D / (2 * kappa)
    tail, head, ae = geo.arc_tail, geo.arc_head, geo.arc_edge
    na = len(tail)
    Bq = sp.csc_matrix((qu[ae[geo.nb_cols]], (geo.nb_rows, geo.nb_cols)), shape=(na, na))
    lu = spla.splu((sp.identity(na, format="csc") - Bq).tocsc())
    if not np.all(lu.solve(np.ones(na)) > 0):
        return dict(bound=np.inf, lam=None, rho_upper=np.nan, kappa=kappa, stats=G.stats)
    P = lu.solve(np.eye(na))                                            # NB walk sums between arcs (P[f, f] includes 1)
    qa = qu[ae]
    # vertex-level walk sums for the pure-error cycles
    Out = np.zeros((na, V)); Out[np.arange(na), head] = 1.0
    In = np.zeros((V, na)); In[tail, np.arange(na)] = qa
    W = In @ P @ Out
    I = np.eye(V)
    s0 = np.zeros(V); s1 = np.zeros(V); c0 = {}; c1 = {}
    for u, e in geo.bedge.items():
        if geo.side[u] == 0:
            s0[u] = qu[e]
        elif geo.side[u] == 1:
            s1[u] = qu[e]
    pure = float(s0 @ (I + W) @ s1)
    J = (head[:, None] == tail[None, :]) & (head[None, :] != tail[:, None])   # junction l -> f, no reversal
    by_tail = np.zeros((V, na)); by_tail[tail, np.arange(na)] = 1.0
    bw = np.full(V, np.nan)
    for u, e in geo.bedge.items():
        bw[u] = wt[e]
    geo_b = np.abs(D[b, :V] - bw) < 0.5                                   # boundary edge b-u is a geodesic
    side = geo.side

    def f(lam):
        c = (1 - ql) * np.exp(-lam * what)
        ca = c[ae]
        E = np.exp(lam * Dh[np.ix_(tail, head)])
        KXa = (qa[:, None] * P) * E
        F = geo.geodesic_sums(c, np.arange(V))
        cond = np.abs(wt[ae][:, None] + D[np.ix_(head, tail)] + wt[ae][None, :] - D[np.ix_(tail, head)]) < 0.5
        KCa = (ca[:, None] * ca[None, :]) * F[np.ix_(head, tail)] * cond
        single = np.abs(wt[ae] - D[tail, head]) < 0.5
        KCa[np.arange(na), np.arange(na)] = np.where(single, ca, 0.0)
        Jf = J.astype(float)
        # boundary pieces
        cb = np.zeros(V)
        for u, e in geo.bedge.items():
            cb[u] = c[e]
        vX = np.where(side == 0, s0 * np.exp(lam * Dh[b, :V]), 0.0)          # X-run = boundary edge only, ends at u
        vC = np.where((side == 0) & geo_b, cb, 0.0)                        # C-run = boundary edge only
        Fb0 = geo.geodesic_sums(c, np.array([b]), {u: cb[u] for u in range(V) if side[u] == 0})[0]
        Fb1 = geo.geodesic_sums(c, np.array([b]), {u: cb[u] for u in range(V) if side[u] == 1})[0]
        # starts into arc states
        firstX = s0 @ by_tail                                              # q_{b,tail f} for arcs leaving side-0 vertices
        sXa = (firstX * qa) @ P * np.exp(lam * Dh[b, head])                # X-run from b with >= 1 interior arc, last arc l
        sCa = Fb0[tail] * ca * (np.abs(D[b, tail] + wt[ae] - D[b, head]) < 0.5)   # C-run from b, last arc l
        sCa = sCa + (vX @ by_tail) @ KCa                                   # X boundary edge, then a C-run from u
        sXa = sXa + (vC @ by_tail) @ KXa                                   # C boundary edge, then an X-run from u
        # ends from arc states (state = last arc of the previous run)
        lastX = (P * (side[head] == 1)[None, :]) @ s1[head]                 # sum over walks from f to a side-1 vertex v
        eXa = qa * lastX * np.exp(lam * Dh[tail, b])                         # X-run starting with f, into b
        eCa = Fb1[head] * ca * (np.abs(D[b, head] + wt[ae] - D[b, tail]) < 0.5)   # C-run starting with f, into b
        evX = np.where(side[head] == 1, s1[head] * np.exp(lam * Dh[head, b]), 0.0)   # X boundary edge only
        evC = np.where((side[head] == 1) & geo_b[head], cb[head], 0.0)               # C boundary edge only
        endX = Jf @ eCa + evC                                              # after an X-run: a C-run into b
        endC = Jf @ eXa + evX                                              # after a C-run: an X-run into b
        # direct start -> end with vertex-only starts
        direct = float(vX @ (by_tail @ eCa)) + float(vC @ (by_tail @ eXa))
        M = np.zeros((2 * na, 2 * na))
        M[:na, na:] = Jf @ KCa                                             # X-ended -> C-ended
        M[na:, :na] = Jf @ KXa                                             # C-ended -> X-ended
        luM = sla.lu_factor(np.eye(2 * na) - M)
        zz = sla.lu_solve(luM, np.ones(2 * na))
        if not np.all(zz > 0):
            return np.inf, np.nan
        tot = pure + direct + float(np.r_[sXa, sCa] @ sla.lu_solve(luM, np.r_[endX, endC]))
        return tot, 1 - 1 / zz.max()

    grid = list(lams) if lams is not None else [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    vals = [f(l) for l in grid]
    k = int(np.argmin([v[0] for v in vals]))
    return dict(bound=vals[k][0], lam=float(grid[k]), rho_upper=vals[k][1], kappa=kappa, stats=G.stats)
