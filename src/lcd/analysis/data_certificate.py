"""Certificate of a logical error rate from the data's own detection statistics (theory, end of Section 4).

Assumption (G): the observable part of the noise is an independent-edge model on the observable graph G_o of a fixed
decoding model: every edge e of G_o flips independently with an unknown probability q_e in [0, 1/2], and the shots are
i.i.d. (Google's p_ij estimator is derived under the same assumption.) Then, for an interior edge e = (i, j) of the simple
graph G_o, with Z_i = (-1)^{x_i}, a_i = E x_i, c_ij = E x_i x_j and s_e = 1 - 2 q_e,

    E Z_i = s_e A,  E Z_j = s_e B,  E Z_i Z_j = A B          (A, B: parities of the other edges at i, at j; independent)

so  s_e^2 = E Z_i E Z_j / E Z_i Z_j,  i.e.  1 - s_e^2 = 4 (c_ij - a_i a_j) / (1 - 2 a_i - 2 a_j + 4 c_ij),
and for the boundary edge b at detector i,  1 - 2 a_i = s_b prod_{e at i, e != b} s_e.

Confidence intervals (simultaneous, Bonferroni over all "slots", total level delta):
  method "cp"      Clopper-Pearson for every a_i and every c_ij, then interval arithmetic (the literal recipe);
  method "paired"  (default, sharper) the covariance is estimated directly: shots are paired (2k, 2k+1) and
                   D = (x_i - x_i')(x_j - x_j') in {-1, 0, 1} has E D = 2 (c_ij - a_i a_j); its mean gets a betting
                   confidence interval (Waudby-Smith & Ramdas, hedged capital with predictable plug-in bets; valid for
                   i.i.d. bounded variables at every n); the denominator 1 - 2 P[x_i != x_j] gets a Clopper-Pearson
                   interval; the boundary a_i get Clopper-Pearson intervals.
  pool=True        additional assumption (T), time stationarity: edges of the same type (same pair of detector types,
                   same relative time offset; the type of a detector is its coordinate list with time shifted to 0) share
                   q_e. The per-shot (pair) average over the group replaces the single-edge statistic.

Every step of the interval propagation is monotone (see ``_q_from_ratio`` and ``boundary_intervals``), so on the event that
all slot intervals cover, q_lo <= q <= q_hi on every edge. Theorem 4.28(c) is monotone in q (use q_hi); Theorem 4.32 is
not, but its proof only needs prod_S q prod_rest (1 - q) <= prod_S q_hi prod_rest (1 - q_lo) (``geodesic_bound(q_upper,
q_lower)``). The decoder's weights are fixed by the decoding model; the bound holds simultaneously for all weights on the
coverage event, so the weights may themselves be fitted to the data.

    from lcd.analysis.data_certificate import edge_intervals, certify
    iv = edge_intervals(G, events, delta=1e-3, coords=circuit.get_detector_coordinates(), pool=False)
    res = certify(dem, events, delta=1e-3)
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field, replace
from math import log, sqrt

import numpy as np
from scipy.stats import beta as beta_dist

from lcd.analysis.circuit_peierls import DemGraph, dem_graph, peierls_bound

CHUNK = 32          # columns per betting_ci call (memory ~ 10 arrays of shots x CHUNK floats)


# ------------------------------------------------------------------ one-dimensional confidence intervals
def clopper_pearson(k, n, alpha):
    """Two-sided Clopper-Pearson interval at level 1 - alpha (vectorized over k)."""
    k = np.asarray(k, float)
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


def betting_ci(X: np.ndarray, alpha: float, c: float = 0.75, width: float = 0.25, iters: int = 60):
    """Two-sided confidence interval at level 1 - alpha for the mean of i.i.d. variables in [0, 1], columnwise.

    Hedged capital process with predictable plug-in bets (Waudby-Smith & Ramdas, JRSS-B 2024, fixed-n tuning):
    mu_t, sigma2_t running estimates from X_1..X_{t-1}; lam_t = sqrt(2 ln(2/alpha) / (n sigma2_{t-1})), truncated at a
    constant cap; K+(m) = prod (1 + lam_t^+ (X_t - m)), K-(m) = prod (1 - lam_t^- (X_t - m)). By Ville's inequality
    P[K+(mu) >= 2/alpha] <= alpha/2 and likewise for K-. The caps do not depend on m inside the search window
    [mhat - width, mhat + width], so K+ is nonincreasing and K- nondecreasing in m on the whole range where the factors
    stay positive (which includes everything below / above the window); hence the rejection regions are half-lines and the
    endpoints are found by bisection. If the window does not contain an endpoint, the trivial endpoint 0 or 1 is used.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n, k = X.shape
    if X.min() < 0 or X.max() > 1:
        raise ValueError("values must lie in [0, 1]")
    t = np.arange(1, n + 1)[:, None]
    cs = np.cumsum(X, 0)
    mu = (0.5 + cs) / (t + 1)                                  # mu_t uses X_1..X_t
    mu_prev = np.vstack([np.full((1, k), 0.5), mu[:-1]])
    sq = np.cumsum((X - mu_prev) ** 2, 0)
    sig2 = (0.25 + sq) / (t + 1)
    sig2_prev = np.vstack([np.full((1, k), 0.25), sig2[:-1]])
    lam = np.sqrt(2 * log(2 / alpha) / (n * sig2_prev))
    mhat = X.mean(0)
    w_lo, w_hi = np.clip(mhat - width, 0, 1), np.clip(mhat + width, 0, 1)
    lam_p = np.minimum(lam, c / np.maximum(w_hi, 1e-12))       # 1 + lam_p (x - m) >= 1 - c > 0 for m <= w_hi
    lam_m = np.minimum(lam, c / np.maximum(1 - w_lo, 1e-12))   # 1 - lam_m (x - m) >= 1 - c > 0 for m >= w_lo
    thr = log(2 / alpha)

    def logK(lmb, m, sign):
        return np.log1p(sign * lmb * (X - m[None, :])).sum(0)

    # lower endpoint: smallest m with logK+(m) < thr (K+ nonincreasing in m)
    lo_a, lo_b = w_lo.copy(), mhat.copy()                     # K+(mhat) < thr always? check below
    rej_at_wlo = logK(lam_p, w_lo, 1) >= thr
    for _ in range(iters):
        mid = (lo_a + lo_b) / 2
        r = logK(lam_p, mid, 1) >= thr
        lo_a, lo_b = np.where(r, mid, lo_a), np.where(r, lo_b, mid)
    lo = np.where(rej_at_wlo, lo_a, 0.0)
    hi_a, hi_b = mhat.copy(), w_hi.copy()
    rej_at_whi = logK(lam_m, w_hi, -1) >= thr
    for _ in range(iters):
        mid = (hi_a + hi_b) / 2
        r = logK(lam_m, mid, -1) >= thr
        hi_a, hi_b = np.where(r, hi_a, mid), np.where(r, mid, hi_b)
    hi = np.where(rej_at_whi, hi_b, 1.0)
    return lo, hi


# ------------------------------------------------------------------ the p_ij identity
def q_from_moments(a_i, a_j, c_ij):
    """Point estimate q_e of an interior edge from a_i, a_j, c_ij: 1 - (1 - 2q)^2 = 4 (c - a_i a_j) / (1 - 2a_i - 2a_j + 4c)."""
    return _q_from_ratio(4 * (c_ij - a_i * a_j) / (1 - 2 * a_i - 2 * a_j + 4 * c_ij))


def _q_from_ratio(y):
    """q = (1 - sqrt(1 - y)) / 2 for y = 1 - s^2 in [0, 1]; nondecreasing in y, clipped to [0, 1/2]."""
    y = np.clip(np.asarray(y, float), 0.0, 1.0)
    return (1 - np.sqrt(1 - y)) / 2


def _y_interval(T_lo, T_hi, R_lo, R_hi):
    """Interval for y = 4 T / R with T in [T_lo, T_hi] and R in [R_lo, R_hi], R > 0 required for a finite upper end.

    y is nondecreasing in T, and for T >= 0 nonincreasing in R: y_hi = 4 T_hi / R_lo (if T_hi > 0), y_lo = 4 T_lo / R_hi
    (if T_lo > 0), else 0. If R_lo <= 0, y_hi = 1 (q_hi = 1/2)."""
    T_lo, T_hi, R_lo, R_hi = map(np.asarray, (T_lo, T_hi, R_lo, R_hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        y_hi = np.where(T_hi <= 0, 0.0, np.where(R_lo > 0, 4 * T_hi / np.where(R_lo > 0, R_lo, 1), 1.0))
        y_lo = np.where((T_lo > 0) & (R_hi > 0), 4 * T_lo / np.where(R_hi > 0, R_hi, 1), 0.0)
    return np.clip(y_lo, 0, 1), np.clip(y_hi, 0, 1)


# ------------------------------------------------------------------ edge groups
def detector_types(coords: dict) -> dict:
    """det -> (type, time): the coordinate list read as (x, y, t) triples, with t shifted so that its minimum t0 is 0,
    together with the layer ("first" if t0 is the smallest time of all detectors, "last" if the largest, else "bulk")."""
    out = {}
    for d, c in coords.items():
        c = list(c)
        if len(c) == 0 or len(c) % 3:
            out[d] = (("det", d), 0.0)
            continue
        tr = [tuple(c[k:k + 3]) for k in range(0, len(c), 3)]
        t0 = min(z[2] for z in tr)
        out[d] = (tuple((x, y, t - t0) for x, y, t in tr), t0)
    ts = [t for _, t in out.values()]
    tmin, tmax = min(ts), max(ts)
    return {d: ((sig, "first" if t == tmin else "last" if t == tmax else "bulk"), t) for d, (sig, t) in out.items()}


def edge_groups(G: DemGraph, coords: dict | None = None, pool: bool = False):
    """(interior groups, boundary groups) as lists of lists of edge indices; singletons unless pool (assumption (T))."""
    inter = [i for i, e in enumerate(G.edges) if len(e) == 2]
    bdry = [i for i, e in enumerate(G.edges) if len(e) == 1]
    if not pool:
        return [[i] for i in inter], [[i] for i in bdry]
    if coords is None:
        raise ValueError("pooling needs detector coordinates")
    ty = detector_types(coords)
    gi, gb = collections.defaultdict(list), collections.defaultdict(list)
    for i in inter:
        u, v = G.edges[i]
        t0 = min(ty[u][1], ty[v][1])
        key = tuple(sorted([(ty[u][0], ty[u][1] - t0), (ty[v][0], ty[v][1] - t0)], key=repr))
        gi[key].append(i)
    for i in bdry:
        gb[ty[G.edges[i][0]][0]].append(i)
    return list(gi.values()), list(gb.values())


# ------------------------------------------------------------------ intervals
@dataclass
class EdgeIntervals:
    q_lo: np.ndarray
    q_hi: np.ndarray
    q_hat: np.ndarray
    method: str
    pool: bool
    delta: float
    slots: int
    info: dict = field(default_factory=dict)


def point_estimates(G: DemGraph, X: np.ndarray) -> np.ndarray:
    """q_e from the moments of the detection events X (shots x detectors, bool), by the identity above."""
    vid, dets = _vertices(G)
    Xf = X[:, dets].astype(np.float64)
    a = Xf.mean(0)
    q = np.zeros(len(G.edges))
    inter = [k for k, e in enumerate(G.edges) if len(e) == 2]
    I = np.array([vid[G.edges[k][0]] for k in inter], int)
    J = np.array([vid[G.edges[k][1]] for k in inter], int)
    c = (Xf[:, I] * Xf[:, J]).mean(0)
    q[inter] = q_from_moments(a[I], a[J], c)
    s = 1 - 2 * q
    for k, e in enumerate(G.edges):
        if len(e) == 1:
            i = e[0]
            P = np.prod([s[m] for m in _incident(G)[i] if m != k])
            q[k] = 0.5 * (1 - min(1.0, max(0.0, (1 - 2 * a[vid[i]]) / P)))
    return q


def _vertices(G):
    dets = sorted({x for e in G.edges for x in e})
    return {x: i for i, x in enumerate(dets)}, np.array(dets, int)


def _incident(G):
    inc = collections.defaultdict(list)
    for k, e in enumerate(G.edges):
        for x in e:
            inc[x].append(k)
    return inc


def boundary_intervals(G: DemGraph, bgroups, q_lo, q_hi, A_lo, A_hi):
    """q_b intervals from 1 - 2 a_i = s_b P_i, P_i = prod_{e at i, e != b} (1 - 2 q_e), for groups sharing s_b.

    Averaging over the detectors i of a group: 1 - 2 abar = s_b * mean_i P_i, abar = mean_i a_i in [A_lo, A_hi].
    s_b = (1 - 2 abar) / mean P is nonincreasing in abar and, when 1 - 2 abar > 0, nonincreasing in mean P; P_i is
    nonincreasing in every q_e (factors in [0, 1]). So s_b >= (1 - 2 A_hi) / mean_i prod (1 - 2 q_lo,e) gives q_b,hi and
    s_b <= (1 - 2 A_lo) / mean_i prod (1 - 2 q_hi,e) gives q_b,lo."""
    inc = _incident(G)
    for g, alo, ahi in zip(bgroups, A_lo, A_hi):
        P_hi = np.mean([np.prod([1 - 2 * q_lo[m] for m in inc[G.edges[k][0]] if m != k]) for k in g])
        P_lo = np.mean([np.prod([1 - 2 * q_hi[m] for m in inc[G.edges[k][0]] if m != k]) for k in g])
        s_lo = (1 - 2 * ahi) / P_hi if P_hi > 0 else 0.0
        s_hi = (1 - 2 * alo) / P_lo if P_lo > 0 else 1.0
        qh = 0.5 * (1 - min(1.0, max(0.0, s_lo)))
        ql = 0.5 * (1 - min(1.0, max(0.0, s_hi)))
        for k in g:
            q_hi[k], q_lo[k] = qh, ql


def edge_intervals(G: DemGraph, X: np.ndarray, delta: float = 1e-3, method: str = "paired", pool: bool = False,
                   coords: dict | None = None) -> EdgeIntervals:
    """Simultaneous intervals q_lo <= q_e <= q_hi on all edges of G_o, at level 1 - delta under (G) (and (T) if pool)."""
    if any(G.split[i] for i in range(len(G.edges))):
        raise NotImplementedError("non-exclusive mechanisms in the decoding model")
    vid, dets = _vertices(G)
    X = np.asarray(X, bool)
    n = X.shape[0]
    igroups, bgroups = edge_groups(G, coords, pool)
    if method == "cp" and pool:
        raise ValueError("method 'cp' is unpooled")
    q_lo, q_hi = np.zeros(len(G.edges)), np.full(len(G.edges), 0.5)
    q_hat = point_estimates(G, X)
    Xd = X[:, dets]
    if method == "cp":
        inter = [g[0] for g in igroups]
        slots = len(dets) + len(inter)
        al = delta / slots
        a_lo, a_hi = clopper_pearson(Xd.sum(0), n, al)
        I = np.array([vid[G.edges[k][0]] for k in inter], int)
        J = np.array([vid[G.edges[k][1]] for k in inter], int)
        c_lo, c_hi = clopper_pearson((Xd[:, I] & Xd[:, J]).sum(0), n, al)
        T_hi, T_lo = c_hi - a_lo[I] * a_lo[J], c_lo - a_hi[I] * a_hi[J]          # c - a b: up in c, down in a, b >= 0
        R_lo = 1 - 2 * a_hi[I] - 2 * a_hi[J] + 4 * c_lo                          # 1 - 2a - 2b + 4c: down in a, b; up in c
        R_hi = 1 - 2 * a_lo[I] - 2 * a_lo[J] + 4 * c_hi
        y_lo, y_hi = _y_interval(T_lo, T_hi, R_lo, R_hi)
        q_lo[inter], q_hi[inter] = _q_from_ratio(y_lo), _q_from_ratio(y_hi)
        bi = [vid[G.edges[g[0]][0]] for g in bgroups]
        boundary_intervals(G, bgroups, q_lo, q_hi, a_lo[bi], a_hi[bi])
    elif method == "paired":
        slots = 2 * len(igroups) + len(bgroups)
        al = delta / slots
        m = n // 2
        Xa, Xb = Xd[0:2 * m:2].astype(np.int8), Xd[1:2 * m:2].astype(np.int8)
        T_lo, T_hi = np.zeros(len(igroups)), np.zeros(len(igroups))
        u_lo, u_hi = np.zeros(len(igroups)), np.ones(len(igroups))
        for c0 in range(0, len(igroups), CHUNK):                                 # bounded memory
            chunk = igroups[c0:c0 + CHUNK]
            Dstat = np.zeros((m, len(chunk)))
            Ustat = np.zeros((n, len(chunk)))
            for gi, g in enumerate(chunk):
                I = np.array([vid[G.edges[k][0]] for k in g], int)
                J = np.array([vid[G.edges[k][1]] for k in g], int)
                Dstat[:, gi] = ((Xa[:, I] - Xb[:, I]) * (Xa[:, J] - Xb[:, J])).mean(1)
                Ustat[:, gi] = (Xd[:, I] != Xd[:, J]).mean(1)
            mlo, mhi = betting_ci((Dstat + 1) / 2, al)                            # mean of (D + 1)/2
            T_lo[c0:c0 + len(chunk)], T_hi[c0:c0 + len(chunk)] = mlo - 0.5, mhi - 0.5   # E D / 2 = mean cov
            single = np.array([len(g) == 1 for g in chunk])
            ulo, uhi = np.zeros(len(chunk)), np.ones(len(chunk))
            if single.any():
                ulo[single], uhi[single] = clopper_pearson(Ustat[:, single].sum(0), n, al)
            if (~single).any():
                ulo[~single], uhi[~single] = betting_ci(Ustat[:, ~single], al)
            u_lo[c0:c0 + len(chunk)], u_hi[c0:c0 + len(chunk)] = ulo, uhi
        R_lo, R_hi = 1 - 2 * u_hi, 1 - 2 * u_lo                                   # E Z_i Z_j = 1 - 2 P[x_i != x_j]
        y_lo, y_hi = _y_interval(T_lo, T_hi, R_lo, R_hi)
        for gi, g in enumerate(igroups):
            q_lo[g], q_hi[g] = _q_from_ratio(y_lo[gi]), _q_from_ratio(y_hi[gi])
        A_lo, A_hi = np.zeros(len(bgroups)), np.ones(len(bgroups))
        bi = [[vid[G.edges[k][0]] for k in g] for g in bgroups]
        for gi, ii in enumerate(bi):
            if len(ii) == 1:
                A_lo[gi], A_hi[gi] = (float(v[0]) for v in clopper_pearson([Xd[:, ii[0]].sum()], n, al))
        multi = [gi for gi, ii in enumerate(bi) if len(ii) > 1]
        for c0 in range(0, len(multi), CHUNK):
            sel = multi[c0:c0 + CHUNK]
            A_lo[sel], A_hi[sel] = betting_ci(np.stack([Xd[:, bi[gi]].mean(1) for gi in sel], 1), al)
        boundary_intervals(G, bgroups, q_lo, q_hi, A_lo, A_hi)
    else:
        raise ValueError(method)
    q_lo = np.minimum(q_lo, q_hi)
    info = dict(interior_groups=len(igroups), boundary_groups=len(bgroups), shots=n,
                max_group=max([len(g) for g in igroups + bgroups], default=0))
    return EdgeIntervals(q_lo=q_lo, q_hi=q_hi, q_hat=q_hat, method=method, pool=pool, delta=delta, slots=slots, info=info)


# ------------------------------------------------------------------ diagnostics of (G) and of i.i.d.
def nonedge_correlations(G: DemGraph, X: np.ndarray, delta: float = 1e-3) -> dict:
    """Test of (G): pairs of G_o detectors not joined by an edge have zero covariance under (G).

    Paired statistic D as above, normal approximation (a diagnostic, not part of the certificate); reports the number of
    pairs whose |z| exceeds the Bonferroni threshold at level delta, and the largest |z|."""
    from scipy.stats import norm
    vid, dets = _vertices(G)
    Xd = X[:, dets].astype(np.float64)
    m = Xd.shape[0] // 2
    Y = Xd[0:2 * m:2] - Xd[1:2 * m:2]
    S1 = Y.T @ Y / m                                                    # mean D over pairs
    S2 = (Y ** 2).T @ (Y ** 2) / m                                      # mean D^2
    var = np.maximum(S2 - S1 ** 2, 1e-300) / m
    z = S1 / np.sqrt(var)
    adj = np.zeros((len(dets), len(dets)), bool)
    for e in G.edges:
        if len(e) == 2:
            adj[vid[e[0]], vid[e[1]]] = adj[vid[e[1]], vid[e[0]]] = True
    iu = np.triu_indices(len(dets), 1)
    ne = ~adj[iu]
    zz = z[iu][ne]
    thr = float(norm.isf(delta / (2 * max(ne.sum(), 1))))
    top = np.argsort(-np.abs(zz))[:5]
    pairs = [(int(dets[iu[0][ne][k]]), int(dets[iu[1][ne][k]]), float(zz[k])) for k in top]
    return dict(pairs=int(ne.sum()), threshold=thr, exceed=int((np.abs(zz) > thr).sum()), max_abs_z=float(np.abs(zz).max()),
                top=pairs)


def stationarity(X: np.ndarray, chunk: int = 1000) -> dict:
    """Detection fraction per chunk of consecutive shots: largest deviation from the overall mean in units of its standard
    error, estimated from the per-shot detection fractions (i.i.d. shots give |z| of order sqrt(2 ln chunks))."""
    k = X.shape[0] // chunk
    per_shot = X[:k * chunk].mean(1)
    f = per_shot.reshape(k, chunk).mean(1)
    sd = per_shot.std(ddof=1) / sqrt(chunk)
    zz = (f - per_shot.mean()) / sd
    return dict(chunks=k, chunk=chunk, max_abs_z=float(np.abs(zz).max()), min_frac=float(f.min()), max_frac=float(f.max()))


# ------------------------------------------------------------------ the certificate
def certify(dem, X: np.ndarray, delta: float = 1e-3, method: str = "paired", pool: bool = False, coords=None,
            G: DemGraph | None = None, lams=(0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), geodesic: bool = True) -> dict:
    """Bounds of Theorems 4.28 (at q_hi) and 4.32 (x from q_hi, c from q_lo) and at the point estimate."""
    from lcd.analysis.circuit_peierls_geodesic import geodesic_bound
    G = G if G is not None else dem_graph(dem)
    iv = edge_intervals(G, X, delta, method, pool, coords)
    qh = np.minimum(iv.q_hat, 0.5)
    out = dict(method=method, pool=pool, delta=delta, slots=iv.slots, info=iv.info,
               q_hat=_pct(iv.q_hat), q_lo=_pct(iv.q_lo), q_hi=_pct(iv.q_hi),
               rel_halfwidth_median=float(np.median((iv.q_hi - iv.q_lo) / np.maximum(2 * iv.q_hat, 1e-12))))
    out["bound_4_28"] = peierls_bound(dem, G=replace(G, q=iv.q_hi))["bound"]
    out["bound_4_28_point"] = peierls_bound(dem, G=replace(G, q=qh))["bound"]
    if geodesic:
        out["bound_4_32"] = geodesic_bound(dem, lams=lams, G=G, q_upper=iv.q_hi, q_lower=iv.q_lo)["bound"]
        out["bound_4_32_point"] = geodesic_bound(dem, lams=lams, G=G, q_upper=qh, q_lower=qh)["bound"]
    out["_intervals"] = iv
    return out


def _pct(q):
    return [float(x) for x in np.percentile(q, [0, 25, 50, 75, 100])]
