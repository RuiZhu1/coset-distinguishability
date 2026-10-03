"""A distance-independent form of the gap refinement (theory Proposition 4.35) for stim's rotated memory circuit.

The observable graph of every distance is embedded in one translation-invariant lattice with twelve offsets
(Proposition 4.30, (H1)-(H2)). With per-offset envelopes q^(v) >= q_e, w-(v) <= w_e <= w+(v) and the lattice distance
D+ of the weights w+, hypothesis (A2) d_Go(z, y) <= D+(y - z) lets every term of Theorem 4.34 be bounded by lattice
kernels with first/last offsets:
  error runs:      K_X[f, l] = sum over NB lattice walks f..l of prod q^ * exp(lam D+(end)) * tilt,
  correction runs: K_C[f, l] = sum over NB lattice walks f..l of prod exp(-(lam+mu) w-) * exp(mu D+(end)) * tilt,
the second by the Chernoff bound 1[sum w- <= D+(end)] <= exp(mu (D+(end) - sum w-)) for geodesics (mu >= 0). Walks
longer than L are cut into blocks of length L (D+ is subadditive; non-backtracking across blocks is relaxed). With the
tilt exp(-theta dy) per step, the junction matrix J (no reversal) and M = [[0, J K_C], [J K_X, 0]], rho(M) < 1 gives
  p_L(d) <= (d+1)^2/2 * (x_b + c_b)^2 * F * exp(-theta (d - 2)),  F = 1 + v0 (I - M)^{-1} 1,
x_b = q_b exp(lam w_b+), c_b = exp(-lam w_b-), v0 = (1^T K_X, 1^T K_C).
"""
from __future__ import annotations

import collections
import heapq

import numpy as np
import stim

from lcd.analysis.circuit_peierls import dem_graph, pymatching_weights


def surface_circuit(d: int, p: float) -> stim.Circuit:
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                                  after_reset_flip_probability=p, before_measure_flip_probability=p,
                                  before_round_data_depolarization=p)


def envelopes(d: int, p: float) -> dict:
    """Per-offset (reduced coordinates x/2, y/2, t) envelopes of q and of pymatching's weights, boundary envelopes."""
    c = surface_circuit(d, p)
    dem = c.detector_error_model(decompose_errors=True)
    co = c.get_detector_coordinates()
    G = dem_graph(dem)
    w = pymatching_weights(dem, G)
    env = collections.defaultdict(lambda: [0.0, np.inf, 0.0])
    bq, bw = 0.0, [np.inf, 0.0]
    for i, e in enumerate(G.edges):
        if len(e) == 2:
            v = tuple(int(x) for x in np.array(co[e[1]]) - np.array(co[e[0]]))
            v = max(v, tuple(-x for x in v))
            E = env[v]
            E[0], E[1], E[2] = max(E[0], G.q[i]), min(E[1], w[i]), max(E[2], w[i])
        else:
            bq, bw = max(bq, G.q[i]), [min(bw[0], w[i]), max(bw[1], w[i])]
    offs, q, wp, wm = [], [], [], []
    for v, (qq, lo, hi) in sorted(env.items()):
        for s in (1, -1):
            offs.append((s * v[0] // 2, s * v[1] // 2, s * v[2]))
            q.append(qq); wp.append(hi); wm.append(lo)
    return dict(offs=np.array(offs), q=np.array(q), wp=np.array(wp), wm=np.array(wm), bq=bq, bw=bw,
                raw={str(k): v for k, v in env.items()})


def lattice_distance(offs, wp, shape) -> np.ndarray:
    """D+ from the centre of a box (Dijkstra restricted to the box: >= the lattice distance, which is conservative)."""
    X, Y, T = (s // 2 for s in shape)
    D = np.full(shape, np.inf)
    D[X, Y, T] = 0.0
    pq = [(0.0, (X, Y, T))]
    while pq:
        dd, x = heapq.heappop(pq)
        if dd > D[x]:
            continue
        for v, wv in zip(offs, wp):
            y = (x[0] + v[0], x[1] + v[1], x[2] + v[2])
            if all(0 <= y[k] < shape[k] for k in range(3)) and dd + wv < D[y] - 1e-12:
                D[y] = dd + wv
                heapq.heappush(pq, (dd + wv, y))
    return D


def _per_length(offs, step, D_weight, L):
    """A[k, f, l]: NB walks of length k+1, first offset f, last offset l, of prod step * D_weight(end)."""
    shape = D_weight.shape
    X, Y, T = (s // 2 for s in shape)
    n = len(offs)
    out = np.zeros((L, n, n))
    allowed = [[i for i in range(n) if not np.all(offs[i] == -offs[j])] for j in range(n)]
    for f, vf in enumerate(offs):
        cur = np.zeros((n,) + shape)
        cur[(f, X + vf[0], Y + vf[1], T + vf[2])] = step[f]
        out[0, f] = [(cur[l] * D_weight).sum() for l in range(n)]
        for k in range(1, L):
            nxt = np.zeros_like(cur)
            for j, v in enumerate(offs):
                nxt[j] = step[j] * np.roll(sum(cur[i] for i in allowed[j]), shift=tuple(v), axis=(0, 1, 2))
            cur = nxt
            out[k, f] = [(cur[l] * D_weight).sum() for l in range(n)]
    return out


def _blocked(A):
    """All lengths: lengths >= L in blocks of length L (row-sum norm); tail mass put on every last offset."""
    L = A.shape[0]
    S, B = A[:L - 1].sum(0), A[L - 1]
    nB, nS = B.sum(1).max(), S.sum(1).max()
    if nB >= 1:
        return None
    return S + np.outer(B.sum(1) / (1 - nB) * (1 + nS), np.ones(B.shape[0]))


class Lattice:
    def __init__(self, p: float, L: int = 16, d_env: int = 9):
        self.p, self.L = p, L
        self.env = envelopes(d_env, p)
        o = self.env["offs"]
        self.shape = (4 * L + 1, 2 * L + 1, 2 * L + 1)
        self.D = lattice_distance(o, self.env["wp"], self.shape)
        self.Dz = np.where(np.isfinite(self.D), self.D, 0.0)      # cells beyond reach of L steps carry no walks
        n = len(o)
        self.J = np.array([[0.0 if np.all(o[f] == -o[l]) else 1.0 for f in range(n)] for l in range(n)])

    def kernels(self, lam: float, mu: float, theta: float):
        o = self.env["offs"]
        tilt = np.exp(-theta * o[:, 1].astype(float))
        KX = _blocked(_per_length(o, self.env["q"] * tilt, np.exp(lam * self.Dz), self.L))
        KC = _blocked(_per_length(o, np.exp(-(lam + mu) * self.env["wm"]) * tilt, np.exp(mu * self.Dz), self.L))
        return KX, KC

    def evaluate(self, lam: float, mu: float, theta: float) -> dict:
        KX, KC = self.kernels(lam, mu, theta)
        if KX is None or KC is None:
            return dict(rho=np.inf)
        n = KX.shape[0]
        M = np.zeros((2 * n, 2 * n))
        M[:n, n:], M[n:, :n] = self.J @ KC, self.J @ KX
        rho = float(max(abs(np.linalg.eigvals(M))))
        out = dict(rho=rho)
        if rho < 1:
            v0 = np.r_[np.ones(n) @ KX, np.ones(n) @ KC]
            out["F"] = float(1 + v0 @ np.linalg.solve(np.eye(2 * n) - M, np.ones(2 * n)))
            out["xb"] = float(self.env["bq"] * np.exp(lam * self.env["bw"][1]))
            out["cb"] = float(np.exp(-lam * self.env["bw"][0]))
        return out

    def theta_star(self, lams=(0.42, 0.46, 0.5, 0.54), mus=(0.1, 0.2, 0.3), iters: int = 11) -> dict:
        best = dict(theta=-1.0)
        for lam in lams:
            for mu in mus:
                if self.evaluate(lam, mu, 0.0)["rho"] >= 1:
                    continue
                lo, hi = 0.0, 2.5
                for _ in range(iters):
                    th = (lo + hi) / 2
                    lo, hi = (th, hi) if self.evaluate(lam, mu, th)["rho"] < 1 else (lo, th)
                if lo > best["theta"]:
                    best = dict(theta=lo, lam=lam, mu=mu)
        return best


def bound(Lt: Lattice, d: int, lam: float, mu: float, theta: float) -> float:
    r = Lt.evaluate(lam, mu, theta)
    if r["rho"] >= 1:
        return np.inf
    return (d + 1) ** 2 / 2 * (r["xb"] + r["cb"]) ** 2 * r["F"] * np.exp(-theta * (d - 2))


def check_A2(d: int, p: float, wp_by_offset: dict) -> dict:
    """(A2): max over detector pairs of d_Go(z, y) - D+(y - z) on the distance-d graph (should be <= 0)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra
    c = surface_circuit(d, p)
    dem = c.detector_error_model(decompose_errors=True)
    co = c.get_detector_coordinates()
    G = dem_graph(dem)
    w = pymatching_weights(dem, G)
    dets = sorted({x for e in G.edges for x in e})
    vid = {x: i for i, x in enumerate(dets)}
    V = len(dets)
    r, cc, ww = [], [], []
    for i, e in enumerate(G.edges):
        a = vid[e[0]]
        b = V if len(e) == 1 else vid[e[1]]
        r += [a, b]; cc += [b, a]; ww += [w[i], w[i]]
    DG = dijkstra(sp.coo_matrix((ww, (r, cc)), shape=(V + 1, V + 1)).tocsr(), directed=False)[:V, :V]
    P = np.array([co[x] for x in dets]).astype(int)
    P = np.c_[P[:, 0] // 2, P[:, 1] // 2, P[:, 2]]
    span = P.max(0) - P.min(0)
    shape = tuple(int(2 * s + 9) for s in span)
    offs, wp = [], []
    for v, wv in wp_by_offset.items():
        for s in (1, -1):
            offs.append((s * v[0] // 2, s * v[1] // 2, s * v[2])); wp.append(wv)
    D = lattice_distance(np.array(offs), np.array(wp), shape)
    X, Y, T = (s // 2 for s in shape)
    worst = -np.inf
    for i in range(V):
        dl = D[X + P[:, 0] - P[i, 0], Y + P[:, 1] - P[i, 1], T + P[:, 2] - P[i, 2]]
        worst = max(worst, float(np.max(DG[i] - dl)))
    return dict(d=d, detectors=V, max_excess=worst)
