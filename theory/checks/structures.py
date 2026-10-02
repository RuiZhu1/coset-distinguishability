"""Exact eps*(p, e) for an arbitrary nested pair of subspaces S <= N of F_2^{2n} (a linear decoding structure on n qubits).

The ML error of a linear decoding structure (V, sigma, L) with V = F_2^{2n} depends only on S = ker sigma ∩ ker L and N = ker sigma:
the syndrome identifies the coset of N, the label identifies the coset of S inside it, and
    eps* = 1 - sum_s max_{S-cosets c inside the N-coset s} P(c).
Code-capacity noise N_cc(p, e) with flags known to the decoder: for every erased set A the Pauli law is uniform on A and
depolarizing (rate p) off A, and eps* = sum_A e^|A| (1-e)^(n-|A|) eps*_A(p).
Everything is enumerated exactly (4^n Pauli errors), so n <= 6 is cheap.
"""
from __future__ import annotations

import itertools

import numpy as np


def all_paulis(n: int) -> np.ndarray:
    """All 4^n Paulis as vectors (x | z) in F_2^{2n}; qubit j is encoded by two bits of the index."""
    idx = np.arange(4 ** n)
    x = np.stack([(idx >> (2 * j)) & 1 for j in range(n)], axis=1)
    z = np.stack([(idx >> (2 * j + 1)) & 1 for j in range(n)], axis=1)
    return np.concatenate([x, z], axis=1).astype(np.uint8)


def rref(rows: np.ndarray) -> tuple[np.ndarray, list[int]]:
    M = np.array(rows, dtype=np.uint8) % 2
    if M.size == 0:
        return M.reshape(0, M.shape[-1] if M.ndim == 2 else 0), []
    piv, r = [], 0
    for c in range(M.shape[1]):
        k = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if k is None:
            continue
        M[[r, k]] = M[[k, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        piv.append(c)
        r += 1
        if r == M.shape[0]:
            break
    return M[:r], piv


def coset_ids(V: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Integer id of the coset v + span(basis) for every row v of V (bits of the reduced vector outside the pivot columns)."""
    B, piv = rref(basis) if len(basis) else (basis, [])
    W = V.copy()
    for b, c in zip(B, piv):
        W[W[:, c] == 1] ^= b
    free = [c for c in range(V.shape[1]) if c not in piv]
    ids = np.zeros(len(V), dtype=np.int64)
    for k, c in enumerate(free):
        ids |= W[:, c].astype(np.int64) << k
    return ids


class Structure:
    """Nested subspaces S <= N of F_2^{2n}; precomputes class and syndrome ids of all 4^n errors."""

    def __init__(self, n: int, S: np.ndarray, N: np.ndarray):
        self.n = n
        V = all_paulis(n)
        self.nz = np.stack([(V[:, j] | V[:, n + j]) for j in range(n)], axis=1).astype(np.int64)    # non-identity indicator
        self.cls = coset_ids(V, S)                  # S-coset
        syn = coset_ids(V, N)                       # N-coset
        ncls = int(self.cls.max()) + 1
        s_of_c = np.full(ncls, -1, dtype=np.int64)
        s_of_c[self.cls] = syn
        assert (s_of_c[self.cls] == syn).all(), "S must be contained in N"
        present = np.unique(self.cls)
        order = present[np.argsort(s_of_c[present], kind="stable")]
        sorted_s = s_of_c[order]
        self.order = order
        self.starts = np.concatenate([[0], np.nonzero(np.diff(sorted_s))[0] + 1])
        self.ncls = ncls
        self.dimS, self.dimN = len(rref(S)[1]) if len(S) else 0, len(rref(N)[1]) if len(N) else 0

    def eps_A(self, p: float, A: tuple[int, ...]) -> float:
        n = self.n
        mask = np.ones(n, dtype=bool)
        mask[list(A)] = False
        k = self.nz[:, mask].sum(axis=1)
        m = (n - len(A)) - k
        P = 0.25 ** len(A) * (1 - p) ** m * (p / 3) ** k
        W = np.bincount(self.cls, weights=P, minlength=self.ncls)[self.order]
        return 1.0 - float(np.maximum.reduceat(W, self.starts).sum())

    def eps(self, p: float, e: float) -> float:
        n = self.n
        tot = 0.0
        for a in range(n + 1):
            for A in itertools.combinations(range(n), a):
                tot += e ** a * (1 - e) ** (n - a) * self.eps_A(p, A)
        return tot

    def rate(self, p: float, e: float, h: float = 1e-6) -> tuple[float, float, float]:
        """(R, d eps*/dp, d eps*/de) at (p, e), right derivative in e; R = nan if the p-derivative vanishes."""
        n = self.n
        dp = (self.eps(p + h, e) - self.eps(p - h, e)) / (2 * h) if e > 0 else None
        if e == 0:
            # only A = empty and |A| = 1 contribute at e = 0
            dp = (self.eps_A(p + h, ()) - self.eps_A(p - h, ())) / (2 * h) * 1.0
            de = sum(self.eps_A(p, (j,)) for j in range(n)) - n * self.eps_A(p, ())
        else:
            de = (self.eps(p, e + h) - self.eps(p, e - h)) / (2 * h)
        R = de / dp if abs(dp) > 1e-13 else float("nan")
        return R, dp, de


def random_chain(n: int, rng: np.random.Generator, dimS: int | None = None, dimN: int | None = None):
    """Random S <= N in F_2^{2n}: first dimS rows / first dimN rows of a random invertible matrix."""
    while True:
        B = rng.integers(0, 2, size=(2 * n, 2 * n), dtype=np.uint8)
        if len(rref(B)[1]) == 2 * n:
            break
    s = int(rng.integers(0, 2 * n)) if dimS is None else dimS
    m = int(rng.integers(s + 1, 2 * n + 1)) if dimN is None else dimN
    return B[:s], B[:m], B


def subspaces_of(m: int) -> list[np.ndarray]:
    """All subspaces of F_2^m as RREF matrices (k x m), built dimension by dimension."""
    vecs = [np.array([(v >> i) & 1 for i in range(m)], dtype=np.uint8) for v in range(1, 2 ** m)]
    out = {(0, b""): np.zeros((0, m), np.uint8)}
    frontier = [np.zeros((0, m), np.uint8)]
    for k in range(1, m + 1):
        new = {}
        for B in frontier:
            for v in vecs:
                R, _ = rref(np.vstack([B, v]) if len(B) else v.reshape(1, -1))
                if len(R) == k:
                    new[(k, R.tobytes())] = R
        frontier = list(new.values())
        out.update(new)
    return list(out.values())


def all_chains(n: int):
    """Every pair S < N (strict) of subspaces of F_2^{2n}, i.e. every linear decoding structure on n qubits with q >= 1."""
    dim = 2 * n
    sub = {m: subspaces_of(m) for m in range(dim + 1)}
    by_dim: dict[int, list[np.ndarray]] = {}
    for B in sub[dim]:
        by_dim.setdefault(len(B), []).append(B)
    for m, Ns in by_dim.items():
        for N in Ns:
            for C in sub[m]:
                if len(C) < m:
                    S = (C.astype(np.int64) @ N.astype(np.int64) % 2).astype(np.uint8) if len(C) else np.zeros((0, dim), np.uint8)
                    yield S, N


def eps_table(st: "Structure", plist) -> tuple[np.ndarray, np.ndarray]:
    """G[i, a] = eps*_A(p_i) for every erased set A (listed by size, then lexicographically) and the sizes |A|."""
    n = st.n
    pats = [A for a in range(n + 1) for A in itertools.combinations(range(n), a)]
    G = np.array([[st.eps_A(p, A) for A in pats] for p in plist])
    return G, np.array([len(A) for A in pats])


def eps_from_table(G_row: np.ndarray, sizes: np.ndarray, n: int, e: float) -> float:
    return float((e ** sizes * (1 - e) ** (n - sizes) * G_row).sum())
