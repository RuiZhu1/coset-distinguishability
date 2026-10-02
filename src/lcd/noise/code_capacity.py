"""Code-capacity Pauli + erasure noise.

Model N(p, e) (README sec. 2, theory/sec2): each qubit independently
  * with prob. e is erased (flag = 1, Pauli uniform on {I, X, Y, Z});
  * otherwise, with prob. p, suffers X, Y or Z (each p/3).
Errors are returned as X- and Z-indicator arrays: X -> (1,0), Z -> (0,1), Y -> (1,1).
"""
from __future__ import annotations

import numpy as np


def sample_iid(n: int, p: float, N: int, rng: np.random.Generator):
    """N i.i.d. depolarizing samples (no erasure). Returns (ex, ez), each (N, n) uint8."""
    err = rng.random((N, n)) < p
    typ = rng.integers(0, 3, (N, n))                    # 0=X 1=Y 2=Z
    return (err & (typ <= 1)).astype(np.uint8), (err & (typ >= 1)).astype(np.uint8)


def sample_iid_erasure(n: int, p: float, e: float, N: int, rng: np.random.Generator):
    """N i.i.d. samples of N(p, e): returns (ex, ez, erased); erased qubits carry a uniform Pauli in {I, X, Y, Z}."""
    erased = rng.random((N, n)) < e
    t = rng.integers(0, 4, (N, n))                                   # 0=I 1=X 2=Z 3=Y
    err = (~erased) & (rng.random((N, n)) < p)
    typ = rng.integers(0, 3, (N, n))                                 # 0=X 1=Y 2=Z
    ex = (np.where(erased, t & 1, err & (typ <= 1))).astype(np.uint8)
    ez = (np.where(erased, t >> 1, err & (typ >= 1))).astype(np.uint8)
    return ex, ez, erased


def sample_bitflip(n: int, p: float, N: int, rng: np.random.Generator):
    """N i.i.d. bit-flip (X-only) samples. Returns (ex, ez) with ez = 0."""
    ex = (rng.random((N, n)) < p).astype(np.uint8)
    return ex, np.zeros_like(ex)


def sample_stratum(n: int, k: int, w: int, N: int, rng: np.random.Generator):
    """N samples conditioned on exactly k erased qubits and exactly w Pauli errors on non-erased qubits.

    Conditional on (k, w) the model N(p, e) is uniform over placements (k + w distinct qubits, the first k erased)
    and uniform over {X,Y,Z} for each error, so the sample law does not depend on (p, e).
    Returns (ex, ez, erased) with erased a bool (N, n) array; erased qubits carry a uniform Pauli in {I,X,Y,Z}.
    """
    if k + w > n:
        raise ValueError("k + w > n")
    ex = np.zeros((N, n), np.uint8)
    ez = np.zeros((N, n), np.uint8)
    erased = np.zeros((N, n), bool)
    m = k + w
    if m == 0:
        return ex, ez, erased
    sub = np.argpartition(rng.random((N, n)), m - 1, axis=1)[:, :m]      # uniformly random m-subset per row
    sub = rng.permuted(sub, axis=1)                                      # uniformly random order within it
    rows = np.arange(N)[:, None]
    if k:
        er = sub[:, :k]
        erased[rows, er] = True
        t = rng.integers(0, 4, (N, k))                                   # 0=I 1=X 2=Z 3=Y
        ex[rows, er] = (t & 1)
        ez[rows, er] = (t >> 1)
    if w:
        pe = sub[:, k:]
        t = rng.integers(0, 3, (N, w))                                   # 0=X 1=Y 2=Z
        ex[rows, pe] = (t <= 1)
        ez[rows, pe] = (t >= 1)
    return ex, ez, erased
