"""Exact maximum-likelihood (coset) decoder for the rotated surface code, with per-qubit priors.

Given a sampled error E (used only as a reference Pauli with the right syndrome) the decoder computes the four
coset sums

    Z_c = sum_{g in S} P(R_c * g),     R_c = E * L_c,   L_c in {I, Xbar, Zbar, Ybar},

where S is the stabilizer group and P(E) = prod_j p_j(E_j) is the product prior.  Class c = 0 is the class of E,
so the ML decoder succeeds iff Z_0 is the maximum.  Using E as the reference is legitimate: the decoder's decision
depends on E only through its syndrome (the class labels are relative to the reference; see theory/sec1,
"label choice does not matter").

Contraction.  Write g as a product of stabilizer generators with binary variables a_i (X-type) and b_i (Z-type).
Qubit j then carries the Pauli R_j * X^{x} Z^{z} with x = XOR of the a_i touching j and z = XOR of the b_i touching j,
so P(R g) = prod_j w_j(x_j, z_j): a two-dimensional classical partition function.  We contract it exactly by sweeping
the qubits in row-major order and keeping only the "frontier" variables (about d + 1 binary variables), i.e. a
transfer matrix of size 2^(d+1).  The cost per decode is O(n 2^(d+1)); the result is exact (no bond-dimension
truncation), which makes this decoder both the reference implementation for d <= ~11 and the oracle against which a
truncated (Bravyi-Suchara-Vargo) MPS contraction should be validated.  The batch of samples is vectorized.

Priors are given as an array of shape (N, n, 4) in the order (I, X, Z, Y), up to a positive factor per qubit
(only ratios matter): a non-erased qubit has (1, r, r, r) with r = p / (3 (1 - p)); an erased qubit has (1, 1, 1, 1).
Ties between classes (which occur with positive probability under erasures, e.g. when the erased set contains the
support of a logical operator) are broken uniformly at random, which attains the Bayes optimum.
"""
from __future__ import annotations

import numpy as np

from lcd.codes import RotatedSurfaceCode


def make_priors(n: int, p: float, erased: np.ndarray | None, N: int | None = None) -> np.ndarray:
    """Priors (N, n, 4) in the order (I, X, Z, Y) for depolarizing rate p with optional erased positions."""
    r = p / (3.0 * (1.0 - p))
    base = np.array([1.0, r, r, r])
    if erased is None:
        if N is None:
            raise ValueError("N is required when erased is None")
        return np.broadcast_to(base, (N, n, 4))
    return np.where(erased[..., None], 1.0, base)


class ExactTNMLDecoder:
    def __init__(self, code: RotatedSurfaceCode, max_chunk_elements: float = 6e6):
        self.code = code
        n = code.n
        nx = code.HX.shape[0]
        qvars = []
        for j in range(n):
            xs = list(np.flatnonzero(code.HX[:, j]))
            zs = [nx + int(i) for i in np.flatnonzero(code.HZ[:, j])]
            qvars.append([int(i) for i in xs] + zs)
        last: dict[int, int] = {}
        for j, vs in enumerate(qvars):
            for v in vs:
                last[v] = j
        # Static contraction plan: identical for every call.
        self._plan = []
        order: list[int] = []
        frontier = 0
        for j in range(n):
            vs = qvars[j]
            n_new = sum(1 for v in vs if v not in order)
            order = order + [v for v in vs if v not in order]
            pos = [order.index(v) for v in vs]
            perm = np.argsort(pos)
            gshape = [1] * len(order)
            for p_ in pos:
                gshape[p_] = 2
            m = len(vs)
            # table[a_1..a_m] = x + 2 z with x = XOR of the X-type variables, z = XOR of the Z-type variables
            table = np.zeros((2,) * m, dtype=np.intp)
            for bits in np.ndindex(*(2,) * m):
                x = z = 0
                for v, b in zip(vs, bits):
                    if v < nx:
                        x ^= b
                    else:
                        z ^= b
                table[bits] = x + 2 * z
            frontier = max(frontier, len(order))
            elim = [v for v in order if last[v] == j]
            axes = tuple(1 + order.index(v) for v in elim)
            self._plan.append((n_new, table, tuple(1 + int(i) for i in perm), tuple(gshape), axes))
            order = [v for v in order if last[v] != j]
        assert not order
        self.frontier = frontier
        self._chunk = max(1, int(max_chunk_elements / 2 ** frontier))

    # ------------------------------------------------------------------ contraction
    def _coset_sum(self, rx: np.ndarray, rz: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """Z = sum_g P(R g) (up to the per-qubit prior scale) for reference Paulis (rx, rz), shapes (N, n)."""
        N, n = rx.shape
        ar = np.arange(N)
        state = np.ones((N,))
        for j, (n_new, table, perm, gshape, axes) in enumerate(self._plan):
            idx = [(rx[:, j] ^ (m & 1)).astype(np.intp) + 2 * (rz[:, j] ^ (m >> 1)).astype(np.intp) for m in range(4)]
            w = np.stack([priors[ar, j, ix] for ix in idx], axis=1)            # (N, 4), indexed by x + 2z
            g = w[:, table]                                                    # (N, 2, ..., 2)
            g = np.transpose(g, (0,) + perm).reshape((N,) + gshape)
            state = state.reshape(state.shape + (1,) * n_new) * g
            if axes:
                state = state.sum(axis=axes)
        return state.reshape(N)

    def class_weights(self, ex: np.ndarray, ez: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """(N, 4) array of coset sums Z_c; c = 0 is the class of the error itself."""
        code = self.code
        ex, ez = ex.astype(np.uint8), ez.astype(np.uint8)
        out = np.empty((ex.shape[0], 4))
        for s in range(0, ex.shape[0], self._chunk):
            sl = slice(s, s + self._chunk)
            for c in range(4):
                rx = ex[sl] ^ ((c & 1) * code.LX)[None, :]
                rz = ez[sl] ^ (((c >> 1) & 1) * code.LZ)[None, :]
                out[sl, c] = self._coset_sum(rx, rz, priors[sl])
        return out

    # ------------------------------------------------------------------ decisions
    def fail_prob(self, ex: np.ndarray, ez: np.ndarray, priors: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
        """Conditional failure probability of the ML decoder given each error: 1 if class 0 is not among the
        maximisers, 1 - 1/m if class 0 is one of m tied maximisers (uniform tie-breaking), 0 if it is the unique one."""
        Z = self.class_weights(ex, ez, priors)
        M = Z.max(axis=1)
        if (M <= 0).any():
            raise FloatingPointError("coset sums underflowed; reduce n, p or use log-space contraction")
        top = Z >= M[:, None] * (1.0 - rtol)
        m = top.sum(axis=1)
        return np.where(top[:, 0], 1.0 - 1.0 / m, 1.0)

    def fail(self, ex: np.ndarray, ez: np.ndarray, p: float, erased: np.ndarray | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
        """Bool array: ML decoder failures at depolarizing rate p (erased qubits have a uniform prior)."""
        N = ex.shape[0]
        priors = make_priors(self.code.n, p, erased, N)
        fp = self.fail_prob(ex, ez, priors)
        if ((fp > 0) & (fp < 1)).any():
            if rng is None:
                raise ValueError("ties between classes present: pass an rng for random tie-breaking")
            return rng.random(N) < fp
        return fp > 0.5
