"""Truncated-MPS maximum-likelihood decoder (Bravyi-Suchara-Vargo) for the rotated surface code.

Same problem as ``ExactTNMLDecoder``: the four coset sums Z_c = sum_{g in S} P(R_c g), now contracted with a boundary
matrix-product state whose bond dimension is truncated to ``chi``.

Geometry.  A stabilizer generator is a plaquette (R, C) with R, C in {-1, ..., d-1}; it carries a binary variable
(a spin).  Qubit (r, c) touches the plaquettes NW = (r-1, c-1), NE = (r-1, c), SW = (r, c-1), SE = (r, c).
If r + c is even, NW and SE are X-type and NE and SW are Z-type (and the other way round if r + c is odd); the Pauli on
the qubit is R_j X^x Z^z with x = XOR of the X-type spins and z = XOR of the Z-type spins.  Absent boundary plaquettes
are spins frozen to 0.  The weight of qubit (r, c) therefore couples the four spins NW, NE (plaquette row r-1) and
SW, SE (plaquette row r).

Contraction.  The boundary state is an MPS over the d + 1 plaquette columns of one plaquette row.  Qubit row r is
an MPO of bond dimension 4 (the bond between columns c-1 and c carries the pair (in-spin, out-spin) of column c-1, which
the next factor needs).  After applying the MPO the bond dimension is 4 chi; the state is brought into left-canonical form
by a QR sweep and truncated to chi by a right-to-left SVD sweep, and normalised (the norms are kept as logs).
Cost per decode O(d^2 chi^3), against O(n 2^(d+3)) for the exact decoder.  For chi >= 2^((d+1)/2) no truncation
happens and the result equals the exact decoder to rounding error.  The batch of samples is vectorised (batched
QR/SVD); every sample has its own prior, so the MPO tensors carry a batch index.

Ties between classes (exact ties occur with positive probability under erasures) are detected with a tolerance
``tie_rtol`` on log Z, which must exceed the truncation error; they are then broken uniformly at random.
A class whose truncated partition function is not positive (chi too small) is treated as never maximal and is counted in
``last_invalid``; if all four classes of a sample are invalid an error is raised.
"""
from __future__ import annotations

import numpy as np

from lcd.codes import RotatedSurfaceCode
from lcd.decoders.tn_ml import make_priors

# weight-table index (x + 2 z) of a qubit as a function of (x', y', x, y) = (NW or NE spin, ...):
# even r + c: x-bit = x' ^ y, z-bit = x ^ y';  odd r + c: x-bit = x ^ y', z-bit = x' ^ y
_MTAB = np.zeros((2, 2, 2, 2, 2), dtype=np.intp)
for _par in (0, 1):
    for _xp, _yp, _x, _y in np.ndindex(2, 2, 2, 2):
        if _par == 0:
            _xb, _zb = _xp ^ _y, _x ^ _yp
        else:
            _xb, _zb = _x ^ _yp, _xp ^ _y
        _MTAB[_par, _xp, _yp, _x, _y] = _xb + 2 * _zb


class MPSMLDecoder:
    def __init__(self, code: RotatedSurfaceCode, chi: int = 8, chunk: int = 64, tie_rtol: float = 1e-6,
                 threads: int = 1):
        """chi: bond dimension; chunk: samples per vectorised batch; tie_rtol: tolerance on log Z for ties (must exceed
        the truncation error); threads: worker threads over chunks (the batched LAPACK calls release the GIL)."""
        if chi < 1:
            raise ValueError("chi must be >= 1")
        self.code, self.chi, self.chunk, self.tie_rtol = code, int(chi), int(chunk), float(tie_rtol)
        self.threads = max(1, int(threads))
        d = code.d
        self.W = d + 1
        pres = np.zeros((d + 1, d + 1), bool)               # pres[R + 1, C + 1]
        for R in range(-1, d):
            for C in range(-1, d):
                bulk = 0 <= R <= d - 2 and 0 <= C <= d - 2
                is_x = (R + C) % 2 == 0
                if bulk or (R in (-1, d - 1) and 0 <= C <= d - 2 and is_x) or \
                        (C in (-1, d - 1) and 0 <= R <= d - 2 and not is_x):
                    pres[R + 1, C + 1] = True
        self._present = pres
        self._check_geometry()
        self.last_truncation: np.ndarray | None = None      # per-sample accumulated discarded weight of the last call
        self.last_invalid = 0                                # classes with a non-positive partition function (last call)

    def _check_geometry(self) -> None:
        """The plaquettes defined above must be exactly the stabilizer generators of the code."""
        code, d = self.code, self.code.d
        assert self._present.sum() == code.n - 1
        mine = set()
        for R in range(-1, d):
            for C in range(-1, d):
                if self._present[R + 1, C + 1]:
                    qs = frozenset(r * d + c for r, c in ((R, C), (R, C + 1), (R + 1, C), (R + 1, C + 1))
                                   if 0 <= r < d and 0 <= c < d)
                    mine.add((("X" if (R + C) % 2 == 0 else "Z"), qs))
        theirs = {("X", frozenset(np.flatnonzero(h).tolist())) for h in code.HX} | \
                 {("Z", frozenset(np.flatnonzero(h).tolist())) for h in code.HZ}
        assert mine == theirs, "plaquette geometry does not match the code's check matrices"

    # ------------------------------------------------------------------ one qubit row
    def _apply_row(self, A: list[np.ndarray], r: int, Wq: np.ndarray) -> list[np.ndarray]:
        """MPS (list of (N, L, 2, R) arrays over plaquette row r-1)  ->  MPS over plaquette row r (bond 4 chi)."""
        d, W = self.code.d, self.W
        N = A[0].shape[0]
        out = []
        for i in range(W):
            Ai = A[i]
            a, b = Ai.shape[1], Ai.shape[3]
            mask = (1.0, 1.0 if self._present[r + 1, i] else 0.0)         # output spin y = 1 allowed only if present
            if i == 0:
                Bi = np.zeros((N, a, 1, 2, b, 4))
                for y in (0, 1):
                    for x in (0, 1):
                        Bi[:, :, 0, y, :, 2 * x + y] = Ai[:, :, x, :] * mask[y]
                out.append(Bi.reshape(N, a, 2, b * 4))
                continue
            c = i - 1
            w = Wq[:, r * d + c, :][:, _MTAB[(r + c) % 2]].reshape(N, 4, 2, 2)      # [n, j=(x',y'), x, y]
            if i < W - 1:
                Bi = np.zeros((N, a, 4, 2, b, 4))
                for y in (0, 1):
                    for x in (0, 1):
                        Bi[:, :, :, y, :, 2 * x + y] = (Ai[:, :, x, :][:, :, None, :]
                                                       * w[:, :, x, y][:, None, :, None]) * mask[y]
                out.append(Bi.reshape(N, a * 4, 2, b * 4))
            else:
                Bi = np.zeros((N, a, 4, 2, b))
                for y in (0, 1):
                    for x in (0, 1):
                        Bi[:, :, :, y, :] += (Ai[:, :, x, :][:, :, None, :]
                                             * w[:, :, x, y][:, None, :, None]) * mask[y]
                out.append(Bi.reshape(N, a * 4, 2, b))
        return out

    # ------------------------------------------------------------------ compression
    def _compress(self, B: list[np.ndarray], disc: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """QR sweep to left-canonical form, then right-to-left SVD truncation to chi (optimal in the canonical gauge).
        Returns (MPS, per-sample log norm); the MPS is normalised.  (A single left-to-right SVD sweep without the QR
        sweep was tried and rejected: without canonical form the local singular values are not Schmidt values and it
        loses accuracy even at chi >= the exact Schmidt rank.)"""
        W = self.W
        N = B[0].shape[0]
        for i in range(W - 1):
            L, R = B[i].shape[1], B[i].shape[3]
            Q, Rm = np.linalg.qr(B[i].reshape(N, L * 2, R))
            K = Q.shape[2]
            B[i] = Q.reshape(N, L, 2, K)
            R2 = B[i + 1].shape[3]
            B[i + 1] = np.matmul(Rm, B[i + 1].reshape(N, R, 2 * R2)).reshape(N, K, 2, R2)
        for i in range(W - 1, 0, -1):
            L, R = B[i].shape[1], B[i].shape[3]
            U, s, Vh = np.linalg.svd(B[i].reshape(N, L, 2 * R), full_matrices=False)
            k = min(self.chi, s.shape[1])
            tot = (s ** 2).sum(axis=1)
            disc += np.sqrt(np.divide((s[:, k:] ** 2).sum(axis=1), tot, out=np.zeros(N), where=tot > 0))
            B[i] = Vh[:, :k, :].reshape(N, k, 2, R)
            L0 = B[i - 1].shape[1]
            B[i - 1] = np.matmul(B[i - 1].reshape(N, L0 * 2, L), U[:, :, :k] * s[:, None, :k]).reshape(N, L0, 2, k)
        nrm = np.sqrt((B[0] ** 2).sum(axis=(1, 2, 3)))
        B[0] = B[0] / nrm[:, None, None, None]
        return B, np.log(nrm)

    # ------------------------------------------------------------------ one class
    def _log_partition(self, rx: np.ndarray, rz: np.ndarray, priors: np.ndarray, disc: np.ndarray) -> np.ndarray:
        d, W = self.code.d, self.W
        N, n = rx.shape
        # Wq[n, j, m] = prior_j[(rx ^ xbit) + 2 (rz ^ zbit)] with m = xbit + 2 zbit
        Wq = np.empty((N, n, 4))
        for m in range(4):
            idx = (rx ^ (m & 1)).astype(np.intp) + 2 * (rz ^ (m >> 1)).astype(np.intp)
            Wq[:, :, m] = np.take_along_axis(priors, idx[..., None], axis=2)[..., 0]
        A = []
        for i in range(W):                                   # plaquette row R = -1: uniform over present spins
            t = np.zeros((N, 1, 2, 1))
            t[:, 0, 0, 0] = 1.0
            t[:, 0, 1, 0] = 1.0 if self._present[0, i] else 0.0
            A.append(t)
        logscale = np.zeros(N)
        for r in range(d):
            A, ln = self._compress(self._apply_row(A, r, Wq), disc)
            logscale += ln
        v = np.ones((N, 1))                                  # sum over the spins of the last plaquette row
        for i in range(W):
            v = np.einsum("nl,nlr->nr", v, A[i].sum(axis=2))
        z = v[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return logscale + np.log(np.where(z > 0, z, 0.0))

    # ------------------------------------------------------------------ public API
    def log_class_weights(self, ex: np.ndarray, ez: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """(N, 4) array of log Z_c; c = 0 is the class of the error itself."""
        code = self.code
        ex, ez = ex.astype(np.uint8), ez.astype(np.uint8)
        N = ex.shape[0]
        out = np.empty((N, 4))
        disc = np.zeros(N)

        def work(start: int) -> None:
            sl = slice(start, start + self.chunk)
            for c in range(4):
                rx = ex[sl] ^ ((c & 1) * code.LX)[None, :]
                rz = ez[sl] ^ (((c >> 1) & 1) * code.LZ)[None, :]
                out[sl, c] = self._log_partition(rx, rz, priors[sl], disc[sl])

        starts = range(0, N, self.chunk)
        if self.threads > 1 and N > self.chunk:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.threads) as pool:
                list(pool.map(work, starts))
        else:
            for st in starts:
                work(st)
        self.last_truncation = disc
        return out

    def fail_prob(self, ex: np.ndarray, ez: np.ndarray, priors: np.ndarray) -> np.ndarray:
        """Conditional failure probability of the (truncated) ML decision: 1, 0, or 1 - 1/m for m tied maximisers."""
        L = self.log_class_weights(ex, ez, priors)
        self.last_invalid = int((~np.isfinite(L)).sum())
        if not np.isfinite(L.max(axis=1)).all():
            raise FloatingPointError("all four partition functions non-positive for some sample: increase chi")
        top = L >= L.max(axis=1, keepdims=True) - self.tie_rtol          # -inf (invalid) classes are never maximal
        m = top.sum(axis=1)
        return np.where(top[:, 0], 1.0 - 1.0 / m, 1.0)

    def fail(self, ex: np.ndarray, ez: np.ndarray, p: float, erased: np.ndarray | None = None,
             rng: np.random.Generator | None = None) -> np.ndarray:
        N = ex.shape[0]
        fp = self.fail_prob(ex, ez, make_priors(self.code.n, p, erased, N))
        if ((fp > 0) & (fp < 1)).any():
            if rng is None:
                raise ValueError("ties between classes present: pass an rng for random tie-breaking")
            return rng.random(N) < fp
        return fp > 0.5
