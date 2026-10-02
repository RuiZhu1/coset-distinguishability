"""MWPM decoder (PyMatching) for the rotated surface code under code-capacity Pauli + erasure noise.

X and Z errors are decoded independently.  Non-erased qubits carry edge weight 1, erased qubits edge weight 0 (their
X and Z marginals are 1/2, i.e. log-likelihood weight ln(1) = 0).  Consequences:
  * the decoder does not depend on the noise rates (p, e) -> the failure probability f(stratum) of a stratum
    (fixed numbers of erasures and Pauli errors) is rate independent, one table serves every (p, e), and the
    derivatives dp_L/dp, dp_L/de computed from the table are exact for this decoder (no envelope argument needed);
  * Y correlations are ignored, so this is NOT the maximum-likelihood decoder.  Never report its results as ML.

Shots without erasures are decoded in one batch; a shot with erasures gets its own matching graph (about 0.1 ms per
graph at d = 15).  Ties between minimum-weight corrections are broken deterministically by PyMatching; since an erased
qubit carries a uniform random Pauli, a tie between logical classes still fails with probability 1/2 on average.
"""
from __future__ import annotations

import numpy as np
import pymatching

from lcd.codes import RotatedSurfaceCode


class MWPMCodeCapacity:
    def __init__(self, code: RotatedSurfaceCode):
        self.code = code
        self._hx = code.HX.astype(np.float32)
        self._hz = code.HZ.astype(np.float32)
        self._lx = code.LX.astype(np.float32)
        self._lz = code.LZ.astype(np.float32)
        self._mx = pymatching.Matching.from_check_matrix(code.HZ, faults_matrix=code.LZ[None, :])
        self._mz = pymatching.Matching.from_check_matrix(code.HX, faults_matrix=code.LX[None, :])

    def fail(self, ex: np.ndarray, ez: np.ndarray, erased: np.ndarray | None = None) -> np.ndarray:
        """ex, ez: (N, n) 0/1 arrays of X- and Z-error indicators; erased: optional (N, n) bool array of erasure flags.
        Returns a bool array: logical failure."""
        ex, ez = ex.astype(np.float32), ez.astype(np.float32)
        sx = np.mod(ex @ self._hz.T, 2).astype(np.uint8)       # X errors flip Z checks
        sz = np.mod(ez @ self._hx.T, 2).astype(np.uint8)
        ox = np.mod(ex @ self._lz, 2).astype(np.uint8)         # actual logical flips
        oz = np.mod(ez @ self._lx, 2).astype(np.uint8)
        px = self._mx.decode_batch(sx)[:, 0]
        pz = self._mz.decode_batch(sz)[:, 0]
        if erased is not None:
            code = self.code
            for i in np.flatnonzero(erased.any(axis=1)):
                w = np.where(erased[i], 0.0, 1.0)
                if sx[i].any():
                    mx = pymatching.Matching.from_check_matrix(code.HZ, weights=w, faults_matrix=code.LZ[None, :])
                    px[i] = mx.decode(sx[i])[0]
                if sz[i].any():
                    mz = pymatching.Matching.from_check_matrix(code.HX, weights=w, faults_matrix=code.LX[None, :])
                    pz[i] = mz.decode(sz[i])[0]
        return (px != ox) | (pz != oz)
