"""MWPM decoder (PyMatching) for the rotated surface code under code-capacity Pauli noise.

X and Z errors are decoded independently with uniform edge weights.  Consequences:
  * the decoder does not depend on the noise rates (p, e) -> the failure probability f(stratum) of a stratum
    (fixed number of errors) is rate independent, one table serves every (p, e);
  * Y correlations are ignored, so this is NOT the maximum-likelihood decoder.  Never report its results as ML.

Erasures are not supported yet (they need per-shot zero-weight edges).
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

    def fail(self, ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
        """ex, ez: (N, n) 0/1 arrays of X- and Z-error indicators. Returns a bool array: logical failure."""
        ex, ez = ex.astype(np.float32), ez.astype(np.float32)
        sx = np.mod(ex @ self._hz.T, 2).astype(np.uint8)       # X errors flip Z checks
        sz = np.mod(ez @ self._hx.T, 2).astype(np.uint8)
        ox = np.mod(ex @ self._lz, 2).astype(np.uint8)         # actual logical flips
        oz = np.mod(ez @ self._lx, 2).astype(np.uint8)
        px = self._mx.decode_batch(sx)[:, 0]
        pz = self._mz.decode_batch(sz)[:, 0]
        return (px != ox) | (pz != oz)
