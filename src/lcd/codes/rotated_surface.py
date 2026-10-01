"""Rotated surface code of odd distance d: check matrices and logical operators.

Conventions (fixed; the decoders and tests rely on them):
  * data qubit (r, c), 0 <= r, c < d, has index r*d + c;
  * a plaquette with top-left corner (r, c) covers (r,c), (r,c+1), (r+1,c), (r+1,c+1) (those inside the lattice)
    and is of X type iff (r + c) is even;
  * weight-2 boundary checks: X type on the top and bottom edges, Z type on the left and right edges;
  * HX / HZ: X- and Z-type check matrices.  X errors are detected by HZ, Z errors by HX;
  * LZ = row 0 (a Z-type logical, commutes with HX), LX = column 0 (an X-type logical, commutes with HZ).
    An X error flips the logical observable iff it has odd overlap with LZ; a Z error iff odd overlap with LX.
"""
from __future__ import annotations

import numpy as np


class RotatedSurfaceCode:
    def __init__(self, d: int):
        if d < 3 or d % 2 == 0:
            raise ValueError("d must be an odd integer >= 3")
        self.d, self.n = d, d * d
        idx = lambda r, c: r * d + c
        hx, hz = [], []
        for r in range(-1, d):
            for c in range(-1, d):
                bulk = 0 <= r <= d - 2 and 0 <= c <= d - 2
                kind = "X" if (r + c) % 2 == 0 else "Z"
                qs = [idx(rr, cc) for rr, cc in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1))
                      if 0 <= rr < d and 0 <= cc < d]
                row = np.zeros(self.n, np.uint8)
                row[qs] = 1
                if bulk:
                    (hx if kind == "X" else hz).append(row)
                elif r in (-1, d - 1) and 0 <= c <= d - 2 and kind == "X":
                    hx.append(row)
                elif c in (-1, d - 1) and 0 <= r <= d - 2 and kind == "Z":
                    hz.append(row)
        self.HX, self.HZ = np.array(hx), np.array(hz)
        self.LZ = np.zeros(self.n, np.uint8)
        self.LZ[[idx(0, c) for c in range(d)]] = 1
        self.LX = np.zeros(self.n, np.uint8)
        self.LX[[idx(r, 0) for r in range(d)]] = 1
        self._check()

    def _check(self) -> None:
        HX, HZ, LX, LZ = self.HX, self.HZ, self.LX, self.LZ
        assert HX.shape[0] + HZ.shape[0] == self.n - 1, "expected n-1 independent checks"
        assert not ((HX.astype(int) @ HZ.T.astype(int)) % 2).any(), "X and Z checks must commute"
        assert not ((HX.astype(int) @ LZ.astype(int)) % 2).any()
        assert not ((HZ.astype(int) @ LX.astype(int)) % 2).any()
        assert int(LX.astype(int) @ LZ.astype(int)) % 2 == 1, "logicals must anticommute"
