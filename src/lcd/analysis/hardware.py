"""Certifiability margin of a hardware detector error model for the circuit-level Peierls bound (theory, Theorem 4.28).

On published hardware models the bound of Theorem 4.28 is infinite: the noise is above the range in which the weighted
non-backtracking walk sum converges. The certifiability margin s* measures how far: scale every mechanism probability of
the model by s (``p -> min(1/2, s p)``), rebuild the decoder's weights from the scaled model (pymatching, as for the
unscaled one), and evaluate the bound at lam = 1/2 (near-optimal for matched weights); s* is the largest s for which it is
finite. s* < 1: the hardware noise must be 1/s* times lower before the theorem certifies anything; s* > 1: headroom.

Finiteness is monotone in s up to the clipping at 1/2: at lam = 1/2 with matched weights every edge factor
2 sqrt(q (1 - q)) (and every split factor) increases with the mechanism probabilities. The bisection relies on this.
For Theorem 4.32 (``method="4.32"``) the edge factors also increase with q, but the set of geodesics depends on the
weights, so monotonicity is not guaranteed; the returned s_star is always a point at which finiteness was verified.

    from lcd.analysis.hardware import certifiability_margin, scale_dem
    res = certifiability_margin(dem)       # dict(s_star, s_upper, capped, below, evaluations)
"""
from __future__ import annotations

import numpy as np
import stim

from lcd.analysis.circuit_peierls import peierls_bound
from lcd.analysis.circuit_peierls_geodesic import gap_bound_nb, geodesic_bound

#: lam grid for Theorem 4.32 (its optimum moves away from 1/2)
GEODESIC_LAMS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5)


def scale_dem(dem: stim.DetectorErrorModel, s: float) -> stim.DetectorErrorModel:
    """The flattened model with every error probability p replaced by min(1/2, s p); other instructions are kept."""
    if s < 0:
        raise ValueError("scale factor must be nonnegative")
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            out.append("error", [min(0.5, inst.args_copy()[0] * s)], inst.targets_copy())
        else:
            out.append(inst)
    return out


def bound_at_scale(dem: stim.DetectorErrorModel, s: float, lam: float = 0.5, method: str = "4.28") -> float:
    """The bound for the model scaled by s (pymatching weights of the scaled model): Theorem 4.28 at one lam, or
    Theorem 4.32 (geodesic refinement) minimized over GEODESIC_LAMS, or Theorem 4.34 (gap refinement) on a lam grid
    (``lam`` is then ignored)."""
    if method == "4.32":
        return float(geodesic_bound(scale_dem(dem, s), lams=GEODESIC_LAMS)["bound"])
    if method == "4.34":
        return float(gap_bound_nb(scale_dem(dem, s), lams=(0.2, 0.3, 0.4, 0.5))["bound"])
    return float(peierls_bound(scale_dem(dem, s), lams=[lam])["bound"])


def certifiability_margin(dem: stim.DetectorErrorModel, s_max: float = 4.0, rtol: float = 1e-3, lam: float = 0.5,
                          s_min: float = 1e-6, method: str = "4.28") -> dict:
    """Largest s in (0, s_max] with a finite bound at lam, by geometric bisection to relative precision rtol.

    Returns dict(s_star, s_upper, capped, below, evaluations): the bound is finite at s_star and infinite at s_upper
    (s_upper / s_star <= 1 + rtol); capped = True if it is finite at s_max (then s_star = s_max, s_upper = None);
    below = True if it is infinite even at s_min (then s_star = None). ValueError / NotImplementedError of the bound
    (model not covered by the theorem) propagate.
    """
    n = 0

    def finite(s):
        nonlocal n
        n += 1
        return bool(np.isfinite(bound_at_scale(dem, s, lam, method)))

    if finite(s_max):
        return dict(s_star=float(s_max), s_upper=None, capped=True, below=False, evaluations=n)
    if s_max > 1 and finite(1.0):
        lo, hi = 1.0, float(s_max)
    else:
        hi = min(1.0, float(s_max))
        lo = hi / 2
        while not finite(lo):
            hi, lo = lo, lo / 2
            if lo < s_min:
                return dict(s_star=None, s_upper=float(hi), capped=False, below=True, evaluations=n)
    while hi / lo > 1 + rtol:
        mid = float(np.sqrt(lo * hi))
        if finite(mid):
            lo = mid
        else:
            hi = mid
    return dict(s_star=float(lo), s_upper=float(hi), capped=False, below=False, evaluations=n)
