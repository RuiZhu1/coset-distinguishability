"""Crossing point of two logical-error curves (threshold estimate from a pair of distances)."""
from __future__ import annotations

import numpy as np


def crossing(xs, a, b, se_a, se_b, boot: int = 400, seed: int = 0) -> tuple[float, float]:
    """Crossing of curve ``a`` (smaller distance) and curve ``b`` (larger distance).

    Root of a weighted straight-line fit to ``b - a`` (weights 1 / sqrt(se_a^2 + se_b^2)), with a parametric-bootstrap
    standard error.  Robust to grid points where the difference is exactly zero.  Returns (root, se); the root is NaN
    if the fitted slope is not positive (curves do not cross in the window).
    """
    xs = np.asarray(xs, float)
    diff = np.asarray(b, float) - np.asarray(a, float)
    se = np.hypot(np.asarray(se_a, float), np.asarray(se_b, float))

    def root(y: np.ndarray) -> float:
        slope, icpt = np.polyfit(xs, y, 1, w=1.0 / se)
        return float(-icpt / slope) if slope > 0 else float("nan")

    rng = np.random.default_rng(seed)
    reps = np.array([root(diff + se * rng.standard_normal(len(xs))) for _ in range(boot)])
    return root(diff), float(np.nanstd(reps))
