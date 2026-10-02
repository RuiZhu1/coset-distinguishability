"""Marginal exchange rate between erasure and Pauli noise from stratified tables (README P1, theory section 0.5).

For one work point (p0, e0) and a list of distances d, every distance contributes a stratified table f_d(k, w).  From it

    p_L(d)          = sum_s P_s f_s
    dp_L/dp (d)     = sum_s (dP_s/dp) f_s        (frozen-decoder derivative; equals the ML derivative by the envelope argument)
    dp_L/de (d)     = sum_s (dP_s/de) f_s        (exact: the ML posterior does not depend on e)

The per-distance marginal rate is R^(d) = (dp_L/de) / (dp_L/dp); it must not exceed c = (3/4 - p0)/(1 - e0) (Theorem 3.2).
The suppression exponent is alpha = -(slope of ln p_L against d), and, differentiating the fit,

    d alpha / d theta = - (slope of  g_theta(d) = d ln p_L / d theta  against d),      theta in {p, e}.

Both derivatives of alpha are negative, so the marginal rate of alpha is

    R_alpha = (d alpha/d e) / (d alpha/d p) = s_e / s_p,   s_theta = slope of g_theta against d.

(README and Theorem 3.2(b) printed a spurious minus sign in front of this ratio; the proof of the theorem used the
positive ratio.)  Errors come from a parametric bootstrap of the tables (``StratifiedEstimator.bootstrap`` logic).
"""
from __future__ import annotations

import numpy as np

from lcd.analysis.stratified import Strata


def quantities(strata: Strata, f: np.ndarray, p: float, e: float) -> tuple[float, float, float]:
    """(p_L, dp_L/dp, dp_L/de) for the table f at (p, e)."""
    P = strata.prob(p, e)
    dp, de = strata.dprob(p, e)
    return float((P * f).sum()), float((dp * f).sum()), float((de * f).sum())


def bootstrap_quantities(strata: Strata, fails: np.ndarray, N: np.ndarray, p: float, e: float, B: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(B, 3) replicates of (p_L, dp_L/dp, dp_L/de): each stratum's failures are redrawn ~ Binomial(N, f_smooth)."""
    ft = (fails + 0.5) / (N + 1.0)
    ft[strata.known_zero] = 0.0
    Ni = N.astype(int)
    P = strata.prob(p, e)
    dp, de = strata.dprob(p, e)
    out = np.empty((B, 3))
    for b in range(B):
        fb = np.divide(rng.binomial(Ni, ft), np.maximum(Ni, 1))
        fb[strata.known_zero] = 0.0
        out[b] = (float((P * fb).sum()), float((dp * fb).sum()), float((de * fb).sum()))
    return out


def _slope(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) least-squares slope of y (last axis) against x; y may carry leading replicate axes; w are fixed weights."""
    if w is None:
        w = np.ones_like(x)
    w = w / w.sum()
    xm = (w * x).sum()
    xc = x - xm
    ym = (y * w).sum(axis=-1, keepdims=True)
    return (w * (y - ym) * xc).sum(axis=-1) / (w * xc ** 2).sum()


def fit_rates(ds, pL, dpL_dp, dpL_de, window: tuple[int, int] | None = None, sigma: dict | None = None) -> dict:
    """alpha, d alpha/dp, d alpha/de and R_alpha from arrays over distances (last axis); leading axes are replicates.

    window = (d_min, d_max) restricts the fit to those distances.  sigma = dict(lnp=..., gp=..., ge=...) of per-distance standard
    errors (arrays aligned with ds) switches to inverse-variance weights, separately for the three slopes (fixed across replicates)."""
    ds = np.asarray(ds, float)
    sel = np.ones(ds.size, bool) if window is None else (ds >= window[0]) & (ds <= window[1])
    x = ds[sel]
    pL, dpL_dp, dpL_de = (np.asarray(a, float)[..., sel] for a in (pL, dpL_dp, dpL_de))
    with np.errstate(divide="ignore", invalid="ignore"):
        lnp, gp, ge = np.log(pL), dpL_dp / pL, dpL_de / pL
    wl = wp = we = None
    if sigma is not None:
        wl, wp, we = (1.0 / np.maximum(np.asarray(sigma[k], float)[sel], 1e-12) ** 2 for k in ("lnp", "gp", "ge"))
    s_p, s_e = _slope(x, gp, wp), _slope(x, ge, we)
    alpha = -_slope(x, lnp, wl)
    return dict(alpha=alpha, dalpha_dp=-s_p, dalpha_de=-s_e, R_alpha=s_e / s_p)


def rate_per_distance(pL, dpL_dp, dpL_de):
    """R^(d) = (dp_L/de) / (dp_L/dp), elementwise."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(dpL_de, float) / np.asarray(dpL_dp, float)
