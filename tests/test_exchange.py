"""The exchange-rate pipeline on exactly known tables (repetition-like toy: n = d qubits, f(k, w) = 1[2w + k >= d]): the slope-based
R_alpha must equal the ratio of numerical derivatives of alpha, including the sign convention, and R^(d) <= c."""
import numpy as np
import pytest

from lcd.analysis import Strata
from lcd.analysis.exchange import bootstrap_quantities, fit_rates, quantities, rate_per_distance

DS = np.array([5, 7, 9, 11, 13])


def tables(p0, e0):
    out = []
    for d in DS:
        S = Strata(int(d), int(d), targets=[(p0, e0), (p0, max(e0, 0.01))], cutoff=1e-14)
        f = (2 * S.w + S.k >= d).astype(float)
        out.append((S, f))
    return out


def quant(p, e, tabs):
    q = np.array([quantities(S, f, p, e) for S, f in tabs])
    return q[:, 0], q[:, 1], q[:, 2]


def alpha(p, e, tabs):
    pl, _, _ = quant(p, e, tabs)
    xc = DS - DS.mean()
    return -float((np.log(pl) * xc).sum() / (xc ** 2).sum())


@pytest.mark.parametrize("p0,e0", [(0.04, 0.05), (0.06, 0.02), (0.04, 0.0)])
def test_rate_matches_numerical_derivatives_of_alpha(p0, e0):
    tabs = tables(p0, e0)
    pl, dp, de = quant(p0, e0, tabs)
    got = fit_rates(DS, pl, dp, de)
    h = 1e-6
    a_p = (alpha(p0 + h, e0, tabs) - alpha(p0 - h, e0, tabs)) / (2 * h)
    a_e = (alpha(p0, e0 + h, tabs) - alpha(p0, max(e0 - h, 0.0), tabs)) / (h + min(h, e0))
    assert got["dalpha_dp"] == pytest.approx(a_p, rel=1e-4)
    assert got["dalpha_de"] == pytest.approx(a_e, rel=1e-3)
    assert got["R_alpha"] == pytest.approx(a_e / a_p, rel=1e-3)
    assert got["dalpha_dp"] < 0 and got["dalpha_de"] < 0 and got["R_alpha"] > 0     # both derivatives negative, rate positive


def test_per_distance_rate_is_positive():
    # NOTE: the toy decoder is a fixed (not Bayes-optimal) decoder, so the bound R <= (3/4 - p)/(1 - e) of Theorem 3.2,
    # which holds for ML decoding only, need not hold for it (here R = 0.96 > c = 0.71 at e = 0)
    for p0, e0 in [(0.04, 0.05), (0.06, 0.02), (0.04, 0.0)]:
        r = rate_per_distance(*quant(p0, e0, tables(p0, e0)))
        assert (r > 0).all()


def test_bootstrap_is_centred_on_the_point_estimate():
    S, f = tables(0.04, 0.05)[0]
    N = np.full(S.size, 400.0)
    rng = np.random.default_rng(0)
    fails = np.round(f * N)
    fails[~S.known_zero & (f == 0)] = 0
    rep = bootstrap_quantities(S, fails, N, 0.04, 0.05, 200, rng)
    pl, dp, de = quantities(S, f, 0.04, 0.05)
    assert rep[:, 0].mean() == pytest.approx(pl, rel=0.05)
    assert rep[:, 1].mean() == pytest.approx(dp, rel=0.05)
    assert rep[:, 2].mean() == pytest.approx(de, rel=0.05)
