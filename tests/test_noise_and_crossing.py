import numpy as np

from lcd.analysis import crossing
from lcd.noise import sample_bitflip, sample_iid_erasure


def test_sample_iid_erasure_rates_and_uniformity():
    rng = np.random.default_rng(0)
    n, p, e, N = 30, 0.1, 0.2, 40000
    ex, ez, er = sample_iid_erasure(n, p, e, N, rng)
    assert abs(er.mean() - e) < 0.003
    nonerased = ~er
    err = ((ex | ez).astype(bool)) & nonerased
    assert abs(err.sum() / nonerased.sum() - p) < 0.003
    freq = np.bincount((ex + 2 * ez)[er], minlength=4) / er.sum()
    assert np.abs(freq - 0.25).max() < 0.01          # erased qubits: uniform over I, X, Z, Y


def test_sample_iid_erasure_without_errors_leaves_nonerased_clean():
    rng = np.random.default_rng(1)
    ex, ez, er = sample_iid_erasure(20, 0.0, 0.3, 2000, rng)
    assert not ex[~er].any() and not ez[~er].any()


def test_sample_bitflip_is_x_only():
    ex, ez = sample_bitflip(25, 0.1, 20000, np.random.default_rng(2))
    assert not ez.any() and abs(ex.mean() - 0.1) < 0.003


def test_crossing_recovers_known_crossing_and_handles_exact_zeros():
    xs = np.linspace(0.1, 0.3, 9)
    a = 0.5 + 1.0 * (xs - 0.2)             # smaller distance: shallower
    b = 0.5 + 2.0 * (xs - 0.2)             # larger distance: steeper, same value at 0.2
    c, se = crossing(xs, a, b, np.full(9, 0.002), np.full(9, 0.002))
    assert abs(c - 0.2) < 1e-9 and se < 0.01
    b0 = b.copy()
    b0[4] = a[4]                           # a grid point with an exactly vanishing difference must not break the fit
    c0, _ = crossing(xs, a, b0, np.full(9, 0.002), np.full(9, 0.002))
    assert abs(c0 - 0.2) < 1e-9


def test_crossing_is_nan_when_curves_do_not_cross():
    xs = np.linspace(0.1, 0.3, 7)
    c, _ = crossing(xs, 0.5 + xs, 0.4 + xs, np.full(7, 0.01), np.full(7, 0.01))
    assert np.isnan(c)
