import numpy as np
import pytest
from scipy.stats import binom

from lcd.analysis import Strata, StratifiedEstimator
from lcd.codes import RotatedSurfaceCode
from lcd.decoders import MWPMCodeCapacity
from lcd.noise import sample_iid, sample_stratum


def toy_sampler(ftab):
    """Sampler whose failure probability on stratum (k, w) is exactly ftab(k, w)."""
    def sampler(k, w, N, rng):
        return rng.random(N) < ftab(k, w)
    return sampler


def f1d(k, w):                       # zero for w <= 3, smooth increase afterwards
    return float(np.clip((w - 3) / 12.0, 0.0, 1.0)) ** 2


def f2d(k, w):                       # zero for 2w + k < 8
    return float(np.clip((2 * w + k - 7) / 25.0, 0.0, 1.0))


def exact(strata, ftab, p, e):
    f = np.array([ftab(int(k), int(w)) for k, w in zip(strata.k, strata.w)])
    return float((strata.prob(p, e) * f).sum()), f


def test_strata_probabilities_sum_to_one_and_known_zero():
    S = Strata(n=40, d=7, targets=[(0.05, 0.02)])
    assert abs(S.prob(0.05, 0.02).sum() + S.omitted_mass(0.05, 0.02) - 1) < 1e-12
    assert S.omitted_mass(0.05, 0.02) < 1e-9
    assert (S.known_zero == (2 * S.w + S.k < 7)).all()


def test_dprob_matches_finite_differences_including_e_zero():
    S = Strata(n=30, d=None, targets=[(0.04, 0.03), (0.04, 0.0)])
    for p, e in [(0.04, 0.03), (0.04, 0.0), (0.0, 0.03)]:
        dp, de = S.dprob(p, e)
        h = 1e-9      # one-sided at the boundary has an O(h * P''(0)) error; keep h tiny
        # one-sided at the boundary, central inside
        p_hi, p_lo = p + h, max(p - h, 0.0)
        e_hi, e_lo = e + h, max(e - h, 0.0)
        np.testing.assert_allclose(dp, (S.prob(p_hi, e) - S.prob(p_lo, e)) / (p_hi - p_lo), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(de, (S.prob(p, e_hi) - S.prob(p, e_lo)) / (e_hi - e_lo), rtol=1e-4, atol=1e-6)


def test_estimator_matches_exact_value_1d():
    S = Strata(n=40, d=7, targets=[(0.05, 0.0)])
    est = StratifiedEstimator(S, toy_sampler(f1d))
    est.run(200_000, np.random.default_rng(5), n0=500)
    truth, _ = exact(S, f1d, 0.05, 0.0)
    r = est.estimate(0.05, 0.0)
    assert abs(r.p_L - truth) < 4 * r.se
    assert r.lo <= truth <= r.hi
    assert r.rel_se < 0.05
    # budget respected, known-zero strata never sampled
    assert 0.9 * 200_000 <= est.N.sum() <= 200_000 + 1
    assert est.N[S.known_zero].sum() == 0


def test_estimator_matches_exact_value_with_erasure_and_derivatives():
    S = Strata(n=30, d=8, targets=[(0.04, 0.03)])
    est = StratifiedEstimator(S, toy_sampler(f2d))
    est.run(400_000, np.random.default_rng(6), n0=200)
    truth, ftrue = exact(S, f2d, 0.04, 0.03)
    r = est.estimate(0.04, 0.03)
    assert abs(r.p_L - truth) < 4 * r.se
    assert r.lo <= truth <= r.hi
    # analytic derivatives vs finite differences of the same table (fixed f)
    f = est.f_hat()
    h = 1e-6
    dp, de = est.derivatives(0.04, 0.03)
    fd_p = (est.p_L_from(S, f, 0.04 + h, 0.03) - est.p_L_from(S, f, 0.04 - h, 0.03)) / (2 * h)
    fd_e = (est.p_L_from(S, f, 0.04, 0.03 + h) - est.p_L_from(S, f, 0.04, 0.03 - h)) / (2 * h)
    assert dp == pytest.approx(fd_p, rel=1e-5)
    assert de == pytest.approx(fd_e, rel=1e-5)
    # and the derivatives of the *true* p_L are recovered within statistical error
    tdp, tde = est.derivatives_from(S, ftrue, 0.04, 0.03)
    boot = est.bootstrap(lambda fb: est.derivatives_from(S, fb, 0.04, 0.03)[1], 40, np.random.default_rng(7))
    assert abs(de - tde) < 5 * boot.std()


def test_certified_interval_coverage():
    S = Strata(n=40, d=7, targets=[(0.05, 0.0)])
    truth, _ = exact(S, f1d, 0.05, 0.0)
    rng = np.random.default_rng(8)
    cover = 0
    reps = 60
    for _ in range(reps):
        est = StratifiedEstimator(S, toy_sampler(f1d))
        est.run(6_000, rng, n0=40)
        r = est.estimate(0.05, 0.0, delta=0.1)
        cover += r.lo <= truth <= r.hi
    assert cover / reps >= 0.9


def test_bootstrap_se_is_consistent_with_analytic_se():
    S = Strata(n=40, d=7, targets=[(0.05, 0.0)])
    est = StratifiedEstimator(S, toy_sampler(f1d))
    est.run(100_000, np.random.default_rng(9), n0=500)
    r = est.estimate(0.05, 0.0)
    boot = est.bootstrap(lambda fb: est.p_L_from(S, fb, 0.05, 0.0), 200, np.random.default_rng(10))
    assert boot.std() == pytest.approx(r.se, rel=0.3)


def test_multi_target_allocation_serves_all_targets():
    targets = [(0.03, 0.0), (0.08, 0.0)]
    S = Strata(n=40, d=7, targets=targets)
    est = StratifiedEstimator(S, toy_sampler(f1d))
    est.run(300_000, np.random.default_rng(11), n0=300)
    for p, e in targets:
        truth, _ = exact(S, f1d, p, e)
        r = est.estimate(p, e)
        assert abs(r.p_L - truth) < 4 * r.se
        assert r.rel_se < 0.1


def test_stratified_mwpm_agrees_with_naive_monte_carlo():
    """d = 5, p = 0.10 (e = 0): stratified and naive estimates of the MWPM failure rate agree."""
    d, p = 5, 0.10
    code = RotatedSurfaceCode(d)
    dec = MWPMCodeCapacity(code)
    S = Strata(code.n, d, targets=[(p, 0.0)])
    est = StratifiedEstimator(S, lambda k, w, N, rng: dec.fail(*sample_stratum(code.n, k, w, N, rng)[:2]))
    rng = np.random.default_rng(12)
    est.run(300_000, rng, n0=1000)
    r = est.estimate(p, 0.0)
    Nn = 400_000
    ex, ez = sample_iid(code.n, p, Nn, rng)
    fn = dec.fail(ex, ez).sum()
    pn, sn = fn / Nn, np.sqrt(fn / Nn * (1 - fn / Nn) / Nn)
    assert abs(r.p_L - pn) < 4 * np.hypot(r.se, sn)
    assert r.lo <= pn <= r.hi
