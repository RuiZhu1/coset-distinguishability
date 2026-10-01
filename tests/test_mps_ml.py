import numpy as np
import pytest

from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MPSMLDecoder
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_stratum


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_plaquette_geometry_matches_the_code(d):
    MPSMLDecoder(RotatedSurfaceCode(d), chi=2)          # the constructor asserts the geometry


def test_rejects_bad_chi():
    with pytest.raises(ValueError):
        MPSMLDecoder(RotatedSurfaceCode(3), chi=0)


@pytest.mark.parametrize("d,k,w", [(3, 0, 2), (5, 0, 4), (5, 2, 3), (7, 3, 3), (7, 0, 6)])
def test_equals_exact_decoder_when_nothing_is_truncated(d, k, w):
    """chi >= the largest Schmidt rank 2^((d+1)/2): the MPS contraction is exact (erasures included)."""
    rng = np.random.default_rng(d * 100 + k * 10 + w)
    code = RotatedSurfaceCode(d)
    ex, ez, er = sample_stratum(code.n, k, w, 25, rng)
    pri = make_priors(code.n, 0.07, er)
    mps = MPSMLDecoder(code, chi=2 ** ((d + 1) // 2 + 1))
    L = mps.log_class_weights(ex, ez, pri)
    Z = ExactTNMLDecoder(code).class_weights(ex, ez, pri)
    np.testing.assert_allclose(np.exp(L - np.log(Z)), 1.0, rtol=0, atol=1e-7)
    assert mps.last_truncation.max() < 1e-9             # nothing of significance discarded
    assert mps.last_invalid == 0


def test_decisions_agree_with_exact_decoder_at_moderate_chi_d7():
    rng = np.random.default_rng(7)
    code = RotatedSurfaceCode(7)
    exd, mps = ExactTNMLDecoder(code), MPSMLDecoder(code, chi=8)
    changed = total = 0
    for k, w in [(0, 5), (0, 7), (0, 9), (2, 4), (3, 6)]:
        ex, ez, er = sample_stratum(code.n, k, w, 120, rng)
        pri = make_priors(code.n, 0.08, er)
        changed += int((exd.fail_prob(ex, ez, pri) != mps.fail_prob(ex, ez, pri)).sum())
        total += 120
    assert changed == 0, f"{changed}/{total} decisions changed"


def test_truncation_error_is_reported_and_shrinks_with_chi():
    rng = np.random.default_rng(8)
    code = RotatedSurfaceCode(7)
    ex, ez, er = sample_stratum(code.n, 0, 8, 30, rng)
    pri = make_priors(code.n, 0.08, er)
    disc = []
    for chi in (2, 4, 8, 16):
        mps = MPSMLDecoder(code, chi=chi)
        mps.log_class_weights(ex, ez, pri)
        disc.append(mps.last_truncation.max())
    assert disc[0] > disc[1] > disc[2] >= disc[3]
    assert disc[0] > 1e-3 and disc[3] < 1e-9


@pytest.mark.parametrize("d,col,chi", [(5, 2, 2), (7, 3, 3), (7, 0, 2)])
def test_exact_ties_survive_truncation(d, col, chi):
    """Erasing a whole column (a support of an X-logical) ties the classes {I, Xbar} exactly: failure probability 1/2."""
    rng = np.random.default_rng(d + col)
    code = RotatedSurfaceCode(d)
    N = 30
    erased = np.zeros((N, code.n), bool)
    erased[:, [r * d + col for r in range(d)]] = True
    t = rng.integers(0, 4, (N, code.n))
    ex, ez = ((t & 1) * erased).astype(np.uint8), ((t >> 1) * erased).astype(np.uint8)
    mps = MPSMLDecoder(code, chi=chi)
    fp = mps.fail_prob(ex, ez, make_priors(code.n, 0.05, erased))
    assert np.allclose(fp, 0.5)
    with pytest.raises(ValueError):
        mps.fail(ex, ez, 0.05, erased)                  # ties need an rng
    assert 0 < mps.fail(ex, ez, 0.05, erased, rng).mean() < 1


def test_chunking_and_threads_do_not_change_results():
    rng = np.random.default_rng(9)
    code = RotatedSurfaceCode(7)
    ex, ez, er = sample_stratum(code.n, 1, 5, 70, rng)
    pri = make_priors(code.n, 0.08, er)
    ref = MPSMLDecoder(code, chi=6, chunk=70).log_class_weights(ex, ez, pri)
    np.testing.assert_array_equal(ref, MPSMLDecoder(code, chi=6, chunk=9).log_class_weights(ex, ez, pri))
    np.testing.assert_array_equal(ref, MPSMLDecoder(code, chi=6, chunk=9, threads=3).log_class_weights(ex, ez, pri))


def test_no_failure_below_half_distance():
    rng = np.random.default_rng(10)
    d = 9
    code = RotatedSurfaceCode(d)
    mps = MPSMLDecoder(code, chi=4)
    for k, w in [(0, 4), (2, 3), (4, 2)]:
        assert 2 * w + k < d
        ex, ez, er = sample_stratum(code.n, k, w, 60, rng)
        assert (mps.fail_prob(ex, ez, make_priors(code.n, 0.05, er)) == 0).all()


def test_convergence_monitor_fixes_truncation_errors_and_counts_them():
    """Re-decoding the ambiguous samples with a larger chi repairs wrong decisions; last_refine_changed counts them."""
    rng = np.random.default_rng(12)
    d = 9
    code = RotatedSurfaceCode(d)
    exd = ExactTNMLDecoder(code)
    wrong_plain = wrong_refined = changed_seen = refined = 0
    for w in (14, 17, 20):                                   # dense strata with many ambiguous samples
        ex, ez, er = sample_stratum(code.n, 0, w, 100, rng)
        pri = make_priors(code.n, 0.10, er)
        fe = exd.fail_prob(ex, ez, pri)
        plain = MPSMLDecoder(code, chi=2)
        fp0 = plain.fail_prob(ex, ez, pri)
        mon = MPSMLDecoder(code, chi=2)
        fp1 = mon.fail_prob(ex, ez, pri, refine_chi=8, margin=8.0)
        wrong_plain += int((fp0 != fe).sum())
        wrong_refined += int((fp1 != fe).sum())
        changed_seen += mon.last_refine_changed
        refined += mon.last_refined
        assert mon.last_refined >= mon.last_refine_changed
    assert wrong_plain > 0                                   # chi = 2 really makes errors here
    assert wrong_refined < wrong_plain / 3
    assert changed_seen >= wrong_plain - wrong_refined       # the monitor sees (at least) the repaired decisions


def test_monitor_does_nothing_when_chi_is_already_large():
    rng = np.random.default_rng(13)
    code = RotatedSurfaceCode(5)
    ex, ez, er = sample_stratum(code.n, 0, 5, 40, rng)
    pri = make_priors(code.n, 0.1, er)
    mon = MPSMLDecoder(code, chi=16)
    a = mon.fail_prob(ex, ez, pri, refine_chi=32, margin=50.0)
    assert mon.last_refined == 40 and mon.last_refine_changed == 0
    np.testing.assert_array_equal(a, MPSMLDecoder(code, chi=16).fail_prob(ex, ez, pri))
