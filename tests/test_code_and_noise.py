import numpy as np
import pytest

from lcd.codes import RotatedSurfaceCode
from lcd.decoders import MWPMCodeCapacity
from lcd.noise import sample_iid, sample_stratum


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_rotated_code_is_consistent(d):
    code = RotatedSurfaceCode(d)      # construction asserts commutation, rank and logical anticommutation
    assert code.n == d * d
    assert code.HX.shape[0] + code.HZ.shape[0] == d * d - 1
    assert code.LX.sum() == d and code.LZ.sum() == d


def test_rotated_code_rejects_even_distance():
    with pytest.raises(ValueError):
        RotatedSurfaceCode(4)


def test_sample_stratum_has_exact_counts_and_uniform_paulis():
    rng = np.random.default_rng(1)
    n, k, w, N = 25, 3, 4, 60000
    ex, ez, erased = sample_stratum(n, k, w, N, rng)
    assert (erased.sum(axis=1) == k).all()
    err = ((ex | ez).astype(bool)) & ~erased
    # erased qubits carry an arbitrary Pauli, so the error count on non-erased qubits is exactly w
    assert (err.sum(axis=1) == w).all()
    x = (ex.astype(bool) & ~ez.astype(bool) & err).sum()
    y = (ex.astype(bool) & ez.astype(bool) & err).sum()
    z = (~ex.astype(bool) & ez.astype(bool) & err).sum()
    tot = x + y + z
    for c in (x, y, z):
        assert abs(c / tot - 1 / 3) < 0.01
    # erased qubits: uniform over I, X, Y, Z
    er_pauli = (ex[erased] + 2 * ez[erased])
    freq = np.bincount(er_pauli, minlength=4) / er_pauli.size
    assert np.abs(freq - 0.25).max() < 0.01
    # placements are uniform: every qubit is erased about k/n of the time
    assert np.abs(erased.mean(axis=0) - k / n).max() < 0.01


def test_sample_stratum_zero_and_overflow():
    rng = np.random.default_rng(0)
    ex, ez, er = sample_stratum(9, 0, 0, 5, rng)
    assert not ex.any() and not ez.any() and not er.any()
    with pytest.raises(ValueError):
        sample_stratum(9, 5, 5, 1, rng)


def test_sample_iid_rate():
    rng = np.random.default_rng(2)
    ex, ez = sample_iid(50, 0.1, 40000, rng)
    assert abs((ex | ez).mean() - 0.1) < 0.003


@pytest.mark.parametrize("d", [3, 5, 7])
def test_mwpm_corrects_every_error_below_half_distance(d):
    """Stratum (k=0, w) with 2w < d must never fail -- the basis of the 'known zero strata' rule."""
    rng = np.random.default_rng(3)
    code = RotatedSurfaceCode(d)
    dec = MWPMCodeCapacity(code)
    for w in range(1, (d + 1) // 2):
        ex, ez, _ = sample_stratum(code.n, 0, w, 4000, rng)
        assert not dec.fail(ex, ez).any()


def test_mwpm_fails_on_a_logical_operator():
    code = RotatedSurfaceCode(5)
    dec = MWPMCodeCapacity(code)
    ex = np.zeros((1, code.n), np.uint8)
    ex[0] = code.LX        # an X-type logical: zero syndrome, flips the logical observable
    ez = np.zeros_like(ex)
    assert dec.fail(ex, ez).all()


def test_mwpm_without_erasure_flags_is_unchanged():
    rng = np.random.default_rng(5)
    code = RotatedSurfaceCode(5)
    dec = MWPMCodeCapacity(code)
    ex, ez = sample_iid(code.n, 0.12, 3000, rng)
    assert (dec.fail(ex, ez) == dec.fail(ex, ez, np.zeros(ex.shape, bool))).all()


@pytest.mark.parametrize("d", [5, 7])
def test_mwpm_with_erasure_corrects_every_stratum_below_half_distance(d):
    """2w + k < d  =>  the zero-weight erasure edges never cause a failure (known-zero strata rule for MWPM)."""
    rng = np.random.default_rng(4)
    code = RotatedSurfaceCode(d)
    dec = MWPMCodeCapacity(code)
    for k in range(1, d):
        for w in range((d - k + 1) // 2):
            assert 2 * w + k < d
            ex, ez, er = sample_stratum(code.n, k, w, 500, rng)
            assert not dec.fail(ex, ez, er).any()


def test_mwpm_is_bayes_optimal_on_pure_erasure():
    """With only erasures, averaging over all 4^k Paulis on a fixed erased set, MWPM fails exactly as often as ML
    (both pick one correction among the equally likely logical classes)."""
    from lcd.decoders import ExactTNMLDecoder
    from lcd.decoders.tn_ml import make_priors

    rng = np.random.default_rng(6)
    code = RotatedSurfaceCode(3)
    mw, ml = MWPMCodeCapacity(code), ExactTNMLDecoder(code)
    sets = [np.flatnonzero(code.LX)] + [rng.choice(code.n, k, replace=False) for k in (2, 3, 4, 5, 5)]
    for A in sets:
        k = len(A)
        t = (np.arange(4 ** k)[:, None] >> (2 * np.arange(k))) & 3          # all Paulis on A: 0=I 1=X 2=Z 3=Y
        ex = np.zeros((4 ** k, code.n), np.uint8)
        ez = np.zeros_like(ex)
        ex[:, A], ez[:, A] = t & 1, t >> 1
        er = np.zeros(ex.shape, bool)
        er[:, A] = True
        f_mw = mw.fail(ex, ez, er).mean()
        f_ml = ml.fail_prob(ex, ez, make_priors(code.n, 0.0, er)).mean()
        assert abs(f_mw - f_ml) < 1e-12      # nontrivial: 0.5 on the logical support, 0.75 / 0.5 on two k = 5 sets
