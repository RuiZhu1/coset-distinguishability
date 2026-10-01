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
