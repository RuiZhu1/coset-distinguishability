import numpy as np
import pytest

from lcd.codes import RotatedSurfaceCode
from lcd.decoders import ExactTNMLDecoder, MWPMCodeCapacity
from lcd.decoders.tn_ml import make_priors
from lcd.noise import sample_iid, sample_stratum


@pytest.fixture(scope="module")
def d3():
    """Brute-force tables for the d = 3 code: syndrome id and raw logical label of all 4^9 Paulis."""
    code = RotatedSurfaceCode(3)
    n = code.n
    idx = np.arange(4 ** n)
    a = np.stack([(idx >> (2 * (n - 1 - j))) & 3 for j in range(n)], axis=1)     # bit0 -> x, bit1 -> z
    ex, ez = (a & 1).astype(np.int64), (a >> 1).astype(np.int64)
    sx = (ez @ code.HX.T.astype(np.int64)) % 2          # X checks detect Z errors
    sz = (ex @ code.HZ.T.astype(np.int64)) % 2          # Z checks detect X errors
    syn = np.concatenate([sx, sz], axis=1) @ (1 << np.arange(sx.shape[1] + sz.shape[1]))
    lab = ((ez @ code.LX.astype(np.int64)) % 2) + 2 * ((ex @ code.LZ.astype(np.int64)) % 2)
    return code, a, ex, ez, syn, lab


def brute_force_weights(d3, e_x, e_z, priors_row):
    code, a, _, _, syn, lab = d3
    n = code.n
    P = np.ones(a.shape[0])
    for j in range(n):
        P *= priors_row[j][a[:, j]]
    ex = e_x.astype(np.int64)
    ez = e_z.astype(np.int64)
    nsyn = code.HX.shape[0] + code.HZ.shape[0]
    s_e = np.concatenate([(code.HX @ ez) % 2, (code.HZ @ ex) % 2]) @ (1 << np.arange(nsyn))
    out = []
    for c in range(4):
        rx = ex ^ ((c & 1) * code.LX)
        rz = ez ^ (((c >> 1) & 1) * code.LZ)
        lab_c = ((rz @ code.LX) % 2) + 2 * ((rx @ code.LZ) % 2)
        out.append(P[(syn == s_e) & (lab == lab_c)].sum())
    return np.array(out)


def test_coset_sums_match_brute_force_with_erasures(d3):
    code = d3[0]
    dec = ExactTNMLDecoder(code)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        k, w = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        ex, ez, er = sample_stratum(code.n, k, w, 1, rng)
        pri = make_priors(code.n, 0.07, er)
        Z = dec.class_weights(ex, ez, pri)[0]
        bf = brute_force_weights(d3, ex[0], ez[0], pri[0])
        worst = max(worst, np.abs(Z - bf).max() / bf.max())
    assert worst < 1e-12


def test_exact_bayes_error_matches_enumeration_at_d3(d3):
    """eps*(p, e=0) from the decoder (summed over all 4^9 errors) equals 1 - sum_s max_l P(s, l)."""
    code, a, _, _, syn, lab = d3
    dec = ExactTNMLDecoder(code)
    p = 0.12
    pri_row = make_priors(code.n, p, np.zeros((1, code.n), bool))[0]
    ex = (a & 1).astype(np.uint8)
    ez = (a >> 1).astype(np.uint8)
    pri = np.broadcast_to(pri_row, (a.shape[0],) + pri_row.shape)
    fp = dec.fail_prob(ex, ez, pri)
    P = np.ones(a.shape[0])
    for j in range(code.n):
        P *= pri_row[j][a[:, j]]
    P /= P.sum()
    eps_dec = float((P * fp).sum())
    joint = np.bincount(syn * 4 + lab, weights=P, minlength=(syn.max() + 1) * 4).reshape(-1, 4)
    eps_exact = 1.0 - joint.max(axis=1).sum()
    assert eps_dec == pytest.approx(eps_exact, abs=1e-12)


def test_ties_when_erasure_contains_a_logical_support():
    """Erasing a whole column (support of Xbar) leaves classes {I, Xbar} exactly tied: failure probability 1/2."""
    code = RotatedSurfaceCode(5)
    dec = ExactTNMLDecoder(code)
    N = 50
    rng = np.random.default_rng(1)
    erased = np.zeros((N, code.n), bool)
    erased[:, code.LX.astype(bool)] = True
    t = rng.integers(0, 4, (N, code.n))
    ex = ((t & 1) * erased).astype(np.uint8)
    ez = ((t >> 1) * erased).astype(np.uint8)
    pri = make_priors(code.n, 0.05, erased)
    fp = dec.fail_prob(ex, ez, pri)
    assert np.allclose(fp, 0.5)
    with pytest.raises(ValueError):
        dec.fail(ex, ez, 0.05, erased)                      # ties need an rng
    f = dec.fail(ex, ez, 0.05, erased, rng)
    assert f.dtype == bool and 0 < f.mean() < 1


def test_no_failure_below_half_distance_even_with_erasures():
    """2w + k < d  =>  failure probability exactly 0 (the 'known zero strata' rule holds for ML too)."""
    rng = np.random.default_rng(2)
    d = 7
    code = RotatedSurfaceCode(d)
    dec = ExactTNMLDecoder(code)
    for k, w in [(0, 3), (2, 2), (4, 1), (6, 0)]:
        assert 2 * w + k < d
        ex, ez, er = sample_stratum(code.n, k, w, 300, rng)
        fp = dec.fail_prob(ex, ez, make_priors(code.n, 0.05, er))
        assert (fp == 0).all()


def test_ml_is_not_worse_than_mwpm_d5():
    d, p, N = 5, 0.12, 20000
    code = RotatedSurfaceCode(d)
    ml, mw = ExactTNMLDecoder(code), MWPMCodeCapacity(code)
    rng = np.random.default_rng(3)
    ex, ez = sample_iid(code.n, p, N, rng)
    f_ml = ml.fail(ex, ez, p, None, rng).astype(float)
    f_mw = mw.fail(ex, ez).astype(float)
    diff = f_mw - f_ml
    assert diff.mean() > -3 * diff.std() / np.sqrt(N)
    assert diff.mean() > 0            # at this p the Y-correlations make ML strictly better


def test_chunking_does_not_change_results():
    code = RotatedSurfaceCode(5)
    rng = np.random.default_rng(4)
    ex, ez, er = sample_stratum(code.n, 1, 3, 200, rng)
    pri = make_priors(code.n, 0.08, er)
    a = ExactTNMLDecoder(code, max_chunk_elements=1e9).fail_prob(ex, ez, pri)
    b = ExactTNMLDecoder(code, max_chunk_elements=1e3).fail_prob(ex, ez, pri)
    np.testing.assert_array_equal(a, b)
