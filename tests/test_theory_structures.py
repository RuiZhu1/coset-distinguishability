"""Tests of the exact structure enumerator and of the closed forms used by the counterexample search (theory/checks)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory" / "checks"))
from structures import Structure, all_chains, rref  # noqa: E402
import universal_order_search as uos  # noqa: E402


def lam(p): return 1 - 4 * p / 3
def mu(p, e): return (1 - e) * lam(p)
def nu(p, e): return (1 - e * e) * lam(p)


def five_qubit():
    def pauli(s):
        v = np.zeros(10, np.uint8)
        for j, ch in enumerate(s):
            v[j] = ch in "XY"
            v[5 + j] = ch in "ZY"
        return v
    gens = np.array([pauli(s) for s in ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")])
    V = np.array([[(i >> k) & 1 for k in range(10)] for i in range(2 ** 10)], np.uint8)
    Om = np.zeros((10, 10), np.uint8)
    Om[:5, 5:] = np.eye(5, dtype=np.uint8)
    Om[5:, :5] = np.eye(5, dtype=np.uint8)
    N = V[((V @ (gens @ Om).T) % 2 == 0).all(1)]
    return Structure(5, gens, rref(N)[0])


@pytest.mark.parametrize("p,e", [(0.0213, 0.0), (0.0213, 0.07), (0.3, 0.5), (0.7, 0.2)])
def test_closed_forms(p, e):
    assert uos.uncoded().eps(p, e) == pytest.approx(0.75 * (1 - mu(p, e)), abs=1e-13)
    assert uos.f2().eps(p, e) == pytest.approx(0.75 * (1 - nu(p, e)), abs=1e-13)


def test_f2_and_rep3_tables():
    p = 0.0213
    f2 = uos.f2()
    assert [f2.eps_A(p, A) for A in [(), (0,), (1,), (0, 1)]] == pytest.approx([p, p, p, 0.75], abs=1e-13)
    r3 = uos.rep(3)
    assert r3.eps_A(p, ()) < 0.2 * p                              # 3 copies of the Pauli: second order in p
    for A in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
        assert r3.eps_A(p, A) == pytest.approx(p, abs=1e-13)      # 2 copies tie, 1 copy: failure p
    assert r3.eps_A(p, (0, 1, 2)) == pytest.approx(0.75, abs=1e-13)


def test_perfect_code_attains_the_rate_bound():
    st = five_qubit()
    for p0 in (0.02, 0.1):
        R, dp, de = st.rate(p0, 0.0)
        assert R == pytest.approx(0.75 - p0, abs=1e-7)


def test_structure_counts_and_two_qubit_minimum():
    assert len(list(all_chains(1))) == 7
    chains = list(all_chains(2))
    assert len(chains) == 446
    p0, e0 = 0.0213, 0.05
    c = (0.75 - p0) / (1 - e0)
    best = min(r / c for S, N in chains for r, dp, de in [Structure(2, S, N).rate(p0, e0)] if np.isfinite(r) and dp > 1e-9 and de > -1e-9)
    assert best == pytest.approx(2 * e0 / (1 + e0), abs=1e-6)


def test_free_order_is_cut_out_by_lambda_and_mu():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        p, pp = rng.uniform(0, 0.749, 2)
        e, ep = rng.uniform(0, 0.99, 2)
        assert uos.reachable(p, e, pp, ep) == (lam(pp) <= lam(p) and mu(pp, ep) <= mu(p, e))
