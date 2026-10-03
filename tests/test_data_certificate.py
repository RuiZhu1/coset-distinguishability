import itertools

import numpy as np
import stim

from lcd.analysis.circuit_peierls import dem_graph
from lcd.analysis.circuit_peierls_geodesic import geodesic_bound
from lcd.analysis.data_certificate import (betting_ci, certify, edge_intervals, point_estimates, q_from_moments)


def surface(d, rounds, p):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds,
                                  after_clifford_depolarization=p, after_reset_flip_probability=p,
                                  before_measure_flip_probability=p, before_round_data_depolarization=p)


def test_pij_identity_exact():
    # a small independent-edge model with a triangle, a boundary edge per detector: exact moments by enumeration
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (0,), (1,), (2,), (3,)]
    q = np.array([0.03, 0.08, 0.05, 0.11, 0.02, 0.07, 0.04, 0.09])
    a, c = np.zeros(4), np.zeros((4, 4))
    for f in itertools.product((0, 1), repeat=len(edges)):
        pr = np.prod(np.where(f, q, 1 - q))
        x = np.zeros(4, int)
        for on, e in zip(f, edges):
            if on:
                x[list(e)] ^= 1
        a += pr * x
        c += pr * np.outer(x, x)
    for k, e in enumerate(edges[:4]):
        i, j = e
        assert abs(q_from_moments(a[i], a[j], c[i, j]) - q[k]) < 1e-12
    # boundary edge of detector 3: 1 - 2 a_3 = (1 - 2 q_b)(1 - 2 q_23)
    assert abs(0.5 * (1 - (1 - 2 * a[3]) / (1 - 2 * q[3])) - q[7]) < 1e-12


def test_point_estimates_and_interval_coverage_on_simulation():
    c = surface(3, 3, 2e-3)
    dem = c.detector_error_model(decompose_errors=True)
    G = dem_graph(dem)
    X = c.compile_detector_sampler(seed=5).sample(50_000)
    qh = point_estimates(G, X)
    assert np.max(np.abs(qh - G.q)) < 0.004
    for meth in ("cp", "paired"):
        iv = edge_intervals(G, X, 1e-3, meth)
        assert np.all(iv.q_lo <= G.q) and np.all(G.q <= iv.q_hi), meth
        assert np.all(iv.q_hi <= 0.5) and np.all(iv.q_lo >= 0)
    iv = edge_intervals(G, X, 1e-3, "paired", pool=True, coords=c.get_detector_coordinates())
    assert np.all(iv.q_lo <= G.q) and np.all(G.q <= iv.q_hi)


def test_box_bound_reduces_to_the_point_bound():
    dem = surface(3, 3, 1e-3).detector_error_model(decompose_errors=True)
    G = dem_graph(dem)
    ref = geodesic_bound(dem, G=G)["bound"]
    assert geodesic_bound(dem, G=G, q_upper=G.q.copy(), q_lower=G.q.copy())["bound"] == ref
    wider = geodesic_bound(dem, G=G, q_upper=np.minimum(0.5, 1.2 * G.q), q_lower=0.8 * G.q)["bound"]
    assert wider > ref


def test_certificate_is_above_the_simulated_rate():
    c = surface(3, 3, 2e-3)
    dem = c.detector_error_model(decompose_errors=True)
    X = c.compile_detector_sampler(seed=7).sample(50_000)
    r = certify(dem, X, 1e-3, "paired")
    assert r["bound_4_32"] >= r["bound_4_32_point"] > 3.0e-3     # simulated pymatching rate 3.0e-3 (2e6 shots)
    assert r["bound_4_28"] >= r["bound_4_28_point"]


def test_betting_interval_covers_and_shrinks():
    rng = np.random.default_rng(0)
    X = rng.choice([0.0, 0.5, 1.0], size=(20_000, 3), p=[0.02, 0.95, 0.03])
    lo, hi = betting_ci(X, 1e-3)
    mu = 0.5 * 0.95 + 0.03
    assert np.all(lo <= mu) and np.all(mu <= hi) and np.all(hi - lo < 0.01)
