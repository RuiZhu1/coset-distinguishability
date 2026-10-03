import numpy as np
import pytest
import stim

from lcd.analysis.circuit_peierls import dem_graph, peierls_bound


def surface_dem(d, p):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                               after_reset_flip_probability=p, before_measure_flip_probability=p,
                               before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True)


def test_repetition_code_by_hand():
    # path b -D0- D1 - b with one odd edge: the only odd cycle has three edges, beta = 2 sqrt(q(1-q)) at lam = 1/2
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1")
    r = peierls_bound(dem)
    # pymatching minimizes rounded weights: every factor carries exp(lam delta), delta = ln 9 / (2 (2^24 - 1)) (4.28(e))
    delta = np.log(9) / (2 * (2 ** 24 - 1))
    assert r["delta"] == pytest.approx(delta, rel=1e-12)
    assert r["bound"] == pytest.approx(0.6 ** 3 * np.exp(3 * 0.5 * delta), rel=1e-9)
    assert r["lam"] == pytest.approx(0.5, abs=1e-4)
    # a decoder that minimizes the float weights exactly: no slack
    w = np.full(3, np.log(9))
    assert peierls_bound(dem, weights=w, lams=[0.5])["bound"] == pytest.approx(0.6 ** 3, rel=1e-12)


def test_inconsistent_flags_and_unbalanced_graphs_are_refused():
    with pytest.raises(ValueError):
        dem_graph(stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0\nerror(0.1) D0 D1\nerror(0.1) D1"))
    with pytest.raises(NotImplementedError):  # an odd triangle that avoids the boundary
        dem_graph(stim.DetectorErrorModel("error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.1) D0 D2\nerror(0.1) D0"))


def test_surface_code_redecomposition_and_value():
    G = dem_graph(surface_dem(3, 1e-3))
    assert G.stats["split_multi"] == 0 and G.stats["exclusive_multi"] == 0 and G.stats["redecomposed"] == 48
    r = peierls_bound(surface_dem(3, 1e-3))
    assert r["bound"] == pytest.approx(2.93e-2, rel=0.01) and r["rho_upper"] < 1


def test_bound_is_monotone_in_the_noise_for_fixed_weights():
    from lcd.analysis.circuit_peierls import pymatching_weights
    dem = surface_dem(3, 1e-3)
    G = dem_graph(dem)
    w = pymatching_weights(dem, G)
    lo = peierls_bound(dem, weights=w, lams=[0.5], G=G)["bound"]
    G2 = dem_graph(surface_dem(3, 2e-3))  # same graph, larger mechanism probabilities, same decoder weights
    assert G2.edges == G.edges
    hi = peierls_bound(dem, weights=w, lams=[0.5], G=G2)["bound"]
    assert hi > lo and np.isfinite(hi)
