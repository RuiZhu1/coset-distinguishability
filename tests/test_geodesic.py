import sys
from pathlib import Path

import pytest
import stim

from lcd.analysis.circuit_peierls import peierls_bound
from lcd.analysis.circuit_peierls_geodesic import geodesic_bound

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory" / "checks"))


def surface_dem(d, p):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d, after_clifford_depolarization=p,
                               after_reset_flip_probability=p, before_measure_flip_probability=p,
                               before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True)


def test_tighter_than_theorem_4_28_and_above_simulation():
    dem = surface_dem(3, 1e-3)
    new = geodesic_bound(dem)["bound"]
    assert new < peierls_bound(dem)["bound"] / 2
    assert new > 7.6e-4 * 5          # simulated pymatching rate 7.6e-4 (Table of Proposition 4.29)


def test_relaxation_dominates_exact_restricted_sum():
    from geodesic_checks import exact_restricted
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2, before_round_data_depolarization=0.02,
                               before_measure_flip_probability=0.02)
    relaxed, exact, old, n = exact_restricted(c.detector_error_model(decompose_errors=True), 0.4)
    assert n == 29 and exact <= relaxed and exact <= old


def test_split_mechanisms_are_refused():
    # a mechanism with two non-boundary edges on an odd cycle cannot be re-decomposed: not covered
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D2\n"
                                  "error(0.05) D0 D1 ^ D1 D2")
    with pytest.raises(NotImplementedError):
        geodesic_bound(dem)
