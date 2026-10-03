from pathlib import Path

import numpy as np
import pytest
import stim

from lcd.analysis.circuit_peierls import peierls_bound
from lcd.analysis.hardware import bound_at_scale, certifiability_margin, scale_dem

DATA = Path(__file__).resolve().parents[1] / "data"


def surface_dem(d, p, rounds=None):
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=rounds or d,
                               after_clifford_depolarization=p, after_reset_flip_probability=p,
                               before_measure_flip_probability=p, before_round_data_depolarization=p)
    return c.detector_error_model(decompose_errors=True)


def test_scale_dem_scales_clips_and_flattens():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        repeat 2 {
            error(0.2) D0 D1
            shift_detectors 1
        }
        detector(0, 0) D0
    """)
    out = scale_dem(dem, 3.0)
    ps = [i.args_copy()[0] for i in out if i.type == "error"]
    assert ps == pytest.approx([0.3, 0.5, 0.5])          # 0.6 clipped at 1/2; repeat block flattened
    assert out.num_detectors == dem.num_detectors and out.num_observables == 1
    assert str(scale_dem(dem, 1.0)) == str(dem.flattened())
    with pytest.raises(ValueError):
        scale_dem(dem, -1.0)


def test_margin_brackets_known_finite_and_infinite_points():
    # uniform circuit noise, d = 5: finite at p = 1e-3, infinite at p = 5e-3 (lam = 1/2)
    lo, hi = surface_dem(5, 1e-3, rounds=3), surface_dem(5, 5e-3, rounds=3)
    assert np.isfinite(bound_at_scale(lo, 1.0)) and not np.isfinite(bound_at_scale(hi, 1.0))
    m = certifiability_margin(lo)
    assert not m["capped"] and not m["below"] and m["s_upper"] / m["s_star"] <= 1 + 1e-3
    # the threshold lies in (1e-3, 5e-3): 1 < s* < 5 for p = 1e-3 and s* < 1 for p = 5e-3
    assert 1 < m["s_star"] < 5
    assert np.isfinite(bound_at_scale(lo, m["s_star"])) and not np.isfinite(bound_at_scale(lo, m["s_upper"]))
    m5 = certifiability_margin(hi)
    assert m5["s_star"] < 1
    # scaling a model by s is (up to stim's float rounding of the mechanisms) the same as scaling the margin by 1/s
    assert m5["s_star"] * 5 == pytest.approx(m["s_star"], rel=0.05)


def test_margin_capped():
    tiny = surface_dem(3, 1e-5, rounds=2)
    assert certifiability_margin(tiny)["capped"]
    # path b - D0 - D1 - b: one odd cycle, no walk can repeat, so the bound is finite for every s
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1")
    m = certifiability_margin(dem)
    assert m["capped"]
    expected = (2 * np.sqrt(0.4 * 0.6)) ** 3                 # beta_e(1/2) = 2 sqrt(q (1 - q)), q = 4 * 0.1, up to rounding
    assert peierls_bound(scale_dem(dem, 4.0), lams=[0.5])["bound"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.skipif(not (DATA / "surface_code_bZ_d3_r05_center_3_5").is_dir(), reason="hardware data not present")
def test_sycamore_d3_margin_below_one():
    dem = stim.DetectorErrorModel.from_file(DATA / "surface_code_bZ_d3_r05_center_3_5" / "circuit_detector_error_model.dem")
    m = certifiability_margin(dem)
    assert 0.6 < m["s_star"] < 0.8
