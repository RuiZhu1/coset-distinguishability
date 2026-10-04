import pytest
import stim

from lcd.analysis.circuit_peierls_geodesic import best_bound
from lcd.analysis.observables import bound_any, project
from lcd.cli import main


TWO_CHAINS = """
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1
error(0.05) D2 L1
error(0.05) D2 D3
error(0.05) D3
"""


def test_two_independent_chains_sum_of_bounds():
    dem = stim.DetectorErrorModel(TWO_CHAINS)
    r = bound_any(dem, best_bound)
    b0 = best_bound(project(dem, 0))["bound"]
    b1 = best_bound(project(dem, 1))["bound"]
    assert r["per_observable"] == pytest.approx([b0, b1])
    assert r["bound"] == pytest.approx(b0 + b1)
    assert b1 < b0


def test_projection_keeps_only_one_observable():
    p = project(stim.DetectorErrorModel(TWO_CHAINS), 1)
    assert p.num_observables == 1
    assert "L0" in str(p) and "L1" not in str(p)


def test_cli_reports_per_observable(tmp_path, capsys):
    f = tmp_path / "two.dem"
    f.write_text(TWO_CHAINS)
    assert main([str(f)]) == 0
    assert "sum over 2 observables" in capsys.readouterr().out
