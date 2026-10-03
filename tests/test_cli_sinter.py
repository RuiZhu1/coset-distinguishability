import json

import pytest
import stim

from lcd.cli import main


def test_cli_generated_and_dem_file(tmp_path, capsys):
    assert main(["--generated", "surface_code:rotated_memory_z", "-d", "3", "-p", "1e-3", "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["method"] == "4.34" and r["bound"] == pytest.approx(6.3e-3, rel=0.05)  # Theorem 4.34 (gap refinement)
    assert main(["--generated", "surface_code:rotated_memory_z", "-d", "3", "-p", "1e-3", "--json", "--method", "4.28"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["bound"] == pytest.approx(2.9e-2, rel=0.05) and 0 < r["delta"] < 1e-6  # Table of Proposition 4.29
    f = tmp_path / "rep.dem"
    f.write_text("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n")
    assert main([str(f), "--lam", "0.5"]) == 0
    assert "p_L(MWPM) <= 2.160e-01" in capsys.readouterr().out


def test_cli_refuses_unbalanced_model(tmp_path, capsys):
    f = tmp_path / "odd.dem"
    f.write_text("error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.1) D0 D2\nerror(0.1) D0\n")
    assert main([str(f)]) == 1
    assert "not covered" in capsys.readouterr().err


def test_sinter_compare_bound_above_sampled_rate():
    sinter = pytest.importorskip("sinter")
    from lcd.integrations.sinter_peierls import compare

    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=2e-3,
                               after_reset_flip_probability=2e-3, before_measure_flip_probability=2e-3,
                               before_round_data_depolarization=2e-3)
    tasks = [sinter.Task(circuit=c, json_metadata={"d": 3})]
    stats = sinter.collect(num_workers=1, tasks=tasks, decoders=["pymatching"], max_shots=20_000, max_errors=500)
    (row,) = compare(tasks, stats)
    assert row["consistent"] and row["bound"] > row["rate_high"]
    assert row["json_metadata"] == {"d": 3} and row["ratio"] > 1
