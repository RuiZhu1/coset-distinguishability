import numpy as np

from lcd.analysis.lattice_gap import Lattice, _blocked, check_A2, envelopes


def test_blocked_sum_dominates_truncated_sum():
    A = np.zeros((3, 2, 2)); A[0] = 0.1; A[1] = 0.05; A[2] = 0.02
    tot = _blocked(A)
    assert np.all(tot >= A.sum(0))


def test_A2_and_theta_star_small():
    env = envelopes(5, 1e-3)
    wp = {tuple(int(x) for x in k.strip("()").split(", ")): v[2] for k, v in env["raw"].items()}
    assert check_A2(5, 1e-3, wp)["max_excess"] <= 1e-9
    b = Lattice(1e-3, L=8).theta_star(lams=(0.5,), mus=(0.2,), iters=6)
    assert b["theta"] > 0.4
