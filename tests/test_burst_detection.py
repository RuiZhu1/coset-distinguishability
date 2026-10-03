import itertools

import numpy as np
import pytest
import stim
from scipy.special import xlogy

from lcd.analysis.burst_detection import (Baseline, CountDetector, ScanDetector, bernoulli_divergences, chain_layout,
                                          counts_and_trials, marked_poisson_divergences, plant_bursts, streak_mask,
                                          thresholds, window_sums)


def brute_scan(K, M, p, heights, widths, model):
    best = 0.0
    T, Q = p.shape
    for h, w in itertools.product(heights, widths):
        for t0, q0 in itertools.product(range(T - h + 1), range(Q - w + 1)):
            sl = (slice(t0, t0 + h), slice(q0, q0 + w))
            k, n, mu = K[sl].sum(), M[sl].sum(), (M[sl] * p[sl]).sum()
            if k <= mu:
                continue
            if model == "poisson":
                v = k * np.log(k / mu) - (k - mu)
            else:
                rho = k / n
                v = (xlogy(k, rho) + xlogy(n - k, 1 - rho)
                     - np.sum(K[sl] * np.log(p[sl]) + (M[sl] - K[sl]) * np.log(1 - p[sl])))
            best = max(best, v)
    return best


@pytest.mark.parametrize("model", ["bernoulli", "poisson"])
def test_scan_equals_brute_force(model):
    rng = np.random.default_rng(0)
    T, Q = 5, 6
    p = rng.uniform(0.05, 0.3, (T, Q))
    bl = Baseline(p=p, count_mean=p.sum(), count_var=1.0, shots=100)
    hs, ws = (1, 2, 3, 5), (1, 2, 4, 6)
    sc = ScanDetector(bl, widths=ws, heights=hs, model=model)
    m = 3
    K = rng.binomial(m, np.clip(p * 1.8, 0, 1), size=(7, T, Q))
    got = sc.score(K, m=m, batch=3)
    want = [brute_scan(K[i], np.full((T, Q), m), p, hs, ws, model) for i in range(len(K))]
    assert np.allclose(got, want, rtol=1e-10, atol=1e-10)
    # per-cell trials (masked cells)
    M = rng.integers(0, m + 1, size=K.shape)
    K2 = np.minimum(K, M)
    got = sc.score(K2, M)
    want = [brute_scan(K2[i], M[i], p, hs, ws, model) for i in range(len(K))]
    assert np.allclose(got, want, rtol=1e-10, atol=1e-10)
    # argmax points at a rectangle with the maximal value
    v, a = sc.score(K, m=m, return_argmax=True)
    for i in range(len(K)):
        t0, q0, h, w = a[i]
        if v[i] > 0:
            L = sc.llr_boxes(K[i:i + 1], h, w, m=m)[0, t0, q0]
            assert L == pytest.approx(v[i], rel=1e-12)


def test_count_statistic_matches_direct_computation():
    rng = np.random.default_rng(1)
    g = (rng.random((500, 4, 5)) < 0.1).astype(np.uint8)
    bl = Baseline.fit(g)
    tot = g.reshape(500, -1).sum(1)
    assert bl.count_mean == pytest.approx(tot.mean())
    assert bl.count_var == pytest.approx(tot.var(ddof=1))
    cd = CountDetector(bl)
    z = cd.score(g[:10])
    assert np.allclose(z, (tot[:10] - tot.mean()) / tot.std(ddof=1))
    K = window_sums(g, 4)
    zw = cd.score(K, m=4)
    direct = np.array([tot[i:i + 4].sum() for i in range(len(g) - 3)])
    assert np.allclose(zw, (direct - 4 * tot.mean()) / np.sqrt(4 * tot.var(ddof=1)))


def test_planted_burst_is_detected_and_beats_counting():
    rng = np.random.default_rng(2)
    T, Q = 20, 24
    null = (rng.random((4000, T, Q)) < 0.1).astype(np.uint8)
    bl = Baseline.fit(null[:2000])
    sc, cd = ScanDetector(bl, widths=(2, 4, 8), heights=(2, 4, 8)), CountDetector(bl)
    th_s = thresholds(sc.score(null[2000:]), [1e-2])[1e-2]
    th_c = thresholds(cd.score(null[2000:]), [1e-2])[1e-2]
    host = (rng.random((300, T, Q)) < 0.1).astype(np.uint8)
    planted = np.concatenate([plant_bursts(host[i:i + 1], rng, R=4, T=4, theta=0.9)[0] for i in range(len(host))])
    ps, pc = (sc.score(planted) > th_s).mean(), (cd.score(planted) > th_c).mean()
    assert ps > 0.9 and ps > pc + 0.3
    _, corner = plant_bursts(host[:2], rng, R=3, T=5, theta=0.5, corner=(2, 7))
    assert corner == (2, 7)


def test_thresholds_and_streak_mask():
    s = np.arange(1000.0)
    th = thresholds(s, [1e-2, 1e-3])
    assert (s > th[1e-2]).mean() <= 1e-2 and (s > th[1e-3]).mean() <= 1e-3
    g = np.zeros((1, 10, 3), np.uint8)
    g[0, 2:8, 1] = 1
    g[0, 0:2, 0] = 1
    m = streak_mask(g, 5)
    assert m[0, 2:8, 1].all() and m.sum() == 6
    K, M = counts_and_trials(g, min_run=5)
    assert K.sum() == 2 and M.sum() == 30 - 6


def test_divergences_identity_and_chain_layout():
    rng = np.random.default_rng(3)
    l0 = rng.uniform(0.05, 0.2, 30)
    l1 = l0.copy()
    l1[:5] += 0.1
    r = marked_poisson_divergences(l0, l1)
    assert abs(r["identity_residual"]) < 1e-12 and r["D_pattern"] > r["D_count"] > 0
    b = bernoulli_divergences(l0, l1)
    assert b["D_pattern"] > b["D_count"] > 0
    # 1-D chain from a repetition-code circuit
    c = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=4, before_round_data_depolarization=0.01)
    L = chain_layout(c.get_detector_coordinates(), c.detector_error_model())
    assert L.shape == (5, 4) and sorted(L.ravel()) == list(range(c.num_detectors))
