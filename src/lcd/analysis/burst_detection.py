"""Burst detection from detection events: counting versus a space-time scan statistic (theory §5.4-5.5).

Theorem 5.9 says that, under the marked-Poisson model (M), a test that uses the space-time pattern of the detection
events has Stein exponent D_pattern = D_count + Lambda_1 D(pi_1 || pi_0) >= D_count, where D_count belongs to the test that
uses only the total number of events. This module provides the two tests for detector data of a 1-D (repetition) code
laid out on a (round, position) grid, with a common interface:

* ``CountDetector``: z-score of the total number of detection events against the baseline mean and the *empirical*
  variance of the per-shot total (any monotone function of the total, e.g. its Poisson or binomial likelihood ratio
  against an elevated rate, gives the same one-sided test and the same ROC once thresholds are calibrated).
* ``ScanDetector``: maximum over rectangles (contiguous positions x contiguous rounds, sizes from a bounded list) of the
  Bernoulli log-likelihood ratio "every cell of the rectangle has a common rate rho > baseline" versus the per-cell
  baseline rates p_i (rho at its maximum-likelihood value k / n). With m shots summed into the counts K, the cell counts
  are Binomial(m, p_i) and

      LLR(A) = k ln(rho) + (n - k) ln(1 - rho) - sum_{i in A} [K_i ln p_i + (m - K_i) ln(1 - p_i)],   n = m |A|, rho = k / n,

  set to 0 unless k exceeds its baseline expectation. All box sums come from 2-D cumulative sums (integral images),
  so one rectangle size costs O(cells) per shot. ``model="poisson"`` gives the Kulldorff Poisson scan
  k ln(k / mu) - (k - mu) with mu = m sum_A p_i instead.

Both detectors take an array K of shape (batch, rounds, positions) holding detection events summed over m consecutive
shots (m = 1: single shots), and optionally the number M of Bernoulli trials per cell; ``counts_and_trials`` builds
(K, M), optionally after masking leakage-like streaks (``streak_mask``: runs of >= min_run consecutive events of one
detector, which dominate the tail of the scan statistic on real data). Thresholds are calibrated empirically on clean shots
(``thresholds``): neither statistic's null distribution is taken from the model, because real detection events are
correlated (one fault flips two detectors) and the per-shot totals are over-dispersed.

``plant_bursts`` XORs synthetic burst events into real shots; ``marked_poisson_divergences`` and
``bernoulli_divergences`` give D_count and D_pattern (Lemma 5.8 / Theorem 5.9, and the independent-Bernoulli analogue)
for a planted burst at a known location.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import xlogy

DEFAULT_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
DEFAULT_HEIGHTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 51)


# ---------------------------------------------------------------------------------------------------------- data access
def chain_layout(coords: dict, dem) -> np.ndarray:
    """Detector indices of a 1-D code on a (round, position) grid, positions ordered along the chain.

    ``coords``: detector -> coordinates (x, y, t) as returned by ``stim.Circuit.get_detector_coordinates``; ``dem``: a
    ``stim.DetectorErrorModel`` whose two-detector mechanisms inside one time slice give the chain's adjacency."""
    times = sorted({float(c[-1]) for c in coords.values()})
    tindex = {t: i for i, t in enumerate(times)}
    sites = sorted({tuple(float(v) for v in c[:-1]) for c in coords.values()})
    adj: dict = collections.defaultdict(set)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        ds = [t.val for t in inst.targets_copy() if t.is_relative_detector_id()]
        if len(ds) == 2 and coords[ds[0]][-1] == coords[ds[1]][-1]:
            a, b = (tuple(float(v) for v in coords[d][:-1]) for d in ds)
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    ends = [s for s in sites if len(adj[s]) == 1]
    if len(ends) != 2 or any(len(adj[s]) > 2 for s in sites):
        raise ValueError("the space-like adjacency is not a simple chain")
    order, prev = [min(ends)], None
    while len(order) < len(sites):
        nxt = [s for s in adj[order[-1]] if s != prev]
        prev = order[-1]
        order.append(nxt[0])
    pos = {s: i for i, s in enumerate(order)}
    layout = np.full((len(times), len(order)), -1, dtype=np.int64)
    for d, c in coords.items():
        layout[tindex[float(c[-1])], pos[tuple(float(v) for v in c[:-1])]] = d
    if (layout < 0).any():
        raise ValueError("some (round, position) cells have no detector")
    return layout


class B8Reader:
    """Random access to a stim b8 file of detection events (one byte-aligned, little-endian record per shot)."""

    def __init__(self, path: str | Path, num_detectors: int):
        self.num_detectors = num_detectors
        self.nbytes = (num_detectors + 7) // 8
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")
        if self.raw.size % self.nbytes:
            raise ValueError("file size is not a multiple of the record size")
        self.shots = self.raw.size // self.nbytes
        self.raw = self.raw.reshape(self.shots, self.nbytes)

    def events(self, idx) -> np.ndarray:
        """Detection events of the shots ``idx`` (slice or index array) as a uint8 array (shots, num_detectors)."""
        return np.unpackbits(np.asarray(self.raw[idx]), axis=1, bitorder="little")[:, :self.num_detectors]

    def grid(self, idx, layout: np.ndarray) -> np.ndarray:
        """Detection events of the shots ``idx`` on the (round, position) grid: uint8 array (shots, rounds, positions)."""
        return self.events(idx)[:, layout]


def window_sums(grids: np.ndarray, w: int, starts: Sequence[int] | None = None) -> np.ndarray:
    """Sums of ``w`` consecutive shots: out[j] = grids[starts[j]:starts[j] + w].sum(0) (all starts if None)."""
    g = np.asarray(grids, dtype=np.int32)
    if w == 1 and starts is None:
        return g
    c = np.concatenate([np.zeros((1,) + g.shape[1:], np.int32), np.cumsum(g, axis=0, dtype=np.int32)])
    s = np.arange(len(g) - w + 1) if starts is None else np.asarray(starts)
    return c[s + w] - c[s]


# ------------------------------------------------------------------------------------------------------------- baseline
def streak_mask(grids: np.ndarray, min_run: int) -> np.ndarray:
    """Cells (shots, rounds, positions) that belong to a run of at least ``min_run`` consecutive detection events of one
    detector in time. Such runs are the signature of leakage (a detector firing in every round), far more frequent in
    real data than independent rates allow; masking them makes a test insensitive to them."""
    x = np.asarray(grids).astype(np.int16)
    T = x.shape[1]
    f = np.zeros_like(x)
    b = np.zeros_like(x)
    f[:, 0] = x[:, 0]
    for t in range(1, T):
        f[:, t] = (f[:, t - 1] + 1) * x[:, t]
    b[:, T - 1] = x[:, T - 1]
    for t in range(T - 2, -1, -1):
        b[:, t] = (b[:, t + 1] + 1) * x[:, t]
    return (f + b - 1 >= min_run).astype(np.uint8)


def counts_and_trials(grids: np.ndarray, min_run: int | None = None, w: int = 1,
                      starts: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """(K, M): detection events and Bernoulli trials per cell summed over windows of ``w`` shots, after masking streak
    cells (``streak_mask``) if ``min_run`` is given; M is None (= w for every cell) without masking."""
    g = np.asarray(grids, dtype=np.uint8)
    if min_run is None:
        return window_sums(g, w, starts), None
    keep = 1 - streak_mask(g, min_run)
    return window_sums(g * keep, w, starts), window_sums(keep, w, starts)


@dataclass
class Baseline:
    """Per-cell detection rates p (rounds, positions) and the mean / variance of the per-shot total, from clean shots.
    With ``min_run`` the rates and totals are those of the streak-masked data and ``count_var`` is the variance of the
    total minus its expectation given the mask."""
    p: np.ndarray
    count_mean: float
    count_var: float
    shots: int
    min_run: int | None = None

    @classmethod
    def fit(cls, grids: np.ndarray, min_run: int | None = None) -> "Baseline":
        """Estimate from grids (shots, rounds, positions); rates clipped to [1/(2n), 1 - 1/(2n)]."""
        g = np.asarray(grids, dtype=np.uint8)
        n = len(g)
        if n < 2:
            raise ValueError("need at least two shots")
        K, M = counts_and_trials(g, min_run)
        M = np.ones_like(K) if M is None else M
        p = np.clip(K.sum(0) / np.maximum(M.sum(0), 1), 0.5 / n, 1 - 0.5 / n)
        res = K.reshape(n, -1).sum(1) - (M * p).reshape(n, -1).sum(1)
        return cls(p=p, count_mean=float(p.sum()), count_var=float(res.var(ddof=1)), shots=n, min_run=min_run)


# ------------------------------------------------------------------------------------------------------------ detectors
class CountDetector:
    """z-score of the total number of detection events of m summed shots: (N - E) / sqrt(m var), E = sum_i M_i p_i
    (= m mu without masking)."""
    name = "count"

    def __init__(self, baseline: Baseline):
        self.p, self.var = baseline.p, baseline.count_var

    def score(self, K: np.ndarray, M: np.ndarray | None = None, m: int = 1) -> np.ndarray:
        K = np.asarray(K)
        N = K.reshape(len(K), -1).sum(1, dtype=np.int64).astype(np.float64)
        E = m * self.p.sum() if M is None else (np.asarray(M) * self.p).reshape(len(K), -1).sum(1)
        return (N - E) / np.sqrt(m * self.var)


def _integral(x: np.ndarray) -> np.ndarray:
    """Integral image over the last two axes, with a leading row and column of zeros."""
    c = np.cumsum(np.cumsum(x, axis=-2), axis=-1)
    pad = [(0, 0)] * (x.ndim - 2) + [(1, 0), (1, 0)]
    return np.pad(c, pad)


def _box(c: np.ndarray, h: int, w: int) -> np.ndarray:
    """All h x w box sums from an integral image: shape (..., T - h + 1, Q - w + 1)."""
    return c[..., h:, w:] - c[..., :-h, w:] - c[..., h:, :-w] + c[..., :-h, :-w]


class ScanDetector:
    """Space-time scan statistic: max over rectangles (heights x widths) of the elevated-rate log-likelihood ratio.

    Each cell i carries K_i events in M_i Bernoulli trials (M_i = m without masking). For a rectangle A with
    k = sum K_i, n = sum M_i, rho = k / n:
        bernoulli:  LLR = k ln rho + (n - k) ln(1 - rho) - sum_A [K_i ln p_i + (M_i - K_i) ln(1 - p_i)]  if k > sum_A M_i p_i
        poisson:    LLR = k ln(k / mu) - (k - mu),  mu = sum_A M_i p_i                                     if k > mu
    and 0 otherwise."""
    name = "pattern"

    def __init__(self, baseline: Baseline, widths: Sequence[int] = DEFAULT_WIDTHS,
                 heights: Sequence[int] = DEFAULT_HEIGHTS, model: str = "bernoulli"):
        if model not in ("bernoulli", "poisson"):
            raise ValueError("model must be 'bernoulli' or 'poisson'")
        p = np.asarray(baseline.p, dtype=np.float64)
        T, Q = p.shape
        self.p, self.model = p, model
        self.widths = sorted({int(w) for w in widths if 1 <= w <= Q})
        self.heights = sorted({int(h) for h in heights if 1 <= h <= T})
        self.logit = np.log(p) - np.log1p(-p)
        self.log1mp = np.log1p(-p)

    def regions(self) -> int:
        T, Q = self.p.shape
        return sum((T - h + 1) * (Q - w + 1) for h in self.heights for w in self.widths)

    def _integrals(self, K: np.ndarray, M: np.ndarray | None, m: int) -> dict:
        K = np.asarray(K, dtype=np.float64)
        Mf = np.full(K.shape[1:], float(m)) if M is None else np.asarray(M, dtype=np.float64)
        return dict(k=_integral(K), ka=_integral(K * self.logit), n=_integral(Mf), nb=_integral(Mf * self.log1mp),
                    mu=_integral(Mf * self.p))

    def llr_boxes(self, K: np.ndarray, h: int, w: int, M: np.ndarray | None = None, m: int = 1,
                  C: dict | None = None) -> np.ndarray:
        """LLR of every h x w rectangle: shape (batch, T - h + 1, Q - w + 1)."""
        C = self._integrals(K, M, m) if C is None else C
        k, mu = _box(C["k"], h, w), _box(C["mu"], h, w)
        if self.model == "poisson":
            with np.errstate(divide="ignore", invalid="ignore"):
                llr = xlogy(k, k / mu) - (k - mu)
            return np.where(k > mu, llr, 0.0)
        n = _box(C["n"], h, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(n > 0, k / np.maximum(n, 1e-300), 0.0)
            alt = xlogy(k, rho) + xlogy(n - k, 1.0 - rho)
        null = _box(C["ka"], h, w) + _box(C["nb"], h, w)
        return np.where(k > mu, alt - null, 0.0)

    def score(self, K: np.ndarray, M: np.ndarray | None = None, m: int = 1, return_argmax: bool = False,
              batch: int = 256):
        """Scan statistic of each row of K (batch, rounds, positions); M: trials per cell (None: m everywhere).

        With ``return_argmax`` also returns an int array (batch, 4) of the maximizing (round0, pos0, height, width)."""
        K = np.asarray(K)
        out = np.zeros(len(K))
        arg = np.zeros((len(K), 4), dtype=np.int64)
        Q = self.p.shape[1]
        for s in range(0, len(K), batch):
            Kb = K[s:s + batch]
            C = self._integrals(Kb, None if M is None else M[s:s + batch], m)
            best = np.zeros(len(Kb))
            ab = arg[s:s + batch]
            for h in self.heights:
                for w in self.widths:
                    L = self.llr_boxes(Kb, h, w, C=C).reshape(len(Kb), -1)
                    j = L.argmax(1)
                    v = L[np.arange(len(Kb)), j]
                    better = v > best
                    best = np.where(better, v, best)
                    if return_argmax and better.any():
                        q = Q - w + 1
                        ab[better] = np.stack([j[better] // q, j[better] % q, np.full(better.sum(), h),
                                               np.full(better.sum(), w)], 1)
            out[s:s + batch] = best
        return (out, arg) if return_argmax else out


# ---------------------------------------------------------------------------------------------- calibration and planting
def thresholds(null_scores: np.ndarray, fars: Sequence[float]) -> dict:
    """Empirical thresholds: the smallest t with (#null scores > t) / n <= far, for each false-alarm rate."""
    s = np.sort(np.asarray(null_scores))[::-1]
    out = {}
    for a in fars:
        j = int(np.floor(a * len(s)))  # allow j exceedances
        out[float(a)] = float(s[j]) if j < len(s) else float(-np.inf)
    return out


def plant_bursts(grids: np.ndarray, rng: np.random.Generator, R: int, T: int, theta: float,
                 corner: tuple[int, int] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """XOR a burst into every shot of ``grids`` (shots, rounds, positions): each cell of one R-position x T-round
    rectangle flips independently with probability theta. The rectangle is the same for all shots (a burst lasting
    the whole window); its corner (round0, pos0) is uniformly random unless given. Returns (new grids, corner)."""
    g = np.array(grids, dtype=np.uint8, copy=True)
    nT, nQ = g.shape[1:]
    if corner is None:
        corner = (int(rng.integers(0, nT - T + 1)), int(rng.integers(0, nQ - R + 1)))
    t0, q0 = corner
    flips = rng.random((len(g), T, R)) < theta
    g[:, t0:t0 + T, q0:q0 + R] ^= flips.astype(np.uint8)
    return g, corner


# --------------------------------------------------------------------------------------------------------------- theory
def planted_rates(p: np.ndarray, R: int, T: int, theta: float, corner: tuple[int, int]) -> np.ndarray:
    """Per-cell rates after XOR-ing Bernoulli(theta) into the rectangle: p + theta (1 - 2p) inside, p outside."""
    p1 = np.array(p, dtype=np.float64, copy=True)
    t0, q0 = corner
    s = p1[t0:t0 + T, q0:q0 + R]
    p1[t0:t0 + T, q0:q0 + R] = s + theta * (1 - 2 * s)
    return p1


def marked_poisson_divergences(lam0: np.ndarray, lam1: np.ndarray) -> dict:
    """Lemma 5.8 / Theorem 5.9: D_pattern, D_count, and the decomposition D_pattern = D_count + Lambda_1 D(pi_1||pi_0)."""
    l0, l1 = np.ravel(lam0).astype(float), np.ravel(lam1).astype(float)
    dp = float(np.sum(xlogy(l1, l1 / l0) - l1 + l0))
    L0, L1 = l0.sum(), l1.sum()
    dc = float(L1 * np.log(L1 / L0) - L1 + L0)
    pi0, pi1 = l0 / L0, l1 / L1
    dpi = float(np.sum(xlogy(pi1, pi1 / pi0)))
    return dict(D_pattern=dp, D_count=dc, Lambda0=float(L0), Lambda1=float(L1), D_pi=dpi,
                identity_residual=float(dp - dc - L1 * dpi), ratio=dp / dc if dc > 0 else float("inf"))


def _poisson_binomial(p: np.ndarray) -> np.ndarray:
    """Distribution of a sum of independent Bernoulli(p_i) (exact, by convolution)."""
    dist = np.ones(1)
    for q in np.ravel(p):
        dist = np.convolve(dist, [1 - q, q])
    return dist


def bernoulli_divergences(p0: np.ndarray, p1: np.ndarray) -> dict:
    """Independent-Bernoulli cells: D_pattern = sum_i KL(Bern(p1_i)||Bern(p0_i)); D_count = KL of the exact
    Poisson-binomial laws of the totals. Per shot, H1 relative to H0."""
    a, b = np.ravel(p0).astype(float), np.ravel(p1).astype(float)
    dp = float(np.sum(xlogy(b, b / a) + xlogy(1 - b, (1 - b) / (1 - a))))
    f0, f1 = _poisson_binomial(a), _poisson_binomial(b)
    mask = f1 > 0
    dc = float(np.sum(f1[mask] * (np.log(f1[mask]) - np.log(np.maximum(f0[mask], 1e-300)))))
    return dict(D_pattern=dp, D_count=dc, ratio=dp / dc if dc > 0 else float("inf"))
