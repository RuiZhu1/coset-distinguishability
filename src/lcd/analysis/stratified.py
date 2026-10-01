"""Weight-stratified estimation of the logical failure probability p_L(p, e).

Scheme (README sec. 2, "采样方案"):

    p_L(p, e) = sum_{(k,w)} P_{p,e}(k, w) * f(k, w)

with k = number of erased qubits, w = number of Pauli errors among the non-erased ones,
P_{p,e}(k, w) = Binom(n, e)(k) * Binom(n-k, p)(w) computed exactly, and f(k, w) the failure probability of the
decoder on a uniformly random error with exactly k erasures and w Pauli errors.  Only f is estimated by Monte Carlo.

* The sampling cost no longer scales like 1/p_L: it is spent on the strata that carry the failure mass.
* If the decoder does not depend on (p, e) (MWPM with uniform weights), one table f serves every (p, e), and
  derivatives with respect to p and e are analytic (``derivatives``) -- no finite-difference noise.
  For a decoder that depends on the priors (ML) the table is valid for its own (p, e) only; the derivative formula
  then rests on the envelope theorem (theory/, not yet verified numerically for ML).
* f(k, w) = 0 exactly when 2w + k < d (distance d), so those strata are never sampled.

Reporting (README sec. 5): ``estimate`` returns a point estimate, a Jeffreys-smoothed standard error, and a certified
interval built from per-stratum Clopper-Pearson intervals with a Bonferroni correction (strata with zero observed
failures contribute their Clopper-Pearson upper limit; omitted probability mass is added to the upper limit).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import beta, binom

Sampler = Callable[[int, int, int, np.random.Generator], np.ndarray]


class Strata:
    """Strata (k, w) of the code-capacity model for n qubits and distance d, restricted to the strata that carry
    probability > ``cutoff`` under at least one of the target (p, e) pairs."""

    def __init__(self, n: int, d: int | None, targets: Sequence[tuple[float, float]], cutoff: float = 1e-14):
        self.n, self.d, self.targets = n, d, [tuple(map(float, t)) for t in targets]
        kmax = n if any(e > 0 for _, e in self.targets) else 0
        kk, ww = np.meshgrid(np.arange(kmax + 1), np.arange(n + 1), indexing="ij")
        kk, ww = kk.ravel(), ww.ravel()
        ok = kk + ww <= n
        kk, ww = kk[ok], ww[ok]
        keep = np.zeros(kk.size, bool)
        for p, e in self.targets:
            keep |= self._prob(kk, ww, p, e) > cutoff
        self.k, self.w = kk[keep], ww[keep]
        self.size = self.k.size
        self.known_zero = (2 * self.w + self.k < d) if d is not None else np.zeros(self.size, bool)

    def _prob(self, k, w, p, e):
        return binom.pmf(k, self.n, e) * binom.pmf(w, self.n - k, p)

    def prob(self, p: float, e: float) -> np.ndarray:
        return self._prob(self.k, self.w, p, e)

    def omitted_mass(self, p: float, e: float) -> float:
        return float(max(0.0, 1.0 - self.prob(p, e).sum()))

    def dprob(self, p: float, e: float) -> tuple[np.ndarray, np.ndarray]:
        """(dP/dp, dP/de) per stratum, from d/dx Binom(m, x)(j) = m [Binom(m-1, x)(j-1) - Binom(m-1, x)(j)]
        (valid at x = 0, unlike score-function formulas)."""
        n, k, w = self.n, self.k, self.w
        be = binom.pmf(k, n, e)
        dbe = n * (binom.pmf(k - 1, n - 1, e) - binom.pmf(k, n - 1, e))
        m = n - k
        bp = binom.pmf(w, m, p)
        mm = np.maximum(m - 1, 0)
        dbp = np.where(m >= 1, m * (binom.pmf(w - 1, mm, p) - binom.pmf(w, mm, p)), 0.0)
        return be * dbp, dbe * bp


@dataclass
class Estimate:
    p_L: float
    se: float                 # Jeffreys-smoothed standard error
    rel_se: float
    lo: float                 # certified interval (coverage >= 1 - delta)
    hi: float
    omitted_mass: float       # probability mass of strata dropped by the cutoff (already included in hi)
    decodes: int
    dominant: tuple[int, int]  # (k, w) of the stratum with the largest P * f


class StratifiedEstimator:
    def __init__(self, strata: Strata, sampler: Sampler):
        """sampler(k, w, N, rng) -> bool array of length N: decoder failures on N errors from stratum (k, w)."""
        self.strata, self.sampler = strata, sampler
        self.fails = np.zeros(strata.size)
        self.N = np.zeros(strata.size)
        self._P = [strata.prob(p, e) for p, e in strata.targets]

    # ---------------------------------------------------------------- sampling
    def _draw(self, i: int, count: int, rng: np.random.Generator, chunk: int) -> None:
        k, w = int(self.strata.k[i]), int(self.strata.w[i])
        done = 0
        while done < count:
            m = min(chunk, count - done)
            self.fails[i] += int(np.count_nonzero(self.sampler(k, w, m, rng)))
            done += m
        self.N[i] += count

    def run(self, budget: int, rng: np.random.Generator, n0: int = 1000, rounds: int = 3, chunk: int = 20000) -> None:
        """Pilot with n0 samples per non-zero stratum, then ``rounds`` of Neyman allocation until about ``budget``
        decodes in total.  The allocation targets every (p, e) in ``strata.targets`` simultaneously."""
        free = np.flatnonzero(~self.strata.known_zero)
        for i in free:
            if self.N[i] == 0:
                self._draw(i, n0, rng, chunk)
        remaining = budget - int(self.N.sum())
        if remaining <= 0:
            return
        per_round = remaining // rounds
        for _ in range(rounds):
            a = self._neyman_weights()
            target = a / a.sum() * (self.N.sum() + per_round)
            add = np.maximum(target - self.N, 0.0)
            if add.sum() <= 0:
                break
            add = np.floor(add / add.sum() * per_round).astype(int)
            for i in np.flatnonzero(add > 0):
                self._draw(i, int(add[i]), rng, chunk)

    def _neyman_weights(self) -> np.ndarray:
        ft = self._f_smooth()
        a = np.zeros(self.strata.size)
        for P in self._P:
            pl = float((P * ft).sum())
            if pl > 0:
                a = np.maximum(a, P * np.sqrt(ft * (1 - ft)) / pl)
        a[self.strata.known_zero] = 0.0
        return a

    # ---------------------------------------------------------------- tables
    def _f_smooth(self) -> np.ndarray:
        ft = (self.fails + 0.5) / (self.N + 1.0)
        ft[self.strata.known_zero] = 0.0
        return ft

    def f_hat(self) -> np.ndarray:
        f = np.divide(self.fails, self.N, out=np.zeros_like(self.fails), where=self.N > 0)
        f[self.strata.known_zero] = 0.0
        return f

    # ---------------------------------------------------------------- estimates
    @staticmethod
    def p_L_from(strata: Strata, f: np.ndarray, p: float, e: float) -> float:
        return float((strata.prob(p, e) * f).sum())

    @staticmethod
    def derivatives_from(strata: Strata, f: np.ndarray, p: float, e: float) -> tuple[float, float]:
        dp, de = strata.dprob(p, e)
        return float((dp * f).sum()), float((de * f).sum())

    def derivatives(self, p: float, e: float) -> tuple[float, float]:
        """(d p_L / d p, d p_L / d e) at fixed table f.  Exact for rate-independent decoders (MWPM, uniform weights)."""
        return self.derivatives_from(self.strata, self.f_hat(), p, e)

    def estimate(self, p: float, e: float, delta: float = 0.05) -> Estimate:
        S, f, ft, N = self.strata, self.f_hat(), self._f_smooth(), self.N
        P = S.prob(p, e)
        pl = float((P * f).sum())
        var = P ** 2 * ft * (1 - ft) / np.maximum(N, 1.0)
        var = np.where(S.known_zero, 0.0, var)
        se = float(np.sqrt(var.sum()))
        free = ~S.known_zero
        alpha_s = delta / max(int(free.sum()), 1)
        x = self.fails
        lo_s = np.zeros(S.size)
        hi_s = np.ones(S.size)
        pos = free & (N > 0) & (x > 0)
        lo_s[pos] = beta.ppf(alpha_s / 2, x[pos], N[pos] - x[pos] + 1)
        ok_hi = free & (N > 0) & (x < N)
        hi_s[ok_hi] = beta.ppf(1 - alpha_s / 2, x[ok_hi] + 1, N[ok_hi] - x[ok_hi])
        lo_s[S.known_zero] = 0.0
        hi_s[S.known_zero] = 0.0
        omitted = S.omitted_mass(p, e)
        dom = int(np.argmax(P * f)) if pl > 0 else 0
        return Estimate(
            p_L=pl, se=se, rel_se=se / pl if pl > 0 else float("inf"),
            lo=float((P * lo_s).sum()), hi=float(min(1.0, (P * hi_s).sum() + omitted)),
            omitted_mass=omitted, decodes=int(N.sum()), dominant=(int(S.k[dom]), int(S.w[dom])),
        )

    def bootstrap(self, fn: Callable[[np.ndarray], float], B: int, rng: np.random.Generator) -> np.ndarray:
        """Parametric bootstrap of any functional of the table f: resample each stratum's failures
        ~ Binomial(N, f_smooth) and apply fn(f_boot).  Returns the B replicates (use .std() for an error bar)."""
        ft, N = self._f_smooth(), self.N.astype(int)
        out = np.empty(B)
        for b in range(B):
            fb = np.divide(rng.binomial(N, ft), np.maximum(N, 1))
            fb[self.strata.known_zero] = 0.0
            out[b] = fn(fb)
        return out

    def decodes_for_rel_error(self, p: float, e: float, target: float = 0.1) -> float:
        """Decodes needed to reach relative standard error ``target``, extrapolated by 1/eps^2 from the current run."""
        est = self.estimate(p, e)
        return est.decodes * (est.rel_se / target) ** 2
