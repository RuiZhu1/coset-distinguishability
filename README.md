# Logical Coset Distinguishability (LCD)

**A resource theory of surface-code noise: noise preorder, local exchange rates, decoder margins, and finite-data extrapolation certificates**

The full research plan is in [`ABSTRACT.md`](ABSTRACT.md). This README is also the task specification for the code, so that developers (including Claude Code) can start work directly from it.

**The theory track and the numerics track advance in parallel**: proofs and notes are in [`theory/`](theory/README.md) (see Section 7), and the numerics (Section 2) are used to test the theory.

---

## 1. Core idea (one paragraph)

Under maximum-likelihood (coset) decoding, the logical failure probability equals the Bayes error of the hypothesis-testing problem "given the syndrome, distinguish the logical cosets". Hence the **distinguishability of logical cosets** can serve as a resource; syndrome coarse-graining, discarding classical side information, and superposing independent, classically samplable noise are free operations, under which the resource does not increase. The error-suppression exponent

```
alpha = lim_{d -> inf} -(1/d) * log p_L(d)
```

is the large-deviation error exponent of this distinguishing problem, and also equals the free-energy tension of a domain wall in the statistical-mechanics mapping. The numerical part of this repository verifies and quantifies the consequences of this framework.

## 2. Current goals: two preliminary results

The research plan is essentially settled; what matters most now is to produce **concrete numbers**. In order of priority:

### P1: local exchange rate between erasure and Pauli noise under code-capacity noise

**Setting**
- Rotated surface code, distances d in {5, 7, 9, 11, 13, 15} (larger if the tensor network can afford it).
- Code-capacity noise: each data qubit independently is erased with probability e (a uniformly random Pauli I/X/Y/Z is applied, location known); otherwise it suffers depolarizing noise with probability p.
- Two decoders:
  - **ML**: maximum-likelihood (coset) decoding, i.e. the Bravyi-Suchara-Vargo method. Erased qubits get the prior 1/4 for each of I/X/Y/Z, the other qubits the depolarizing prior. `qecsim`'s MPS decoder takes a single prior for all qubits and has no erasure support (checked on qecsim 1.0b9), so this is implemented here: an exact transfer-matrix decoder (`tn_ml.py`, d <= ~11) and a truncated-MPS decoder (`mps_ml.py`, d >= 11); see "ML decoders" below.
  - **MWPM**: PyMatching; erased qubits get edge weight 0 (or very small). Implemented for e = 0; erasure is still to do.

**Quantities to compute**
1. Estimate p_L(d; p, e) on a (p, e) grid, with confidence intervals (see "Sampling scheme").
2. For each (p, e), fit `log p_L(d) ~ a - alpha*d` to get alpha(p, e). Report the range of d used for the fit, and check whether alpha is stable after dropping the smallest distance (a diagnostic for finite-size corrections).
3. **Local exchange rate**: at a work point (p0, e0)

   ```
   R_{e->p}(p0, e0) = (d alpha/d e) / (d alpha/d p)      (both derivatives are negative, so R > 0)
   ```

   Meaning: along a level curve of alpha, increasing the erasure rate by one unit is equivalent to increasing the Pauli error rate by how much. Compute it from the analytic derivatives of the stratified estimator (see "Sampling scheme"), and give error estimates.
4. **Decoder margin**: alpha_ML(p, e) - alpha_MWPM(p, e).

**Suggested work points**: e0 in {0, 0.02, 0.05}, p0 in {0.02, 0.04, 0.06}. Adjust according to statistics and compute time.

**Sanity checks (must pass first)** — script `experiments/p1_erasure_pauli/threshold_check.py`, data in `results/threshold_check_*.{csv,json}`. Finite-size crossings of consecutive distances (weighted straight-line fit of the difference of the two curves; the estimates depend on the fit window and on the sampling seed at the level of 0.3-0.5 percentage points; an earlier independent run of the depolarizing scenario gave 18.44 +- 0.21% and 18.69 +- 0.23% for ML):

| Check | Known value | Measured crossings | Verdict |
|---|---|---|---|
| Pure depolarizing noise, exact ML | about 18.9% | 18.74 +- 0.15% (d = 5, 7), 18.54 +- 0.32% (d = 7, 9) | consistent (about 1 sigma) |
| Pure erasure, exact ML | 50% | 50.10 +- 0.31% (d = 5, 7), 50.01 +- 0.36% (d = 7, 9) | pass |
| Pure bit-flip noise, MWPM | about 10.3% | 9.46%, 9.71%, 9.61%, 9.63%, 9.82% (d = 5 to 15, consecutive pairs; +- 0.1%) | **approaches from below, 5-8% under 10.3% at d <= 15**; no finite-size-scaling fit yet, so this check is not closed |
| Depolarizing noise, MWPM (X/Z independent) | about 15% (literature value, not verified here) | 13.63 +- 0.21%, 14.47 +- 0.17% (d = 5, 7, 9) | no claim |

If numerical results deviate noticeably from the known values, debug the implementation first and do not continue.

**Checks from the theory track** (see `theory/`, Theorems 3.1/3.2):
- Monotonicity: the ML p_L(d; p, e) is nondecreasing in p and e; for any delta, p_L(p, e0+delta) <= p_L(p'', e0), where p'' = p + delta(3/4 - p)/(1 - e0).
- Exchange-rate bound: `R_{e->p}(p0, e0) <= c(p0, e0) = (3/4 - p0)/(1 - e0)`. Note that this is not simply 3/4: for p0 < (3/4) e0 we have c > 3/4 (for example c = 0.768 at the work point (0.02, 0.05)).
- Report the gap Delta = c - R, and compare with the analytic reference values R_B (Bhattacharyya) and R_hash (quantum hashing capacity); the table of reference values is in `theory/sec5-information.tex`.
- Small-code exact oracle: `theory/checks/exact_small_codes.py` gives the exact ML p_L(p, e) for codes with n <= 9. The ML decoder agrees with it on the d = 3 rotated surface code to machine precision (`tests/test_tn_ml.py`).

**ML decoders (M1; code `src/lcd/decoders/tn_ml.py` and `mps_ml.py`)**

The coset sums Z_c = sum_{g in S} P(R_c g) are a two-dimensional classical partition function: every stabilizer generator (plaquette) carries a binary spin, and qubit j carries the Pauli R_j X^x Z^z with x, z the XOR of the X-type and Z-type spins of the four plaquettes around it. Ties between classes (positive probability under erasures) are broken uniformly at random, which attains the Bayes optimum. Per-qubit priors (hence erasures) are built in. Two contractions are provided:

*Exact decoder* (`ExactTNMLDecoder`): a row-by-row transfer matrix whose frontier has d + 3 binary variables; cost O(n 2^(d+3)) per decode, no truncation, batch-vectorized. It is the reference implementation and the oracle for the truncated decoder.

*Truncated-MPS decoder* (`MPSMLDecoder`, Bravyi-Suchara-Vargo): the boundary state is an MPS over the d + 1 plaquette columns; one row of qubits is an MPO of bond dimension 4; after each row the MPS is brought to left-canonical form by a QR sweep and truncated to bond dimension chi by a right-to-left SVD sweep; norms are kept as logs. Cost O(d^2 chi^3) per decode; for chi >= 2^((d+1)/2) nothing is truncated and it reproduces the exact decoder (<= 5e-10). Samples whose truncated partition function is not positive (this happens with erasures at small chi) are automatically decoded again with doubled chi (`last_fallback`). A **convergence monitor** (`fail_prob(..., refine_chi=2*chi)`) re-decodes only the ambiguous samples (best wrong class within `margin` = 8 of the true class in log Z) with a larger chi and reports how many decisions changed (`last_refine_changed`); on working strata this touches 0.3-9% of the samples (+3-25% time) and measures the truncation error of the run directly. (A single SVD sweep without the QR sweep was tried and rejected: it is not exact even for chi at least the Schmidt rank.)

Measured cost per decode (ms; one core, batched, idle machine, depolarizing noise at a stratum typical of p = 0.04; `experiments/p1_erasure_pauli/decoder_timing.py`, `results/decoder_timing.json`):

| d | 7 | 9 | 11 | 13 | 15 |
|---|---|---|---|---|---|
| exact | 0.22 | 1.5 | 8.8 | 63 | 433 |
| MPS chi = 4 | 2.4 | 4.5 | 8.8 | 11.8 | 16.9 |
| MPS chi = 6 | 2.9 | 7.4 | 14 | 20 | 32 |
| MPS chi = 8 | 3.1 | 10 | 20 | 33 | 49 |
| MPS chi = 12 | 4.2 | 15 | 33 | 63 | 99 |

The exact decoder is faster up to d = 11; the MPS decoder is 3x (chi = 6) to 14x faster at d = 13 / 15 (chi = 4: 5x / 26x). `threads=2` gains another factor of about 1.7; multiprocessing over strata scales better.

Validation of the exact decoder (all passing): coset sums equal brute-force enumeration at d = 3 including erasures (asserted to 1e-12, observed 1e-15); the Bayes error summed over all 4^9 errors equals the exact value (`tests/test_tn_ml.py`); stratified ML estimates with erasures agree with the exact oracle at (p, e) = (0.05, 0.10) and (0.15, 0.30) (|z| < 1.3, certified intervals cover); an independent implementation (qecsim 1.0b9, `RotatedPlanarMPSDecoder`, chi = 16) agrees at d = 3 and d = 5, p = 0.10 (|z| < 1.6). Details in `results/ml_crosscheck.json`.

Validation of the MPS decoder (`tests/test_mps_ml.py`, 21 tests, plus two experiments). Only the **decisions** are validated: individual log Z_c of classes far from the truth can be off by O(1) at moderate chi, so do not use Z_c as probabilities.
- *e = 0*, `mps_validation.py` (`results/mps_validation.json`): on the strata that carry p_L at p = 0.04 and 0.06 (90% of p_L, from the MWPM table), using an MWPM-enriched stratified design (all MWPM failures plus a Bernoulli subsample of the successes, Horvitz-Thompson weights), paired with the exact decoder (d = 11, 13) or with chi = 16 anchored to it (d = 15):

  | d, p | samples tested (ML-failing) | chi = 3 | chi = 4 | chi = 6, 8 |
  |---|---|---|---|---|
  | 11, 0.04 | 8890 (364) | 16 changed, +0.8 +- 0.7% | 0 changed | 0 changed |
  | 11, 0.06 | 9544 (554) | 24 changed, +1.9 +- 0.6% | 0 changed | 0 changed |
  | 13, 0.04 | 8136 (212) | 14 changed, +0.6 +- 0.6% | 1 changed, -0.13 +- 0.13% | 0 changed |
  | 13, 0.06 | 9701 (392) | 29 changed, +75 +- 42% | 3 changed, +6 +- 6% | 0 changed |
  | 15, 0.04 | 6289 (52) | 9 changed (weighted +1000%) | 1 changed, +3 +- 3% | 0 changed (chi = 12 too) |

  (entries: changed decisions, and the relative change of the p_L contribution of these strata; the percentages at chi = 3 are inflated by the weights of the few flipped samples.) chi = 6 changed no decision anywhere. At d = 15 only 52 ML-failing samples were tested, so "no flip" bounds the relative error of p_L by about 3/52 = 6% (95%); the anchor (chi = 16 against the exact decoder, 480 samples) has no changed decision.
- *With erasures*, `mps_erasure_check.py` (`results/mps_erasure_check.json`): 30,000 samples in ten strata at d = 11, 13 (k = 4..10 erasures, w = 10..18 errors, 785 ML-failing; strata with exactly zero failures were discarded because they test nothing), plain paired comparison with the exact decoder:

  | chi | 4 | 6 | 8 | 12 |
  |---|---|---|---|---|
  | decisions changed | 7 | 0 | 0 | 0 |
  | samples needing the invalid-Z fallback | 5.8% | 1.6% | 0.26% | 0.01% |

  **Erasures need a larger chi than e = 0.**

*Recommended chi.* e = 0: chi = 6. With erasures: chi >= 8. In every production run use the monitor (`refine_chi = 2*chi`) and report `last_refine_changed` and `last_fallback`.

*Not verified.* The envelope identity (below) has not been tested directly with the MPS decoder (its decisions agree with the exact decoder on all tested samples, so it is expected to transfer); d = 15 with erasures; the test power at d = 15 is limited (above); chi was validated at p = 0.04 and 0.06, not at larger p.

**Sampling scheme (settled; code `src/lcd/analysis/stratified.py`, benchmark `experiments/p1_erasure_pauli/sampling_benchmark.py`, data `results/p1_sampling_benchmark.*`)**

*Why it is needed.* Naive Monte Carlo needs about 1/(eps^2 p_L) decodes to reach relative error eps. At p0 = 0.02, d = 15 we have p_L ~ 4e-7, so 10% relative error needs about 2e8 decodes (already marginal for MWPM, and infeasible for a tensor-network ML decoder).

*Scheme.*
1. **Stratify by error weight.** p_L(p, e) = sum_{k,w} P_{p,e}(k, w) * f(k, w), where k is the number of erased qubits and w the number of Pauli errors on non-erased qubits; P = Binom(n, e)(k) * Binom(n-k, p)(w) is computed exactly, and **only f(k, w) is estimated by Monte Carlo** (given (k, w), the errors are uniform in position and Pauli type, independent of (p, e)). f(k, w) = 0 exactly when 2w + k < d, so those strata are never sampled.
2. **Allocation.** Each nonzero stratum gets a pilot of 1000 samples, then 3 rounds of Neyman allocation (proportional to P*sqrt(f(1-f))) up to the total budget. **Each (d, p0, e0) target is allocated separately**; when one batch of samples must serve several targets, take the maximum of the normalized weights.
3. **Error bars.** Report the point estimate, the Jeffreys-smoothed relative standard error, and a **certified interval**: per-stratum Clopper-Pearson (Bonferroni-corrected), where strata with zero observed failures contribute their upper limit and the truncated probability mass is added to the upper limit. In the hard corner the certified interval is much wider than the standard error (about an order of magnitude at d = 15, p = 0.02); this is the real cost of certification, not a defect. Errors of fitted quantities (alpha, exchange rate) come from a parametric bootstrap (`StratifiedEstimator.bootstrap`).
4. **No finite differences for the exchange rate.** For decoders independent of (p, e) (MWPM, uniform weights) one table f serves the whole (p, e) plane, and dp_L/dp, dp_L/de follow from the analytic derivatives of P (`derivatives`). For ML (the decoder depends on the prior) the table is valid only at its matched (p, e); the derivative formula rests on the envelope argument (`theory/` Remark 3.13): the derivative of the ML error equals the derivative of the failure rate of the decoder frozen at the working point, so each work point needs a single decode. The ML decoder depends on p only (erased positions are known, so e never enters the posterior), so the e-derivative is an exact identity and only the p-derivative needs the envelope argument. The identity has been verified to relative 2e-10 with exact ML on three codes with n <= 9 (check C8 in `theory/checks`, Observation 3.14) and at d = 5, 7 on common random samples (`experiments/p1_erasure_pauli/envelope_check.py`, `results/envelope_check.json`): after separating the O(h^2) discretisation error of the central difference, the difference between re-matched and frozen decoders is at most 0.04% of the derivative at step h (within 1.6 standard errors in all four cases, which include a d = 5 case with erasure strata), and one 2.05-sigma deviation (0.1% of the derivative) appears at step 2h at d = 7. **It has not been verified for truncated tensor-network ML, nor at d >= 9.**
5. **Cost and the d range of ML.** The table gives the decodes needed for 10% relative error (MWPM, e = 0; stratified numbers are extrapolated by 1/eps^2 from measured runs with 1.5e6 decodes, with a conservative Jeffreys standard error):

   | | p = 0.02 naive -> stratified | p = 0.04 | p = 0.06 |
   |---|---|---|---|
   | d = 11 | 9.7e6 -> 1.9e5 | 2.0e5 -> 3.1e4 | 2.2e4 -> 7.5e3 |
   | d = 13 | 5.6e7 -> 1.3e6 | 5.1e5 -> 1.0e5 | 4.0e4 -> 1.6e4 |
   | d = 15 | 2.4e8 -> 1.3e7 | 1.4e6 -> 3.5e5 | 7.4e4 -> 3.2e4 |

   Stratification gains about 20-50x at p = 0.02 and only 2-6x at p >= 0.04. Combining with the measured cost per decode and assuming the ML strata profile is similar to MWPM's (to be re-measured): one core, the exact decoder for d <= 11 and the MPS decoder (chi = 6 at e = 0, with the +10% monitor) for d >= 13 give for 10% relative error: d = 11 at most 30 minutes per target; d = 13: 6 min (p = 0.06), 37 min (0.04), 8 h (0.02); d = 15: 19 min (0.06), 3.4 h (0.04), **126 h (0.02)**. With erasures use chi = 8 (about 1.6x slower at d = 13, 15). Multiply the decode counts by 4 for 5% error; divide the times by about 4 on four cores. **So every work point is reachable up to d = 13 and, for p >= 0.04, up to d = 15; only (p = 0.02, d = 15) stays expensive** (reduce the target precision, or the Bravyi-Vargo rare-event MCMC, PRA 88, 062308). MWPM is not limited in this way.

6. **M2 result: the ML exchange rate (first pass).** `experiments/p1_erasure_pauli/p1_exchange_rate.py`, data `results/p1_exchange_rate.json`, tables `results/p1_m2/tables/`. ML decoder matched at p0 (exact for d <= 11; truncated MPS, chi = 8 with the monitor, at d = 13), weight-stratified tables per (d, work point), Neyman allocation for p_L and both derivatives, analytic derivatives (frozen decoder for p by the envelope argument, exact for e), parametric bootstrap (1000 replicates). `R^(d) = (dp_L/de)/(dp_L/dp)`; `R_alpha = (d alpha/de)/(d alpha/dp)` from inverse-variance weighted slope fits in d (window 5-11); `c = (3/4 - p0)/(1 - e0)`; `R_B` is the Bhattacharyya reference (heuristic, theory section 5):

   | (p0, e0) | c | R^(5) | R^(7) | R^(9) | R^(11) | R_alpha (d = 5-11) | R_alpha / c | R_B |
   |---|---|---|---|---|---|---|---|---|
   | (0.06, 0) | 0.690 | 0.387 +- 0.028 | 0.289 +- 0.038 | 0.382 +- 0.072 | 0.209 +- 0.118 | 0.15 +- 0.11 | 0.22 | 0.244 |
   | (0.06, 0.02) | 0.704 | 0.323 +- 0.014 | 0.273 +- 0.014 | 0.282 +- 0.020 | 0.257 +- 0.034 | 0.205 +- 0.039 | 0.29 | 0.249 |
   | (0.06, 0.05) | 0.726 | 0.308 +- 0.010 | 0.294 +- 0.010 | 0.260 +- 0.012 | 0.278 +- 0.019 | 0.237 +- 0.024 | 0.33 | 0.257 |
   | (0.04, 0) | 0.710 | 0.313 +- 0.026 | 0.431 +- 0.048 | 0.384 +- 0.096 | 0.251 +- 0.138 | 0.53 +- 0.13 (window-dependent) | 0.74 | 0.221 |
   | (0.04, 0.02) | 0.724 | 0.300 +- 0.012 | 0.237 +- 0.015 | 0.270 +- 0.029 | 0.311 +- 0.044 | 0.196 +- 0.043 | 0.27 | 0.226 |
   | (0.04, 0.05) | 0.747 | 0.264 +- 0.008 | 0.265 +- 0.011 | 0.252 +- 0.018 | 0.231 +- 0.024 | 0.231 +- 0.028 | 0.31 | 0.233 |

   * **Theorem 3.2 is not violated**: `R^(d) <= c` in every entry with d <= 11, with a large margin (`R^(d)/c` is about 0.3 to 0.6). At e0 > 0 the exponent rate is `R_alpha / c` about 0.3, that is, an erasure is worth about a third of the Pauli noise that the flag-discarding bound allows; the gap `c - R_alpha` is about 0.5.
   * At e0 = 0.02 and 0.05 `R_alpha` agrees with the Bhattacharyya reference `R_B` within about one standard error (ratios 0.82-0.99): hypothesis H_B (theory Conjecture 3.12) is **consistent with the data, not confirmed** (the fits cover d = 5-11 only, and the fit window matters at the level of the quoted errors).
   * The e0 = 0 points are poor: dp_L/de at e = 0 is a difference between the strata with one erasure and without, with large variance; `R_alpha` at e0 = 0 is window-dependent (0.06 to 0.64) and carries no information beyond `R^(d) <= c`.
   * **The cost estimates of item 5 were too optimistic for ML.** They assumed the MWPM strata profile (not re-measured at the time); the ML failure fractions in the dominant strata are far smaller (for example 0.2% in the dominant stratum k = 0, w = 15 at d = 13, p0 = 0.04, where 92 failures were seen in 6.0e4 decodes). At d = 13 the relative standard error of p_L is 54% (p0 = 0.06, e0 = 0.05, 3.0e4 decodes) to 870% (p0 = 0.04, e0 = 0, 6.0e4 decodes); extrapolating by 1/eps^2, 10% needs about 10^6 to 10^7 decodes (10-100 core-hours per point at 40 ms per decode). d = 13 therefore enters the fits only with its negligible inverse-variance weight, and d = 11 had to be raised to 2e5-4e5 decodes (relative standard errors 4-12%). p0 = 0.02 was not run.
   * Not done / not verified: the ML-MWPM margin (MWPM with erasure is not implemented); the envelope identity for ML at d >= 9 and for the MPS decoder (the d = 13 data carry almost no weight); finite-size drift beyond d = 11.
   * Two decoder bugs fixed on the way: a production run aborted at d = 13 because a class whose true weight is below the truncation noise of the best class stays non-positive even at chi = 64; such classes are now dropped as non-maximal (`last_unrepaired`: 2 samples among the 2.7e5 d = 13 decodes of the production run, none in the validation runs); see `results/mps_invalid_check.json`.

*Verified and not verified.* Verified (`tests/`): estimates, derivatives, and interval coverage on toy decoders with exactly known f (one-dimensional and with erasure strata); the stratified MWPM estimate agrees with naive Monte Carlo at d = 5; stratified ML with erasures agrees with the exact oracle at d = 3 (`results/ml_crosscheck.json`); decoders never fail on strata with 2w + k < d. **Not verified:** the envelope identity for truncated tensor-network ML and at d >= 9; MWPM with erasure (it needs per-sample zero-weight edges and is not implemented); the pilot overhead of two-dimensional (k, w) strata with a real decoder at larger d (the number of strata is about an order of magnitude larger than in one dimension).

### P2: detection of local burst events; space-time pattern discrimination versus Poisson counting

**Setting**
- Use stim to generate rotated surface-code memory experiments, with uniform circuit-level depolarizing noise p = 1e-3 as the background, distances d in {5, 7, 9}, and a tunable number of rounds.
- **Burst-event injection**: burst events are triggered at rate r (per qubit per round) at random space-time locations; qubits within radius R of the event are affected for T_b rounds, during which the error rate at those locations rises to p_b (for example 0.1 to 0.3).

  Implementation hint: for Pauli noise, detection events are linear in the errors (the syndrome is the parity check of the Pauli frame). One can therefore sample separately the detection events produced by the "background" and by "circuits with noise only in the burst regions" and superpose them by bitwise XOR, without expressing random spatial correlations in a single stim circuit.

**Comparing two methods**
- **Baseline (Poisson counting)**: count only logical failures or "high-weight syndromes", and test H0 (no burst) against H1 (bursts at rate r).
- **Pattern method**: exploit the space-time clustering of detection events, for example local detection-event counts in sliding windows, likelihood ratios, or matched-filter statistics, to do the same test and to estimate r and R.

**Quantities to report**
- For a given significance delta and power, the number of rounds (or shots) needed by each method, and their ratio.
- Bias and variance of the estimates of r and R.
- Compare explicitly with the Poisson limit ln(1/delta)/r: the advantage of the pattern method shows up in constants and in distinguishing event classes; **do not claim to beat the 1/r scaling**.

## 3. Repository layout

```
.
├── ABSTRACT.md
├── README.md
├── pyproject.toml
├── theory/                 # theory track: LaTeX notes, proof-status board, exact small-code checks (Section 7)
│   ├── main.tex, sec*.tex
│   ├── README.md           # status board: every statement is PROVED / SKETCH / NUMERICAL / CONJECTURE / PLAN
│   └── checks/             # exact ML check script (numerical falsification tests of the theory)
├── src/lcd/
│   ├── noise/              # noise models: depolarizing, erasure, bit flip (done: code_capacity.py); local bursts, chip-scale events (to do)
│   ├── codes/              # rotated surface code: construction, check matrices, logical operators (done)
│   ├── decoders/
│   │   ├── tn_ml.py        # exact transfer-matrix ML decoder with per-qubit priors (done; d <= ~11)
│   │   ├── mps_ml.py       # truncated-MPS ML decoder (Bravyi-Suchara-Vargo) with fallback and convergence monitor (done; d >= 11)
│   │   └── mwpm.py         # PyMatching wrapper (done for e = 0; erasure to do)
│   ├── circuits/           # stim circuit generation and burst-event injection (to do)
│   └── analysis/
│       ├── stratified.py   # stratified estimator, certified intervals, analytic derivatives (done)
│       ├── crossing.py     # threshold crossing of two curves (done)
│       ├── fit_alpha.py    # fit of alpha and finite-size diagnostics (to do)
│       ├── exchange.py     # local exchange rate (to do)
│       └── detection.py    # test statistics of the Poisson baseline and the pattern method (to do)
├── experiments/
│   ├── p1_erasure_pauli/   # sampling_benchmark.py, crosscheck_ml.py, threshold_check.py
│   └── p2_burst_detection/
├── results/                # raw data (CSV/JSON/NPZ) with configuration, seeds and code version
└── tests/                  # unit tests (26+); threshold sanity checks are run by experiments/.../threshold_check.py
```

## 4. Environment

Python >= 3.10.

```bash
pip install stim sinter pymatching numpy scipy matplotlib pandas
pip install qecsim        # optional: independent MPS ML decoder, used for cross-checks
pip install quimb         # optional: if implementing the truncated tensor-network decoder with it
```

Development install and tests: `pip install -e ".[dev]" && pytest` (about 10 seconds).

Theory track: `make -C theory check` needs numpy and scipy (both are core dependencies of the package); `make -C theory pdf` needs a minimal TeX Live: `sudo apt-get install -y --no-install-recommends texlive-latex-recommended latexmk` (about 120 MB; plain `pdflatex`, no CJK fonts).

**Cloud sessions.** A Claude Code cloud session runs in an ephemeral container: anything installed with `apt-get` or `pip` disappears when the container is reclaimed. To avoid reinstalling, add the install commands above to the **setup script** of the cloud environment (environment menu in the session title bar, then Edit); it runs when each new session starts. Files committed and pushed to the repository persist.

## 5. Engineering rules

- **Reproducible**: every random process uses an explicit seed; every run writes its configuration (distance, noise parameters, shots, seeds, code version) together with its results into `results/`.
- **Statistics**: every p_L comes with an interval. Stratified estimates report the point estimate, the relative standard error, and the per-stratum Clopper-Pearson (Bonferroni) certified interval, and **points with relative standard error above 10% are flagged in plots**; points from naive Monte Carlo use Wilson intervals, and points with fewer than about 100 failures are flagged (this old rule applies only to naive Monte Carlo and is meaningless for stratified estimates). Fits report parameter errors (parametric bootstrap). Reports state the decoder (MWPM or ML) and the sampling method.
- **Tests first**: the threshold sanity checks of Section 2 are run before the production experiments.
- **Data collection**: for circuit-level experiments prefer `sinter` for parallel collection.
- **No overclaiming**: reports distinguish "rigorous conclusions" from "numerical estimates"; MWPM results must not be presented as ML results; code-capacity conclusions must not be extrapolated to circuit-level noise.
- **Theory-numerics loop**: every theorem in `theory/` comes with a numerical falsification test (the correspondence table is in `theory/sec6-map.tex`). Monotonicity, the union bound, and the certification theorem are statements about Bayes-optimal (ML) decoding, and **MWPM data can neither confirm nor refute them**; lower-bound statements (floor, Poisson bound) hold for any decoder and can be tested with stim + MWPM.
- **State the proof status**: when citing a theoretical conclusion, give its status tag; `PROVED` is still a draft that has not been independently reviewed.

## 6. Milestones

| Stage | Content | Acceptance criterion |
|---|---|---|
| M0 | Code construction, noise models, MWPM decoding, sanity checks | Three threshold checks pass. Status: rotated surface code, MWPM (e = 0), stratified sampling module and tests are done; pure erasure and depolarizing-ML crossings are consistent with 50% and 18.9%; the bit-flip MWPM crossing is 5-8% below 10.3% at d <= 15 and needs a finite-size-scaling fit; MWPM with erasure is to do |
| M1 | ML decoding (code capacity, with erasure priors) | Depolarizing ML threshold about 18.9%, trend consistent with MWPM. Status: **done**: exact transfer-matrix decoder (d <= ~11) and truncated-MPS decoder (d >= 11, validated against the exact decoder at e = 0 and with erasures; chi = 6 / 8 recommended), thresholds consistent with 50% and 18.9%. Open: the envelope identity tested directly with the MPS decoder and at d >= 9; d = 15 with erasures; limited test power at d = 15 |
| M2 | Full P1 result | R_{e->p} and the ML-MWPM margin at at least 3 work points, with errors |
| M3 | Burst-event injection and Poisson baseline | The injected event rate can be estimated without bias |
| M4 | Full P2 result | Ratio of the experiment time needed by the pattern method relative to Poisson counting, with errors |
| M5 | Summary | `results/SUMMARY.md`: key numbers, figures, limitations |

The theory-track milestones run **in parallel** with the table above (from week one, without waiting for the numerics):

| Stage | Content | Interlock with the numerics / acceptance criterion |
|---|---|---|
| T0 | Constitutional audit against the resource-theory axioms (theory §0), monotonicity lemma (three free operations), representation of erasure/leakage/events, exact small-code checks | First draft and `theory/checks` exist (all pass); M1 uses them as the oracle (done) |
| T1 | Exchange-rate bound R <= (3/4-p0)/(1-e0); preorder theorem for the Pauli+erasure class | M2 reports R <= c and the gap Delta; P1 decides whether the gap comes from code dependence or incompleteness of the free operations |
| T2 | Counterexample search for "code-universal order = local free order"; proof of Observation 3.5 | Exact computation on a library of random stabilizer codes; no large compute needed |
| T3 | Conditional certification: code-capacity version (done) -> bursts + chip events (needs O1-O3) | M3/M4: floor scaling, parameter estimation, certificate >= exact/TN-ML failure rate |
| T4 | Information-theoretic bounds on the exchange rate; experimental-design theorem (Stein/Chernoff exponents) | M4: T_count / T_pattern against the Poisson bound ln(1/delta)/r |

## 7. Theory track and out of scope

### 7.1 The theory track: advanced in parallel in `theory/`

Theory does not wait for the numerics to finish; it starts in week one, theory leads, and the numerics test the theory (including the monotonicity lemma). The notes are in `theory/` (LaTeX, `make -C theory pdf`); the status board is in [`theory/README.md`](theory/README.md). Contents and order. **Section 0 of the notes is a constitutional audit**: it states the resource-theory axioms assumed (free objects, free operations closed under composition and product, golden rule, faithful and complete monotones, rates), checks the framework against them, and fixes what may be claimed (status of every item in `theory/README.md`; the axiom list is the author's assumption and is to be confirmed or replaced by the group's reference formulation).


1. **Monotonicity under the three free operations** (rigorous proofs): coarse-graining and discarding side information by the data-processing inequality; superposing independent, classically samplable noise by a simulation argument; how erasure and leakage are represented in the model, under which assumptions the argument still holds and where it fails.
2. **The preorder theorem in the simplest subclass.**
3. **A necessary and sufficient condition for Pauli plus erasure noise at code capacity.** Entry point: an erased qubit carries a location flag; discarding the flag is a free operation, and after discarding it the qubit is a depolarizing error of rate 3/4. Monotonicity then gives a rigorous upper bound on the exchange rate, `R_{e->p}(p0, e0) <= (3/4 - p0)/(1 - e0)` (equal to 3/4 to first order for small p0, e0). The numerics of P1 compute the true exchange rate; given the erasure threshold of about 50% and the depolarizing ML threshold of about 18.9%, the true rate should be far below the bound. **This gap is itself a research question**: it measures how much of the value of erasure the "discard the flag" conversion leaves out, that is, which finer monotones the preorder still needs.
4. **Conditional certification theorem.** In the model class with local burst events and chip-scale events, give finite-sample upper bounds directly on p_L(d); the numerics of P2 provide the tests.
5. **Information-theoretic upper and lower bounds on the exchange rate, and the experimental-design theorem**: using channel divergences and Stein and Chernoff exponents.

Current progress and what is unfinished (which parts have a complete written proof and which only have statements and routes) is in `theory/README.md`.

**Counterexample search for the single-loss form of Conjecture 3.11** (`theory/checks/universal_order_search.py`, notes section 3.6, `results/universal_order_search.json`). The question: is every pair of code-capacity noises that no local free operation connects reversed by the ML failure rate of some linear decoding structure? Proved: the free order is cut out by two scalars (p and mu = (1-e)(1-4p/3)); the uncoded qubit has eps* = (3/4)(1 - mu) and a two-qubit structure ("guess E0 from E0 + E1") has eps* = (3/4)(1 - (1-e^2)(1-4p/3)), so the conjecture holds for every unreachable pair with e' < e and for all pairs with nu' > nu. Exhaustive over **all** linear structures on up to 3 qubits (92,881 of them) plus hill-climbing at 4 and 5 qubits: no structure reverses the six "trade" pairs tested (less Pauli noise, more erasure; smallest failure-rate ratio found at 4 and 5 qubits: 1.28, typically 1.5 to 3.7). This is evidence that the single-loss form is **false**, not a proof (n >= 6 is not searched). It does not affect the all-losses form, which is proved; no new free operation is needed. Follow-up (notes section 3.7, `theory/checks/hardened_structures_search.py`, `exponent_criterion.py`): (i) proved obstructions to "erasure-hardened" structures: a logical operator of weight w forces a failure probability of at least 1/2 once its support is erased; a structure that fails at order p^j needs a logical operator of weight at most 2j, so no structure that fails at first order can tolerate three arbitrary erasures; (ii) the marginal exchange rate of the k-fold Pauli repetition structure tends to c * 2e/(2e + (k-1)(1-e)) as p -> 0, so the smallest rate over structures tends to 0 as p0 -> 0 (a lower bound on the exchange rate cannot be uniform in p0), while at p0 = 0.0213 it is positive in every search (exhaustive up to 3 qubits, hill-climbing at 4, 5 and 6); (iii) the channel-coding exponent criterion that would prove a reversal by large structures is not met by any of the six trade pairs. **The single-loss conjecture is still undecided**: evidence against it at p0 = 0.0213, no proof either way.

### 7.2 Out of scope (not for the current stage)

- Coherent errors (the framework relies on the twirling approximation; see the limitations part of ABSTRACT).
- qLDPC codes.
- Exact ML decoding under circuit-level noise.

## 8. References

- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, *Topological quantum memory*, J. Math. Phys. (2002).
- S. Bravyi, M. Suchara, A. Vargo, *Efficient algorithms for maximum likelihood decoding in the surface code*, Phys. Rev. A (2014).
- S. Bravyi, A. Vargo, *Simulation of rare events in quantum error correction*, Phys. Rev. A 88, 062308 (2013).
- C. T. Chubb, S. T. Flammia, *Statistical mechanical models for quantum codes with correlated noise*, Ann. Inst. Henri Poincare D (2021).
- C. Gidney, *Stim: a fast stabilizer circuit simulator*, Quantum (2021).
- O. Higgott, C. Gidney, *Sparse Blossom: correcting a million errors per core second with minimum-weight matching* (PyMatching v2).
- Y. Wu, S. Kolkowitz, S. Puri, J. D. Thompson, *Erasure conversion for fault-tolerant quantum computing in alkaline earth Rydberg atom arrays*, Nat. Commun. (2022).
- M. McEwen et al., *Resolving catastrophic error bursts from cosmic rays in large arrays of superconducting qubits*, Nat. Phys. (2022).
- Google Quantum AI, *Quantum error correction below the surface code threshold*, Nature (2025).

## 9. License

To be decided.
