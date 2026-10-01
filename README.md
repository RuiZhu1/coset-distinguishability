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
  - **ML**: maximum-likelihood (coset) decoding, i.e. the Bravyi-Suchara-Vargo method. Erased qubits get the prior 1/4 for each of I/X/Y/Z, the other qubits the depolarizing prior. `qecsim`'s MPS decoder takes a single prior for all qubits and has no erasure support (checked on qecsim 1.0b9), so this is implemented here (`src/lcd/decoders/tn_ml.py`, see "ML decoder" below).
  - **MWPM**: PyMatching; erased qubits get edge weight 0 (or very small). Implemented for e = 0; erasure is still to do.

**Quantities to compute**
1. Estimate p_L(d; p, e) on a (p, e) grid, with confidence intervals (see "Sampling scheme").
2. For each (p, e), fit `log p_L(d) ~ a - alpha*d` to get alpha(p, e). Report the range of d used for the fit, and check whether alpha is stable after dropping the smallest distance (a diagnostic for finite-size corrections).
3. **Local exchange rate**: at a work point (p0, e0)

   ```
   R_{e->p}(p0, e0) = -(d alpha/d e) / (d alpha/d p)
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

**ML decoder (M1; code `src/lcd/decoders/tn_ml.py`)**

The coset sums Z_c = sum_{g in S} P(R_c g) are a two-dimensional classical partition function (qubit j carries the Pauli R_j X^x Z^z, with x and z the XOR of the X-type and Z-type stabilizer variables touching j). The decoder contracts it **exactly** by a row-by-row transfer matrix whose frontier has d + 3 binary variables: cost O(n 2^(d+3)) per decode, no bond-dimension truncation, per-qubit priors (hence erasures) built in, and the batch of samples is vectorized. Ties between classes (which have positive probability under erasures) are broken uniformly at random, which attains the Bayes optimum.

Measured cost per decode (single core, batched, depolarizing noise at the dominant stratum): 

| d | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|
| ms per decode | 0.07 | 0.27 | 1.2 | 8.6 | 51 |

d = 15 is extrapolated at about 0.25 s (not measured). Beyond d ~ 11-13 a truncated MPS contraction (Bravyi-Suchara-Vargo) is needed; the exact decoder will then serve as its oracle.

Validation (all passing): coset sums equal brute-force enumeration at d = 3 including erasures (asserted to 1e-12, observed 1e-15); the Bayes error summed over all 4^9 errors equals the exact value (`tests/test_tn_ml.py`); stratified ML estimates with erasures agree with the exact oracle at (p, e) = (0.05, 0.10) and (0.15, 0.30) (|z| < 1.3, certified intervals cover); an independent implementation (qecsim 1.0b9, `RotatedPlanarMPSDecoder`, chi = 16) agrees at d = 3 and d = 5, p = 0.10 (|z| < 1.6). Details in `results/ml_crosscheck.json` (script `experiments/p1_erasure_pauli/crosscheck_ml.py`).

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

   Stratification gains about 20-50x at p = 0.02 and only 2-6x at p >= 0.04. Combining with the measured ML cost per decode (and assuming the ML strata profile is similar to MWPM's, to be re-measured): with the exact decoder on one core, d = 11 takes at most about 30 minutes per target; d = 13 takes about 1.4 h at p = 0.04 and 18 h at p = 0.02; d = 15 takes about 2 h at p = 0.06 (extrapolated) and a day at p = 0.04. (Multiply the decode counts by 4 for 5% relative error.) So **the exact decoder covers d <= 11 at every work point, d = 13 for p >= 0.04, and little beyond; p = 0.02 at d >= 13 and p = 0.04 at d = 15 need the truncated MPS decoder** (or the Bravyi-Vargo rare-event MCMC, PRA 88, 062308). MWPM is not limited in this way.

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
│   │   ├── tn_ml.py        # exact transfer-matrix ML decoder with per-qubit priors (done; d <= ~11-13); truncated MPS (to do)
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

Theory track: `make -C theory check` needs only numpy; `make -C theory pdf` needs a minimal TeX Live: `sudo apt-get install -y --no-install-recommends texlive-latex-recommended latexmk` (about 120 MB; plain `pdflatex`, no CJK fonts).

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
| M1 | ML decoding (code capacity, with erasure priors) | Depolarizing ML threshold about 18.9%, trend consistent with MWPM. Status: **exact transfer-matrix decoder done and validated** (oracle, qecsim cross-check, thresholds), usable to d ~ 11-13; truncated MPS for d = 13, 15 and the envelope identity at d >= 9 / for truncated MPS are to do (the identity holds at d = 5, 7 with exact ML) |
| M2 | Full P1 result | R_{e->p} and the ML-MWPM margin at at least 3 work points, with errors |
| M3 | Burst-event injection and Poisson baseline | The injected event rate can be estimated without bias |
| M4 | Full P2 result | Ratio of the experiment time needed by the pattern method relative to Poisson counting, with errors |
| M5 | Summary | `results/SUMMARY.md`: key numbers, figures, limitations |

The theory-track milestones run **in parallel** with the table above (from week one, without waiting for the numerics):

| Stage | Content | Interlock with the numerics / acceptance criterion |
|---|---|---|
| T0 | Monotonicity lemma (three free operations), representation of erasure/leakage/events, exact small-code checks | First draft and `theory/checks` exist (all pass); M1 uses them as the oracle (done) |
| T1 | Exchange-rate bound R <= (3/4-p0)/(1-e0); preorder theorem for the Pauli+erasure class | M2 reports R <= c and the gap Delta; P1 decides whether the gap comes from code dependence or incompleteness of the free operations |
| T2 | Counterexample search for "code-universal order = local free order"; proof of Observation 3.5 | Exact computation on a library of random stabilizer codes; no large compute needed |
| T3 | Conditional certification: code-capacity version (done) -> bursts + chip events (needs O1-O3) | M3/M4: floor scaling, parameter estimation, certificate >= exact/TN-ML failure rate |
| T4 | Information-theoretic bounds on the exchange rate; experimental-design theorem (Stein/Chernoff exponents) | M4: T_count / T_pattern against the Poisson bound ln(1/delta)/r |

## 7. Theory track and out of scope

### 7.1 The theory track: advanced in parallel in `theory/`

Theory does not wait for the numerics to finish; it starts in week one, theory leads, and the numerics test the theory (including the monotonicity lemma). The notes are in `theory/` (LaTeX, `make -C theory pdf`); the status board is in [`theory/README.md`](theory/README.md). Contents and order:

1. **Monotonicity under the three free operations** (rigorous proofs): coarse-graining and discarding side information by the data-processing inequality; superposing independent, classically samplable noise by a simulation argument; how erasure and leakage are represented in the model, under which assumptions the argument still holds and where it fails.
2. **The preorder theorem in the simplest subclass.**
3. **A necessary and sufficient condition for Pauli plus erasure noise at code capacity.** Entry point: an erased qubit carries a location flag; discarding the flag is a free operation, and after discarding it the qubit is a depolarizing error of rate 3/4. Monotonicity then gives a rigorous upper bound on the exchange rate, `R_{e->p}(p0, e0) <= (3/4 - p0)/(1 - e0)` (equal to 3/4 to first order for small p0, e0). The numerics of P1 compute the true exchange rate; given the erasure threshold of about 50% and the depolarizing ML threshold of about 18.9%, the true rate should be far below the bound. **This gap is itself a research question**: it measures how much of the value of erasure the "discard the flag" conversion leaves out, that is, which finer monotones the preorder still needs.
4. **Conditional certification theorem.** In the model class with local burst events and chip-scale events, give finite-sample upper bounds directly on p_L(d); the numerics of P2 provide the tests.
5. **Information-theoretic upper and lower bounds on the exchange rate, and the experimental-design theorem**: using channel divergences and Stein and Chernoff exponents.

Current progress and what is unfinished (which parts have a complete written proof and which only have statements and routes) is in `theory/README.md`.

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
