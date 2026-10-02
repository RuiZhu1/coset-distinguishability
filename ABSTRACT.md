# Distinguishability of Logical Cosets as a Resource: Preorder, Local Exchange Rates, and Finite-Data Extrapolation Certificates for Surface-Code Noise

**Author**: [Name], Department of Mathematics, Technion – Israel Institute of Technology, Gilad Gour group
**Status**: Research plan (draft abstract), annotated with the status of every claim as of 2026-10-02.

The abstract text is the original plan, unchanged. Each paragraph is followed by a **Status** note that says what is proved,
what is only numerical, and what is open. Theorem numbers refer to `theory/main.pdf` (built from `theory/`; the status board is
`theory/README.md`); data are in `results/`.

| Tag | Meaning |
|---|---|
| **PROVED (draft)** | A complete written argument is in `theory/`. It is a first draft and has **not been independently reviewed**. |
| **NUMERICAL** | Checked by exact enumeration or Monte Carlo in this repository; no proof. |
| **PARTIAL** | Part of the claim is proved or done; the rest is stated explicitly. |
| **OPEN** | An explicit open problem or conjecture (evidence is noted where there is some). |
| **NOT STARTED** | Nothing in the repository yet. |
| **CITED** | Taken from the literature; not re-derived and not used by any proof here. |

---

## Abstract

Superconducting surface-code experiments have measured an error-suppression factor Λ ≈ 2 at distances d = 3, 5, 7. Whether this suppression persists to large distance, and how much damage different noise types (leakage, erasure, biased noise, correlated burst events) each do to error correction, is currently judged mainly by numerical fits and untested noise assumptions; there is no unified and rigorous framework for comparing them. This project builds the foundations for that question in the setting of quantum resource theories.

We fix the code family (the surface code) and the standard syndrome-extraction circuit, and use the following fact: under maximum-likelihood (coset) decoding, the logical failure probability is exactly the Bayes error of the hypothesis-testing problem of distinguishing the logical cosets given the syndrome. Accordingly, we take the **distinguishability of logical cosets** as the resource, and define syndrome coarse-graining, discarding classical side information, and superposing independent, classically samplable noise as the free operations. Monotonicity under the first two follows directly from the data-processing inequality; for the third it follows from a simulation argument: the decoder for the old problem can itself sample the additional noise E′, call the optimal decoder of the new problem, and then use the known E′ to convert the coset label back to the original problem. The error-suppression exponent α is defined as the large-deviation error exponent of this distinguishing problem; by the statistical-mechanics mapping of Dennis–Kitaev–Landahl–Preskill and Chubb–Flammia, α equals the disorder-averaged free-energy tension of a domain wall in the corresponding random-coupling model.

> **Status.**
> - ML failure probability = Bayes error of coset distinguishing: **PROVED (draft)**, Lemma 1.4 (for any linear decoding structure, which covers code capacity and detector error models).
> - Free operations and monotonicity (data processing; simulation argument): **PROVED (draft)**, Theorem 1.7, Proposition 1.9, Corollary 1.10. Two limits that the plan did not state: independence of the superposed noise cannot be dropped (Remark 1.11), and the statements hold for Bayes-optimal decoding only, not for MWPM (Remark 1.12).
> - Resource-theory audit (new, section 0): the framework is checked against the axioms of a resource theory and the missing structure is supplied (category of free operations, free objects and golden rule, product, a complete family of monotones, Theorem 0.6). **PROVED (draft).** The axiom list used is the author's assumption and must be confirmed or replaced by the group's reference formulation.
> - α as a large-deviation exponent: defined (section 1). The equality with the domain-wall free-energy tension is **CITED** (Dennis–Kitaev–Landahl–Preskill, Chubb–Flammia); no proof or computation here uses it.

We consider an explicit class of noise models 𝒩(p, e, r, R, r_c, τ): a local random-Pauli background at rate p, erasure at rate e, local burst events at rate r with radius at most R, chip-scale events at rate r_c, and leakage with lifetime at most τ. For d ≫ R, local burst events only change the suppression exponent, whereas chip-scale events produce an error floor proportional to the running time. Within this framework we will obtain four classes of results.

> **Status.**
> - The class is defined in section 2 (additive event layers; leakage under the classical-samplability assumption (L)). Monotonicity in the rate directions p, e, r, r_c: **PROVED (draft)**, Propositions 2.7 and 2.10. Monotonicity in the shape parameters (R, T_b, p_b) of unflagged bursts is **not** given by the free operations: **OPEN** (O1, Remark 2.11).
> - Chip-scale events give a floor ∝ r_c T independent of d: **PROVED (draft)**, Lemma 4.6 (upper bound without extra assumptions, lower bound q_c(1 − 2^(−q)) under assumption (K)); the stim test is **NOT STARTED** (P2).
> - "For d ≫ R bursts only change the exponent": **OPEN** (O3); neither proved nor tested.

First, a **noise-preorder theorem** (the core result of the resource theory): within 𝒩, sufficient conditions for "noise 𝒩 is no worse than ℳ for error correction", together with necessary and sufficient conditions in tractable subclasses (such as Pauli plus erasure noise at the code-capacity level). This is a convertibility criterion, of the same type as the conversion conditions in entanglement theory.

> **Status: PARTIAL.**
> - Sufficient conditions (reachable by free operations ⇒ ML failure rate ordered): **PROVED (draft)**, Theorem 1.7; this is all the certificates need.
> - Pauli plus erasure at code capacity: free reachability is characterised exactly by two scalars, p′ ≥ p and (1−e′)(1−4p′/3) ≤ (1−e)(1−4p/3) (Theorem 3.7, Proposition 3.15): **PROVED (draft)**. For the uncoded qubit the same order equals reachability by arbitrary simulable reductions and the order of all Bayes-risk monotones (Proposition 0.9): **PROVED (draft)**.
> - The converse for the decoder failure rate ε\* alone (every pair not connected by free operations is reversed by some linear decoding structure): proved on the part of the plane with μ′ > μ or ν′ > ν (Proposition 3.16, including every unreachable pair with e′ < e); on the remaining "trade" pairs (less Pauli noise, more erasure) an exhaustive search over all linear structures on up to 3 qubits and hill-climbing at 4 and 5 qubits finds **no** reversal, and the channel-coding exponent criterion for large non-degenerate structures is not met either (Remark 3.25). Structural obstructions to erasure-hardened structures are proved (Lemmas 3.19–3.21, Corollary 3.22). This is **numerical evidence that this form is false**, not a proof (degenerate structures at large n are not covered): **OPEN**, Conjecture 3.11, Observation 3.17. The all-losses form is proved, so no further free operation is needed.
> - "Convertibility criterion of the same type as in entanglement theory": a complete family of monotones is proved at the level of label–observation pairs (Theorem 0.6, **PROVED (draft)**); rates in the tensor-power sense (n independent blocks) are a different quantity and are **NOT STARTED**.

Second, **local noise exchange rates**: from the sensitivity of the domain-wall tension to the noise parameters we derive equivalences between different noise types near a given operating point, and bound them above and below using channel divergences. The rates explicitly depend on the operating point and on the decoder; we make no globally universal claim.

> **Status: PARTIAL.**
> - The quantity computed is the marginal rate R = (∂α/∂e)/(∂α/∂p) of the exponent (both derivatives are negative, R > 0). The domain-wall-tension representation is not used.
> - Upper bound R ≤ c = (3/4 − p₀)/(1 − e₀) (not simply 3/4: c > 3/4 when p₀ < (3/4)e₀): **PROVED (draft)**, Theorem 3.2. It is the sharp bound for the marginal rate of every monotone of the free order (Proposition 0.11). The [[5,1,3]] code attains it at e₀ = 0: **NUMERICAL**, Observation 3.5.
> - The marginal rate of the surface code itself (P1, M2, Bayes-optimal decoder at code capacity, d = 5–11 at p₀ ∈ {0.04, 0.06}, e₀ ∈ {0, 0.02, 0.05}): **NUMERICAL** (`results/p1_exchange_rate.json`, theory Observation 3.26). R^(d) ≤ c in every entry (Theorem 3.2 is never violated, R^(d)/c ≈ 0.3–0.6); at e₀ > 0 the exponent rate is R_α = 0.20–0.24 (± 0.03–0.04), i.e. R_α/c ≈ 0.3, within about one standard error of the Bhattacharyya reference R_B. The e₀ = 0 points and d = 13 are statistically weak; p₀ = 0.02 was not run.
> - Lower bounds "using channel divergences": α is sandwiched by −Φ(B) ≤ α ≤ ln(1/B) in the Bhattacharyya parameter B (Proposition 5.2 **PROVED (draft)**, Proposition 5.1 SKETCH), but a sandwich does not bound derivatives; the reference rates R_B and R_hash are **heuristics**, not theorems. A universal lower bound on marginal rates over decoding structures is **OPEN** and cannot be uniform in p₀: the infimum over structures tends to 0 as p₀ → 0 at fixed e₀ (**PROVED (draft)**, Proposition 3.23), while at p₀ = 0.0213 the smallest rate found is positive (exhaustive up to 3 qubits, hill-climbing up to 6; **NUMERICAL**, Observations 3.17, 3.24).

Third, **decoder margins**: any practical decoder is a further processing of the syndrome, so its suppression exponent cannot exceed the maximum-likelihood exponent. We will give computable estimates of the gap between the two, to judge how much room remains for decoder improvements.

> **Status: PARTIAL.** "No decoder beats ML": **PROVED (draft)** (Bayes optimality, Remark 1.12). Computable estimates of the gap: **NOT STARTED** with erasures (MWPM with erasure is not implemented); the ML side exists (exact transfer-matrix and truncated-MPS decoders, validated against exact enumeration and an independent implementation) and MWPM without erasure exists.

Fourth, **conditional extrapolation certificates and experimental design**: within 𝒩, we give finite-sample upper bounds on the large-distance logical error rate p_L(d) directly from finite-distance data, rather than first estimating α and then extrapolating, thereby accounting for finite-size corrections; the error floor depends explicitly on r_c. With the Poisson benchmark ln(1/δ)/r as reference, we do not try to beat the detection limit for unobserved events; instead we exploit the differences in the space-time patterns that different event classes leave on the detectors, characterize the optimal rates for distinguishing event classes and estimating model parameters by Stein and Chernoff exponents, and use them to optimize code distance, number of rounds, and adaptive strategies.

> **Status: PARTIAL.**
> - Code-capacity certificate: **PROVED (draft)**, Theorem 4.5. It is computed from flag and syndrome calibration data and holds for all d at once; it does **not** use finite-distance p_L(d) data as the plan's wording suggests. It rests on the union bound and is loose (its threshold lies far below the ML threshold), so at realistic parameters it may be vacuous (O7). The coverage simulation is **NOT STARTED**.
> - With bursts and chip-scale events: **PLAN** (Theorem 4.8; needs O1–O3).
> - Floor depends explicitly on r_c: **PROVED (draft)**, Lemma 4.6.
> - Poisson benchmark ln((1−a)/δ)/r: **PROVED (draft)**, Proposition 5.5.
> - Stein/Chernoff characterisation: the chain D_count ≤ D_pattern ≤ D_event is **PROVED (draft)** (Proposition 5.6); the asymptotic ratio of rounds is a SKETCH. Optimising distance, rounds and adaptive strategies (experimental-design theorem): **OPEN** (O4).

The numerics are organized in layers: under code-capacity and phenomenological noise we use tensor-network maximum-likelihood decoding for accurate computation; circuit-level noise corresponds to a three-dimensional statistical-mechanics model, for which we use only MCMC or approximate tensor networks, commit to small and medium code distances, and label conclusions as numerical estimates. In stim simulations we will plant three families of models — superconducting leakage and burst events, neutral-atom erasure, and biased noise — to test the accuracy of the preorder, the exchange rates, and the bounds; we will apply the methods to public superconducting surface-code experimental data, and state explicitly the assumptions that cannot be tested for lack of long-running or raw-readout data; and we will release all methods as an open-source toolkit that plugs directly into stim/sinter workflows.

> **Status: PARTIAL.**
> - Code-capacity tensor-network ML decoding: **DONE and validated** (exact transfer-matrix decoder to d ≈ 11; truncated-MPS decoder, validated against the exact decoder at d = 11, 13, with and without erasures; weight-stratified estimation with analytic derivatives; sanity checks against known thresholds, one of which — bit-flip MWPM — is not closed).
> - Phenomenological noise; circuit-level MCMC or approximate tensor networks: **NOT STARTED**.
> - stim simulations with planted leakage/burst, erasure and biased noise (P2): **NOT STARTED**.
> - Public superconducting data: **NOT STARTED**.
> - Open-source toolkit: the Python package `lcd` (codes, noise, decoders, stratified estimator, exchange-rate analysis) exists; the stim/sinter integration is **NOT STARTED**.

---

## Status at a glance

| Claim | Status | Where |
|---|---|---|
| ML failure probability = Bayes error | PROVED (draft) | theory 1.4 |
| Monotonicity under the three free operations | PROVED (draft) | theory 1.7, 1.9 |
| Resource-theory structure; complete family of monotones | PROVED (draft); axiom list to be confirmed | theory §0 |
| α = domain-wall tension | CITED | — |
| Rate monotonicity in p, e, r, r_c; floor ∝ r_c T | PROVED (draft) | theory 2.10, 4.6 |
| Shape monotonicity for unflagged bursts; "bursts only change the exponent" | OPEN | theory 4.9 (O1), 4.11 (O3) |
| Preorder: sufficient condition; free reachability for Pauli + erasure | PROVED (draft) | theory 1.7, 3.7, 3.15 |
| Preorder: converse for the failure rate ε\* alone | OPEN; numerical evidence against; obstructions to hardened structures proved | theory 3.11, 3.17, 3.19–3.25 |
| Exchange rate upper bound (3/4 − p₀)/(1 − e₀) | PROVED (draft) | theory 3.2, 0.11 |
| Surface-code exchange rate | NUMERICAL: R_α/c ≈ 0.3 at e₀ > 0, d = 5–11 | theory 3.26, `results/p1_exchange_rate.json` |
| Lower bound on exchange rates | OPEN; cannot be uniform in p₀ | theory 3.23, 3.24 |
| Decoder margin | NOT STARTED with erasures | — |
| Code-capacity certificate | PROVED (draft); loose; coverage test not done | theory 4.5 |
| Certificate with bursts and chip events; experimental design | PLAN / OPEN | theory 4.8, 5.7 |
| Code-capacity ML tensor-network decoding | DONE, validated | README |
| Circuit-level and stim numerics; public data; stim/sinter toolkit | NOT STARTED | — |

---

## Scope and limitations

- **Coherent errors are outside the model class.** The framework targets random Pauli noise plus prior information (erasure flags, leakage), and relies on a twirling approximation or on coherent errors being negligible; this assumption can itself be tested experimentally and is listed as one of the model-class assumptions.
- **The code family is restricted to the surface code.** Extension to qLDPC codes is a later direction.
- **Circuit-level maximum-likelihood exponents can only be computed approximately**, which limits the accuracy of decoder margins and exchange rates under circuit-level noise.
- **Limits of public data.** Long continuous runs, raw readout, leakage dynamics and similar information are often not public, so certificates based on public data have limited strength; their main value is as a methodological demonstration.

## Keywords

quantum resource theory; quantum error correction; surface code; maximum-likelihood decoding; statistical-mechanics mapping; hypothesis testing; noise preorder; extrapolation certificates; experimental design
