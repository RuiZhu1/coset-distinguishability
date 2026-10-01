# Distinguishability of Logical Cosets as a Resource: Preorder, Local Exchange Rates, and Finite-Data Extrapolation Certificates for Surface-Code Noise

**Author**: [Name], Department of Mathematics, Technion – Israel Institute of Technology, Gilad Gour group
**Status**: Research plan (draft abstract)

---

## Abstract

Superconducting surface-code experiments have measured an error-suppression factor Λ ≈ 2 at distances d = 3, 5, 7. Whether this suppression persists to large distance, and how much damage different noise types (leakage, erasure, biased noise, correlated burst events) each do to error correction, is currently judged mainly by numerical fits and untested noise assumptions; there is no unified and rigorous framework for comparing them. This project builds the foundations for that question in the setting of quantum resource theories.

We fix the code family (the surface code) and the standard syndrome-extraction circuit, and use the following fact: under maximum-likelihood (coset) decoding, the logical failure probability is exactly the Bayes error of the hypothesis-testing problem of distinguishing the logical cosets given the syndrome. Accordingly, we take the **distinguishability of logical cosets** as the resource, and define syndrome coarse-graining, discarding classical side information, and superposing independent, classically samplable noise as the free operations. Monotonicity under the first two follows directly from the data-processing inequality; for the third it follows from a simulation argument: the decoder for the old problem can itself sample the additional noise E′, call the optimal decoder of the new problem, and then use the known E′ to convert the coset label back to the original problem. The error-suppression exponent α is defined as the large-deviation error exponent of this distinguishing problem; by the statistical-mechanics mapping of Dennis–Kitaev–Landahl–Preskill and Chubb–Flammia, α equals the disorder-averaged free-energy tension of a domain wall in the corresponding random-coupling model.

We consider an explicit class of noise models 𝒩(p, e, r, R, r_c, τ): a local random-Pauli background at rate p, erasure at rate e, local burst events at rate r with radius at most R, chip-scale events at rate r_c, and leakage with lifetime at most τ. For d ≫ R, local burst events only change the suppression exponent, whereas chip-scale events produce an error floor proportional to the running time. Within this framework we will obtain four classes of results.

First, a **noise-preorder theorem** (the core result of the resource theory): within 𝒩, sufficient conditions for "noise 𝒩 is no worse than ℳ for error correction", together with necessary and sufficient conditions in tractable subclasses (such as Pauli plus erasure noise at the code-capacity level). This is a convertibility criterion, of the same type as the conversion conditions in entanglement theory.

Second, **local noise exchange rates**: from the sensitivity of the domain-wall tension to the noise parameters we derive equivalences between different noise types near a given operating point, and bound them above and below using channel divergences. The rates explicitly depend on the operating point and on the decoder; we make no globally universal claim.

Third, **decoder margins**: any practical decoder is a further processing of the syndrome, so its suppression exponent cannot exceed the maximum-likelihood exponent. We will give computable estimates of the gap between the two, to judge how much room remains for decoder improvements.

Fourth, **conditional extrapolation certificates and experimental design**: within 𝒩, we give finite-sample upper bounds on the large-distance logical error rate p_L(d) directly from finite-distance data, rather than first estimating α and then extrapolating, thereby accounting for finite-size corrections; the error floor depends explicitly on r_c. With the Poisson benchmark ln(1/δ)/r as reference, we do not try to beat the detection limit for unobserved events; instead we exploit the differences in the space-time patterns that different event classes leave on the detectors, characterize the optimal rates for distinguishing event classes and estimating model parameters by Stein and Chernoff exponents, and use them to optimize code distance, number of rounds, and adaptive strategies.

The numerics are organized in layers: under code-capacity and phenomenological noise we use tensor-network maximum-likelihood decoding for accurate computation; circuit-level noise corresponds to a three-dimensional statistical-mechanics model, for which we use only MCMC or approximate tensor networks, commit to small and medium code distances, and label conclusions as numerical estimates. In stim simulations we will plant three families of models — superconducting leakage and burst events, neutral-atom erasure, and biased noise — to test the accuracy of the preorder, the exchange rates, and the bounds; we will apply the methods to public superconducting surface-code experimental data, and state explicitly the assumptions that cannot be tested for lack of long-running or raw-readout data; and we will release all methods as an open-source toolkit that plugs directly into stim/sinter workflows.

---

## Scope and limitations

- **Coherent errors are outside the model class.** The framework targets random Pauli noise plus prior information (erasure flags, leakage), and relies on a twirling approximation or on coherent errors being negligible; this assumption can itself be tested experimentally and is listed as one of the model-class assumptions.
- **The code family is restricted to the surface code.** Extension to qLDPC codes is a later direction.
- **Circuit-level maximum-likelihood exponents can only be computed approximately**, which limits the accuracy of decoder margins and exchange rates under circuit-level noise.
- **Limits of public data.** Long continuous runs, raw readout, leakage dynamics and similar information are often not public, so certificates based on public data have limited strength; their main value is as a methodological demonstration.

## Keywords

quantum resource theory; quantum error correction; surface code; maximum-likelihood decoding; statistical-mechanics mapping; hypothesis testing; noise preorder; extrapolation certificates; experimental design
