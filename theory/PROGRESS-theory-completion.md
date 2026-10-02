# Theory completion: progress notes (branch `theory-completion`, paused 2026-10-02)

Working tree: `../coset-theory` (git worktree), kept separate from the main tree, where another agent is doing F0–F7.

**Update (tex written):** everything below is now in the notes: sec3 §3.x (Lemmas 3.28–3.29, Theorems 3.30–3.31, Prop 3.32), sec4 (Lemma 4.8, Def 4.9, Thm 4.10, Prop 4.11, Lemma 4.12, Prop 4.13, Thm 4.14, Thm 4.15, Cor 4.16), sec5 (Prop 5.1 PROVED), sec6 (new table rows, open-problem list), main.tex abstract, README status board. The PDF builds with no undefined references. K4 and K5 have now run and pass (each about 1 s, 140 MB). K6 and K7 are still not run, and a full run of the script was OOM-killed once (cause not located). The MWPM numerical check is not written.

## Results derived (proof worked out) and numerically checked

1. **Prop 5.1 (genie lower bound): SKETCH -> PROVED.**
   - Step 1 for every odd d: for each row r there is exactly one Z-type bulk plaquette meeting an interior column in rows {r, r+1}, which forces x to be constant; the same holds for X-type plaquettes, so z is constant; the top X boundary check meets the column in one qubit, which forces z = 0. Check K1 (d = 3..41, every interior column): passes.
   - Step 3 is non-asymptotic: tilt R = sqrt(PQ)/B^d. By symmetry, E_R[LLR] = 0. Jensen's inequality then gives
     eps_genie(d) >= 1/2 B^d exp(-sqrt(d sigma^2)/2), where sigma^2 = 2(1-e) sqrt((1-p)p/3) L^2 / B and L = ln(3(1-p)/p). This removes the citation of Chernoff's theorem. Check K2: passes.
2. **O3 solved (tilted union bound).** For any full-support Q: BC(P;G) <= E_P[sqrt(Q(X+G)/Q(X))] (AM-GM, using G+G = 0).
   - With Q = N_cc(t, e) and independent regions (activation probability rho_z, size <= V, coverage density rho_bar = max_j sum_{z containing j} rho_z, covered rates in [p, 3/4]), this gives
     BC(P;G) <= B_burst^wt(G), where B_burst = m0 exp(rho_bar (kappa^V - 1)/V), m(s) = e + (1-e)[(1-s)sqrt(tau) + s/(3 sqrt(tau)) + 2s/3], m0 = m(p), kappa = m(3/4)/m0 and tau = t/(3(1-t)).
   - Hence eps* <= 1/2 W~(B_burst): a single sum with no d-independent floor. The bound holds uniformly in the burst strength.
   - Check K3 ([[5,1,3]] with erasure, d = 3 surface code): passes. It takes 148 s, so the number of cases should be reduced.
3. **Theorem 4.8: PLAN -> PROVED (code capacity).** Chain: rate monotonicity (Prop 2.10, Poisson model) -> floor upper bound (Lemma 4.6) -> item 2 at the upper limits. The result is uniform over shapes, so O1 is not needed.
4. **O1 answered in the negative.** On the d = 3 rotated surface code, with one region {1,3,5,7}, p = 0.005 and rho = 0.05, eps* peaks at p_b ≈ 0.5 and falls by about 26% at p_b = 3/4 (float search). A second example: n = 3, S = <X2 Z0>, N = <X2 Z0, Z0 Z1>, regions {0,2} and {1}, p = 0.01, rho = 0.05: eps* is 0.01470 at p_b = 0.7 and 0.01409 at 0.75.
   - Check K4 (exact rationals at p = 1/200, rho = 1/20, p_b = 1/2 vs 3/4) is written but **has not been run yet**.
5. **O2 partial.**
   - (a) The flag estimate of e is unchanged.
   - (b) Lemma 4.4 stays valid (conservative) under any independent additive contamination: E[(-1)^bit] = lambda^w0 * chi with |chi| <= 1. Shots must be i.i.d.; for multi-round data, use windows separated by gaps of at least T_b.
   - (c) Upper limits on r need a lower bound eta on the detection efficiency: 1 - e^{-r|W|} <= P_bar_T / eta.
   - Impossibility without a floor on burst strength: as p_b -> 0 the model with large r is TV-close to r = 0, so any uniformly valid upper limit must be infinite.
   - Check K7 for (b) is written but not run.
6. **O5 solved for code-capacity MWPM** (X and Z decoded separately, uniform weights, erased edges given weight 0).
   - If decoding fails, E xor C contains a simple cycle that has odd overlap with row 0. Every top-row qubit lies in exactly one Z check (K1), so the cycle passes through the boundary vertex and runs from top to bottom, hence has length >= d.
   - Minimality of the matching gives w(gamma ∩ C) <= w(gamma ∩ E). A Chernoff bound per edge gives B_M = e + (1-e) 2 sqrt(q(1-q)), with q = 2p/3.
   - The number of such cycles of length l is at most d 3^(l-1). Therefore p_L^MWPM <= 2 d B_M (3 B_M)^(d-1) / (1 - 3 B_M).
   - The bound is increasing in p and e, which gives a finite-sample MWPM certificate (Lemmas 4.3 and 4.4) without needing MWPM to be monotone.
   - Numerical check vs MWPM simulation (needs pymatching): not written yet.
7. **O6 / Obs 3.5 / Remark 3.13.**
   - eps*_A is concave in one qubit's rate, and a qubit at rate 3/4 is the same as a flagged qubit.
   - Danskin: the right derivative equals the minimum over the active decision rules (elementary, since there are finitely many rules). This proves the envelope argument; the derivative is exact when the ML decision is unique.
   - Consequence: d eps/de <= c * (right d eps/dp) holds at every point, with no differentiability assumption.
   - Equality R = c holds iff, for every A with Pr[A] > 0 and every j not in A, some decision that is ML-optimal for A stays optimal for A ∪ {j}, plus a Danskin condition that is automatic when the ML decision is unique.
   - [[5,1,3]]: the condition holds for all 0 < p < 3/4 (float scan p ≤ 0.74). Check K5 is an exact certificate on integer polynomials ((1+x)^m P(1/(1+x)) has nonnegative coefficients) but **has not been run yet**. Steane and the d = 3 surface code violate the condition at every p.

## Remaining open (not attempted)
O4, O7, R-c (tensor-power rates), Conjecture 3.11 for n >= 6, H_B, the gap problem. The literature item in section 0 is for the author.

## Next steps
1. Profile K4–K7 under `ulimit -v 3000000`: some part of `completion_checks.py` (probably K4, K5 or K6) used about 5 GB and was OOM-killed. Reduce K3 cases.
2. Write `theory/checks/mwpm_peierls.py` (MWPM vs bound, e = 0; e > 0 after F3).
3. Write the tex:
   - sec3: new subsection at the end of the file, and the status tag of Remark 3.13;
   - sec4: tilted lemma, burst theorem, Theorem 4.8, O1 proposition, O2 lemma and impossibility result, MWPM section;
   - sec5: Prop 5.1;
   - sec6: open-problem list and table.
   Then update the README status board and the Makefile, build the PDF, and merge with main after the other agent's F-tasks are committed (expect small conflicts in theory/README.md and sec6-map.tex).

## Pen-and-paper pass (2026-10-02, after the merge; no code run)

Written into the tex (numbers as in the status board of `theory/README.md`, counted by hand, PDF not rebuilt):
- **Prop 0.12**: H(L|Y) is in the complete family, so Q_hash is a proved monotone (was N7 only); Φ_α multiplicative and H(L|Y) additive give necessary conditions for tensor-power conversions (R-c). Sufficiency open.
- **§3.9 (Props 3.33–3.37)**: (E1) in polynomial/lexicographic form; split-logical lemma (minimum weights never decide (E1)); first-order obstruction; **[[5,1,3]] by hand** (closed-form class polynomials, all factorizations (1−r)^3·(...); with one extra flag at a single-error syndrome the four classes tie exactly); **gap problem: code dependence** (no code-universally sound enlargement lowers c). Heuristic small-B support for H_B (remark only).
- **§4.10–4.11 (Def 4.17 – Prop 4.22)**: space-time burst bound for any linear decoding structure (DEM locations, K = |V_ℓ|), one-detector parameter limits, circuit-level certification theorem (needs the location-weight enumerator, not computed); O7: ML ≤ MWPM-Peierls closed form for every d, SAW refinement (c_10, c_20 cited; range 4.29% → ~4.9%, hand-computed).
- **§5.5 (Lemma 5.8 – Thm 5.11)**: marked-Poisson model; D_pattern = D_count + Λ1 D(π1‖π0) (weak signal: 1 + χ²); Stein form of the round ratio (Prop 5.6 no longer SKETCH); adaptive designs do not beat the best single experiment (weak converse).
- Stale text updated: sec0 (Q_hash, R-c, verdict), sec3 (Obs 3.5 tag, Remark 3.27), sec4 (intro, circuit-level remarks, SAW remark), sec6 (table, open problems, T0–T4), main.tex abstract, theory/README status board, README §6 T-rows, ABSTRACT.

Still open after this pass: H_B (quantitative gap), single-loss Conjecture 3.11 (n ≥ 6), R-c sufficiency, O4 composite hypotheses / adaptive strong converse, O7 beyond ~5%, O6 families (perfect-tensor codes?), circuit-level enumerator.
Checks to write (none run): K6, K7 run; class polynomials of Prop 3.36; c_K enumeration; small-DEM check of Thm 4.18; marked-Poisson Monte Carlo; mwpm_peierls.py. Rebuild the PDF and re-verify the numbering against main.aux.
