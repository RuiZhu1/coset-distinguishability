# Baseline evaluation, 2026-10-03

- Evaluator: Claude Opus 5.5, as a subagent with a fresh context.
- Run: one run, no web access.
- Materials: the protocol bundle (ABSTRACT.md, README.md, results/README.md, theory/main.pdf at 60 pages) and section 1 of EVALUATION.md.
- Tree: 02d1c4b.

**Result: I = 4.3, A = 5.9.**

## Closest prior work named by the evaluator

| Part of the project | Closest prior work | Evaluator's verdict |
|---|---|---|
| §0 (Lemma 0.2, Thm 0.6) | Gour, Grudka, Horodecki, Kłobus, Łodyga, Narasimhachar, "Conditional uncertainty principle", PRA 97, 042130 (2018); Brandsen–Geng–Gour; Blackwell–Sherman–Stein; Torgersen (1991); Le Cam; Buscemi (CMP 2012) | Conditional majorization. The notes say these are "deliberately not cited" (sec0 l. 263). |
| Lemma 1.4, Thm 1.7 | DKLP 2002; Bravyi–Suchara–Vargo 2014; Iyer–Poulin 2015 | Folklore; Thm 1.7 is data processing for H_min. |
| Peierls / no-floor results | DKLP 2002; Fowler 2012; Kovalev–Pryadko 2013; Terhal–Burkard 2005; Aharonov–Kitaev–Preskill 2006; Gottesman 2014 | New part: the explicit non-backtracking-matrix evaluation on a detector error model (DEM). |
| Chip-scale floor | McEwen et al. 2022; Xu et al. 2022; Suzuki et al. 2022 | Already established. |
| Adaptive design / strong converse | Chernoff 1959; Hayashi 2009; Nitinawarat–Atia–Veeravalli 2013 | Has precedents. |
| Erasure vs Pauli | Wu–Kolkowitz–Puri–Thompson 2022; Kubica et al. 2023; Sahay et al. 2023; Gu–Retzker–Kubica; Stace–Barrett–Doherty 2009; Delfosse–Zémor 2020 | The "2w + k ≥ d" effective-distance rule is standard. |
| Exponent / H_B | Watson–Barrett 2014; Beverland–Brown–Kastoryano–Marolf 2019 | Uncited. |

**What the evaluator judged new:**
- Thm 3.7 / Prop 3.15;
- Thm 3.31 together with Prop 3.36 and Prop 3.37;
- Prop 4.11 and Obs 4.25–4.26;
- Thm 4.24;
- Thm 4.28 and Prop 4.30;
- Thm 5.16.

## The three most serious weaknesses

1. **No contact with the hardware regime or real data.**
   - Everything central is at code capacity with p₀ = 4–6%.
   - The circuit-level bound holds only for p < 1.405×10⁻³ under uniform noise.
   - Public data, P2 and sinter are all NOT STARTED.
2. **Novelty is overstated and the positioning is missing.**
   - The resource-theory core is conditional majorization and is not cited.
   - The bibliography has 13 entries.
   - The certificates do not need the resource-theory vocabulary.
3. **Hard questions are open; outputs are loose or statistically weak.**
   - Open: Conj 3.11, H_B at e = 0, R-c for codes.
   - Looseness: the certificates are 30–100×; the circuit-level bound is 39–240×.
   - Statistics: Δα has an unquantified systematic error from the fit window.
   - Rigour gaps:
     - **Integer weights.** PyMatching minimises integer-rounded weights, while Thm 4.28 is evaluated at unrounded weights (sec4 l. 494).
     - **Floating point.** Prop 4.29 is a floating-point computation.
     - **No independent review.**

## Sub-scores

| I-a | I-b | I-c | I-d | I-e | I-f | **I** |
|---|---|---|---|---|---|---|
| 5.5 | 4.5 | 3.0 | 4.0 | 5.5 | 3.5 | **4.3** |

| A-a | A-b | A-c | A-d | A-e | A-f | **A** |
|---|---|---|---|---|---|---|
| 5.0 | 5.5 | 6.5 | 6.0 | 7.5 | 6.0 | **5.9** |

## Evaluator's route to 7.5–8

### Axis I (4.3 → about 7.5)

| # | Deliverable | Gain |
|---|---|---|
| 1 | Packaging: Apache-2.0 license, sinter wrapper, CLI, docs, PyPI | +0.30 |
| 2 | Public Google data: parameter limits, the Thm 4.28 bound on the fitted DEM, floor estimate, untestable assumptions stated | +1.05 |
| 3 | Tighter circuit bound: exact per-walk tail, factor B instead of B_M, rounded weights, SI1000 and heralded erasure; target within 10× and valid to p ≈ 3×10⁻³ | +0.78 |
| 4 | P2 planted-noise study and a circuit-level erasure break-even table | +0.50 |
| 5 | One qLDPC family (bivariate bicycle [[144,12,12]], colour code) via Kovalev–Pryadko cluster counting | +0.55 |

### Axis A (5.9 → about 7.6)

| # | Deliverable | Gain |
|---|---|---|
| 1 | Related-work section and a claim-by-claim statement of what is new | +0.45 |
| 2 | Two arXiv papers: (i) exchange rates as comparison of experiments; (ii) computable circuit-level Peierls certificates | +0.33 |
| 3 | One hard theorem: H_B at e = 0, Conj 3.11, or (H1)–(H2) for all d | +0.65 |
| 4 | Independent verification: referee by a group member, interval arithmetic, fix the integer-weight gap, optional Lean | +0.30 |
| 5 | Make the framing do work: a lower bound on the exchange rate from Φ_Λ, or a coded R-c result | +0.30 |

Deliverables 1, 2 and 4 alone (no new mathematics) reach about 6.9; a 7.5 needs deliverable 3.
