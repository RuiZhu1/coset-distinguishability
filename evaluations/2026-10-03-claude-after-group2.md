# Evaluation after groups 1 and 2, 2026-10-03

| | |
|---|---|
| Evaluator | Claude Opus 5.5, two fresh-context subagents |
| Runs | Two, no web access |
| Materials | The protocol bundle (ABSTRACT, README, results/README, theory/main.pdf at 69 pages, EVALUATION.md §1) |
| Tree | 0008094 |

**Result: I = 5.4 / 5.5 (mean 5.5), A = 5.9 / 6.3 (mean 6.1).** The baseline was I 4.3, A 5.9.

## Sub-scores (run 1 / run 2)

| | I-a | I-b | I-c | I-d | I-e | I-f |
|---|---|---|---|---|---|---|
| baseline | 5.5 | 4.5 | 3.0 | 4.0 | 5.5 | 3.5 |
| now | 6.0 / 6.5 | 5.5 / 5.5 | 6.5 / 6.5 | 4.0 / 4.0 | 6.5 / 6.5 | 3.5 / 3.5 |

| | A-a | A-b | A-c | A-d | A-e | A-f |
|---|---|---|---|---|---|---|
| baseline | 5.0 | 5.5 | 6.5 | 6.0 | 7.5 | 6.0 |
| now | 5.5 / 5.5 | 5.5 / 6.0 | 6.5 / 6.5 | 6.0 / 6.5 | 7.0 / 8.0 | 5.0 / 6.0 |

## Weaknesses named by both runs

1. **The bounds are loose and miss the hardware regime.**
   - Thm 4.28 is 39–240× the simulated rate; Thm 4.32 is 9–62×.
   - The bound for every d holds only for p < 1.4e-3, and depends on (H1)–(H2) beyond d = 15.
   - Willow d ≥ 5 is not certifiable: s* = 0.73 and 0.55.
   - No certificate is computed from the data's own statistics.
2. **The resource-theory core is known, and the framing is not used by the most useful results.** Prior work that is still uncited:
   - Wang–Wilde (PRR 2019), resource theory of asymmetric distinguishability;
   - Takagi–Regula (PRX 2019);
   - Gour, *Comparison of quantum channels by superchannels* (IEEE TIT 2019);
   - Buscemi–Gour, relative Lorenz curves (PRA 2017);
   - Rosset–Buscemi–Liang (PRX 2018);
   - Gour–Scandolo, dynamical resource theories;
   - Pryadko (Quantum 2020), ML decoding with circuit-level errors;
   - Aliferis–Gottesman–Preskill;
   - Spitz et al. 2018, estimating a DEM from syndromes.
3. **There is no review; the hard questions are open; scope and numerics are narrow.**
   - Nothing is independently reviewed, and the computer-assisted steps use floating point.
   - Conj 3.11 and H_B at e = 0 are open.
   - Scope is the surface code only.
   - The ML fits use d = 5–11 only, and Δα has a systematic error from the fit window.
   - The planted-noise stim numerics are not started.

## Route to 7.5 (consensus of both runs)

### Axis I

| # | Deliverable | Score effect |
|---|---|---|
| 1 | Certificates from the data's own statistics: confidence limits on the mechanism probabilities, pushed through Thm 4.32 | I-c to about 8 |
| 2 | Tighter bound: exact short cycles plus a walk-sum tail; Y-correlated factor B instead of B_M; a d-independent Thm 4.32; target s* ≥ 1 on Willow d = 5 | I-d +2–2.5, I-b +1 |
| 3 | Tool: several observables, PyPI, notebook, sinter hook | I-e to 8 |
| 4 | Circuit-level decoder margin and erasure exchange rate; P2 on surface-code circuits | I-a and I-b +1 |
| 5 | Breadth: colour code, heavy-hex, bivariate-bicycle qLDPC, neutral-atom erasure circuit | I-f +2–2.5 |

Items 1–4 give about 7.0–7.4; adding item 5 gives about 7.8.

### Axis A

| # | Deliverable | Score effect |
|---|---|---|
| 1 | Positioning against the missing literature, and a channel-level (superchannel) formulation | A-a +0.5–1, A-d +1–1.5 |
| 2 | Two arXiv papers: (i) circuit-level Peierls bounds, hardware margins and the tool; (ii) erasure–Pauli exchange rates | A-f to 7.5–8 |
| 3 | Independent verification: interval arithmetic, (H1)–(H2) for all d | A-c to 8 |
| 4 | One hard theorem: Conj 3.11, or H_B at e = 0 | A-b +1.5–2 |

Items 1–3 give about 7.1; adding item 4 gives 7.5–7.8.
