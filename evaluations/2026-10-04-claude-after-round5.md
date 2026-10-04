# Evaluation after rounds 3-5 (literature, data certificate, Thms 4.34 and 4.35), 2026-10-04

- **Evaluator:** Claude Opus 5.5, two fresh-context subagents, no web.
- **Bundle:** as in the protocol; main.pdf has 76 pages.
- **Tree:** d10546a.

**Result: I = 5.5 / 5.4 (mean 5.5), A = 6.2 / 5.8 (mean 6.0).** The previous evaluation (0008094) gave I 5.5, A 6.1; the baseline was I 4.3, A 5.9.

## Sub-scores (run A / run B)

| | I-a | I-b | I-c | I-d | I-e | I-f |
|---|---|---|---|---|---|---|
| previous | 6.0 / 6.5 | 5.5 / 5.5 | 6.5 / 6.5 | 4.0 / 4.0 | 6.5 / 6.5 | 3.5 / 3.5 |
| now | 6.0 / 6.0 | 5.5 / 5.0 | 7.0 / 6.5 | 4.0 / 4.5 | 6.5 / 6.5 | 3.5 / 3.5 |

| | A-a | A-b | A-c | A-d | A-e | A-f |
|---|---|---|---|---|---|---|
| previous | 5.5 / 5.5 | 5.5 / 6.0 | 6.5 / 6.5 | 6.0 / 6.5 | 7.0 / 8.0 | 5.0 / 6.0 |
| now | 5.5 / 5.5 | 6.0 / 5.5 | 6.5 / 6.0 | 6.5 / 6.0 | 7.5 / 7.0 | 6.0 / 5.0 |

## Reading

The tightening rounds (Thms 4.34, 4.35) and the data certificate (Prop 4.33) did not move I-d. Both reviewers judge the bounds by whether they say something about real hardware, and they still do not:
- on Willow at d = 5 the bound is finite but its value is above 1;
- Willow d = 7 is not certifiable;
- the data certificate is vacuous for multi-round runs;
- Lambda* is certified for the stim uniform-noise model, not for hardware.

The work did raise I-c a little: the reviewers credit testing the certificate's own assumptions on hardware data and reporting where they fail.

Literature still missing: Wagner, Kampermann, Bruss and Kliesch (Pauli noise from syndrome statistics; closest to Lemma 4.4 and Prop 4.33), Bombin et al. 2012 (18.9% threshold), belief-matching and correlated decoders, and the 2024 cosmic-ray papers.

## Route to 7.5 (consensus)

### Axis I

| # | Deliverable | Effect |
|---|---|---|
| 1 | Data certificate v2: joint (a, T) regions, a boundary statistic, hyperedge (G), time stationarity (T); a non-vacuous certificate on Willow d = 5 | |
| 2 | Circuit-level decoder-headroom study against correlated, belief and ML-approximate decoders | |
| 3 | Tool release: PyPI, multiple observables, notebook | I-e to 8 |
| 4 | Bulk/boundary envelopes; (H1)–(H2) proved; interval arithmetic | |
| 5 | Breadth: colour code, erasure circuits, qLDPC | I-f |

### Axis A

| # | Deliverable | Effect |
|---|---|---|
| 1 | Two arXiv papers | A-f to 7.5–8 |
| 2 | Interval arithmetic plus independent review | A-c to 8 |
| 3 | One hard theorem: Conj 3.11 or H_B at e = 0 | Needed for A ≥ 7.5–8 |

Run B's estimate: with A1, A2 and I1, I3, I4 done, both axes reach about 7.3–7.5 without solving an open problem.
