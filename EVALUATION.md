# External evaluation protocol

The project is scored on two axes, each out of 10. The targets for the finished repository are:

| Axis | Target | Baseline (2026-10-03) |
|---|---|---|
| **I. Practical value to industry research labs** (Google Quantum AI, IBM Quantum, AWS, Quantinuum, QuEra, …) | 7.5–8 | **4.3** (Claude Opus, fresh context, this protocol) |
| **A. Academic quality as a QRT + QEC theory programme** | 7.5–8 | **5.9** (same run) |

Scores from language models depend strongly on how they are asked. This file fixes the rubric, the materials and the prompt, so that
scores at different dates and from different models can be compared. **Do not change the rubric between evaluations.** If it must change,
start a new log table and re-score the baseline.

## 1. Rubric

Each axis is a weighted mean of sub-scores (0–10). The anchors describe what a score means. A sub-score above 6 needs evidence
that is already in the repository (a file, a number, a theorem). Plans do not count.

### Axis I: practical value to industry labs

| Code | Weight | Criterion | 3 | 6 | 9 |
|---|---|---|---|---|---|
| I-a | 15% | Relevance to current scaling bottlenecks (Λ at large d, leakage, bursts/cosmic rays, decoder headroom, erasure) | Adjacent questions | The right questions, answered in idealised models | Answers questions labs are actively asking, in their terms |
| I-b | 20% | Validity in the regimes labs operate in (circuit-level noise, p ≈ 1–3×10⁻³, realistic noise) | Code capacity only | Circuit level, but restricted range or idealised noise | Circuit level, covers hardware-relevant parameters |
| I-c | 20% | Validation on real experimental data | None | Synthetic data that mimics hardware | Applied to public hardware data, with stated assumptions |
| I-d | 15% | Tightness and actionability of the quantitative outputs | Correct but orders of magnitude loose | Within one order of magnitude of simulation | Tight enough to change a design or decoding decision |
| I-e | 15% | Tooling (stim/sinter integration, install, docs, tests, reproducibility) | Research scripts | Installable package, tests, examples | Drop-in tool for an existing stim/sinter pipeline |
| I-f | 15% | Breadth (beyond the surface code: qLDPC, other platforms) | Surface code only | One more family or platform | Code-agnostic in practice |

### Axis A: academic quality (QRT + QEC theory)

| Code | Weight | Criterion | 3 | 6 | 9 |
|---|---|---|---|---|---|
| A-a | 25% | Novelty against the literature (with citations) | Mostly known results | New results in a known framework | A new framework with new results, clearly placed in the literature |
| A-b | 20% | Depth of the main theorems | Direct consequences of standard tools | Non-trivial, but with simple statements | At least one hard result (a converse, an exact characterisation, a sharp exponent) |
| A-c | 20% | Rigour and verification (proofs, falsification tests, honest status tags, review) | Sketches | Complete drafts with numerical checks | Checked proofs, independently reviewed |
| A-d | 15% | Coherence of the resource-theory framing (axioms, free operations, monotones; position relative to Blackwell order and the comparison of statistical experiments) | Mainly vocabulary | Consistent framework, positioning incomplete | Framing needed for the results; positioning explicit |
| A-e | 10% | Coupling of theory and numerics | Separate | Numerics test the theory | Numerics and theory drive each other |
| A-f | 10% | Presentation (paper-ready, clear claims and limits) | Notes | Long notes with a clear status board | arXiv-ready paper(s) |

## 2. Materials

Give every model the same bundle:

1. `ABSTRACT.md` (plan with per-claim status),
2. `README.md`,
3. `theory/main.pdf` (built with `make -C theory pdf`),
4. `results/README.md`,
5. the GitHub URL (some models cannot open it; the files above must be enough).

## 3. Prompt (use verbatim; a fresh chat for every run)

> You are a critical reviewer with expertise in quantum error correction and quantum resource theories, writing for (i) a research manager at an industry quantum lab and (ii) a theory journal. Evaluate the attached research project using the rubric below. Do not assume the attached claims are correct or novel: check novelty against the literature you know and name the closest prior work. First list the three most serious weaknesses. Then give each sub-score (I-a … I-f, A-a … A-f) with one sentence of evidence that cites a file, theorem or number. Then give the two weighted totals (one decimal). A sub-score above 6 needs evidence that is already in the project, not a plan.
>
> [paste section 1 of EVALUATION.md here]

Do not say whose project it is: models score their user's work higher.

## 4. Procedure

- Run each model **twice** in fresh chats and record the mean. If the two runs differ by more than 1.5 on an axis, run a third time and use the median.
- Record the commit hash of the evaluated tree.
- Keep the full answers in `evaluations/<date>-<model>.md`, especially the weaknesses, which drive the next steps.

## 5. Log

| Date | Commit | Model | I (industry) | A (academic) | Top weakness named |
|---|---|---|---|---|---|
| 2026-10-03 | 02d1c4b | Claude Opus 5.5 (fresh-context subagent, one run, no web) | 4.3 | 5.9 | No real data or hardware regime; resource-theory core is conditional majorization (Gour et al. 2018) and uncited; hard questions open, bounds loose |
| 2026-10-03 | 0008094 | Claude Opus 5.5 (fresh-context subagents, two runs, no web): I 5.4 / 5.5, A 5.9 / 6.3 | **5.5** | **6.1** | Bounds still 9-240x loose and Willow d >= 5 not certifiable; no certificate from the data's own statistics; core order known, more QRT prior work uncited (Wang-Wilde, Takagi-Regula, Gour superchannels); notes not paper-shaped; surface code only |
