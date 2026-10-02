# results/

Raw data with the configuration that produced them. Every `.json` records the script, its arguments, seeds, package
versions, the git commit and whether `src/` or `experiments/` had uncommitted changes (`git_dirty_src`).

| Files | Script (run from the repository root) | Content |
|---|---|---|
| `p1_sampling_benchmark.{csv,json,npz}` | `experiments/p1_erasure_pauli/sampling_benchmark.py --dmax 15 --budget 1.5e6` | Cost of the stratified estimator for MWPM, e = 0, d = 5..15, p in {0.02, 0.04, 0.06}. The npz holds the per-stratum tables `[k, w, N, fails]`; the MWPM table does not depend on p. |
| `ml_crosscheck.json` | `experiments/p1_erasure_pauli/crosscheck_ml.py` | Stratified exact-ML estimate with erasures vs exhaustive enumeration at d = 3; qecsim MPS decoder vs the exact ML decoder at d = 3, 5. |
| `threshold_check_depolarizing.{csv,json}` | `threshold_check.py --scenario depolarizing --shots 40000` | Exact ML and MWPM crossings, depolarizing noise, d = 5, 7, 9. |
| `threshold_check_bitflip.{csv,json}` | `threshold_check.py --scenario bitflip --ds 5 7 9 11 13 15 --shots 100000` | MWPM crossings, bit-flip noise, d = 5..15. |
| `threshold_check_erasure.{csv,json}` | `threshold_check.py --scenario erasure --shots 30000` | Exact ML crossings, pure erasure, d = 5, 7, 9. |
| `mps_validation.json` | `experiments/p1_erasure_pauli/mps_validation.py` | Truncated-MPS decoder vs the exact decoder (d = 11, 13) and vs chi = 16 anchored to it (d = 15) on the strata that carry p_L, e = 0, MWPM-enriched stratified design. |
| `mps_erasure_check.json` | `experiments/p1_erasure_pauli/mps_erasure_check.py` | Truncated-MPS decoder vs the exact decoder on strata with erasures (d = 11, 13); counts the samples repaired by the invalid-Z fallback. A first version used strata where ML never fails and tested nothing; it was replaced by strata probed to have a failure fraction of 1e-2 or more. |
| `decoder_timing.json` | `experiments/p1_erasure_pauli/decoder_timing.py` | Cost per decode, exact vs MPS, d = 7..15 (idle machine, single core). |
| `universal_order_search.json` | `theory/checks/universal_order_search.py` | Counterexample search for the single-loss form of Conjecture 3.11: exact enumeration of all linear decoding structures on n <= 3 qubits (92,881 structures, 528 distinct eps* tables at n = 3): smallest marginal exchange rates, reversed pairs, trade pairs; time-budgeted hill-climbing for n = 4, 5 (heuristic, not bit-reproducible). Produced from a clean committed tree (commit and `git_dirty_src` recorded in the file). |
| `hardened_structures_search.json` | `theory/checks/hardened_structures_search.py` | Obstruction lemmas on every linear decoding structure with n <= 3 (erasure jump, erasure-only failure, first-order failure needs a short logical), smallest marginal rate at p0 = 1e-4, hill-climbing at n = 6 (heuristic, time-budgeted). Produced from a clean committed tree. |
| `exponent_criterion.json` | `theory/checks/exponent_criterion.py` | Channel-coding exponent criterion (random coding / expurgated vs sphere packing) for the trade pairs of the counterexample search; deterministic. |
| `p1_exchange_rate.json` | `experiments/p1_erasure_pauli/p1_exchange_rate.py analyze` | M2 analysis: p_L(d), R^(d), alpha and R_alpha (weighted fits over several d windows) at six work points with bootstrap errors (1000 replicates). Tables of the stratified ML estimates are in `p1_m2/tables/` (one `.npz` per (d, p0, e0): strata, failures, decodes, decoder statistics); the run is resumable and was extended in two passes (all points; then d = 11 to 5x the first-pass budget), so the tables are the sum of several runs with different seeds. |
| `mps_invalid_check.json` | `experiments/p1_erasure_pauli/mps_invalid_check.py` | Counts of repaired / dropped non-positive classes of the MPS decoder (d = 11 against the exact decoder, d = 13 production configuration); the drop path was never exercised in these validation runs; it occurred for 2 of the 2.7e5 decodes of the d = 13 production run (the first version of the run aborted on such a sample). |
| `envelope_check.json` | `experiments/p1_erasure_pauli/envelope_check.py` | Envelope identity for the exact ML decoder at d = 5, 7 (matched vs frozen finite differences on common random samples; discretisation error reported separately). A first version of the test compared the matched finite difference with the analytic derivative and wrongly failed, because it mixed in the O(h^2) discretisation error of the central difference; the corrected design was fixed before looking at its results. |

Notes on provenance.
- `git_dirty_src: true` in the threshold and cross-check files comes from a then-untracked, unrelated script
  (`envelope_check.py`) in `experiments/`; the code that produced the numbers was committed (commit recorded in each file).
- The bit-flip, erasure and cross-check files were regenerated from a committed tree with the same seeds as the first run and
  reproduced it exactly. The depolarizing scenario was regenerated with a different seeding scheme (the scenario index was added to
  the seed sequence), so its numbers differ from the first run within statistical error (see README, Section 2).
