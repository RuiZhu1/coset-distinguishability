# Literature and novelty check (2026-10-03)

This file comes from four literature searches and one rubric evaluation (`evaluations/2026-10-03-claude-baseline.md`), all
done by Claude subagents on 2026-10-03. A reference marked † was found in a web search in that session, usually as an abstract
or a metadata page; the full text has **not** been read. A reference marked ‡ was named by the rubric evaluator from memory and
has not been looked up. A reference marked ✓ had its metadata (authors, title, year, journal reference) checked against arXiv or Crossref and, where arXiv has one, its abstract read on 2026-10-03, when the notes were given citations (§7 of the theory notes; Shannon 1958, Chernoff 1959, Torgersen 1991 and Nitinawarat–Atia–Veeravalli 2013: Crossref metadata only); for Dumer–Kovalev–Pryadko 1412.6172 the definitions of Υ and Υ_CSS (Eqs. 4–5) were also checked in the full text. ✓ does **not** mean the full text was read. **Before any novelty claim goes into a paper, read the closest references in full**, starting with those
marked ⚠.

## 1. Verdicts claim by claim

| Claim (theory number) | Verdict | Closest prior work | What the paper must say |
|---|---|---|---|
| ML failure = Bayes error of coset discrimination (Lemma 1.4); monotonicity by data processing (Thm 1.7) | **Known** (standard; not phrased as Bayes error) | DKLP quant-ph/0110143 ✓; Iyer–Poulin 1310.3235 ✓; Bravyi–Suchara–Vargo 1405.4883 ✓ | A reformulation, not a result |
| Complete family of monotones (Thm 0.6, Lemma 0.2) | **Known in substance**: Blackwell–Sherman–Stein with relabeling = conditional majorization | ⚠ Gour, Grudka, Horodecki, Kłobus, Łodyga, Narasimhachar, "The conditional uncertainty principle", 1506.07124 ✓ (PRA 97, 042130, 2018); Buscemi, CMP 2012, 1004.3794 ✓; Brandsen–Geng–Gour ‡; Torgersen 1991 ✓; Matsumoto 1012.2650 ✓; Jenčová 1512.07016 ✓; Chefles 0907.0866 ✓ | State it as an instance of conditional majorization and cite it. Remove "deliberately not cited" (sec0 l. 263) |
| Noise models as resources ordered by coset distinguishability (framework) | **Appears new** | Wang, Liu, Wang, Luo, "Quantum resource theory of coding for error correction", 2409.09416 ✓ (there the resources are codes, not noise); Wang–Wilde 1905.11629 †, 1907.06306 † (asymmetric distinguishability) | Contrast explicitly with 2409.09416 |
| Two-scalar characterisation of free reachability for Pauli + erasure (Thm 3.7, Prop 3.15) | **Appears new**; the second scalar is the Bloch shrinking factor, so referees may find it "natural" | BEC/BSC degradation orders (Hirche–Shaya 2504.16726 ✓); Hirche–Rouzé–Stilck França 2011.05949 ✓; Fanizza–Kianvash–Giovannetti, quantum flags, 1911.01977 ✓ | Cite the classical BEC/BSC orders as the analogue |
| Code-universal order, Conj 3.11 / Props 0.9, 3.16 | **Partly known**: "better for every code ⇔ degradable" holds when all encodings are allowed | Shannon 1958 channel inclusion ✓ (via 1705.01394 ✓, 2105.02281 ✓); ⚠ Buscemi, "Comparison of noisy channels and reverse data-processing theorems", 1803.02945 ✓; Buscemi 1511.08893 ✓ | The new content is the restriction to **linear/stabilizer** codes and the smaller set of free operations; say so |
| Tensor-power / catalytic rates, uncoded qubit (Props 0.13, Thms 0.14–0.15) | **Likely a special case** | Mu–Pomatto–Strack–Tamuz 1906.02838 ✓; Farooq–Fritz–Haapasalo–Tomamichel 2301.07353 ✓; ⚠ Rubboli–Haapasalo–Tomamichel 2601.23213 ✓ | Present it as an instance, or drop it as a headline |
| Exchange-rate bound R ≤ (3/4 − p₀)/(1 − e₀), equality criterion, [[5,1,3]] attains it, gap = code dependence (Thms 3.2, 3.30, 3.31, Props 3.32, 3.36, 3.37) | **Appears new** (no analytic exchange rate found) | Numerical precedents: Wu–Kolkowitz–Puri–Thompson 2201.03540 ✓ (exponent interpolates from (d+1)/2 to d); Stace–Barrett–Doherty 0904.3556 ✓; Chang et al. 2408.00842 ✓; Gu–Vaknin–Retzker–Kubica 2408.00829 ✓; Sahay et al. 2302.03063 ✓; Gu–Retzker–Kubica 2312.14060 ✓; AWS dual-rail 2307.08737 †; Kubica et al. PRX 2023 (2208.05461) ✓ | The headline result. Cite the numerical trade-off literature and the 2w + k ≥ d rule |
| Bhattacharyya sandwich ln(1/(μB_M)) ≤ α ≤ ln(1/B) (Prop 5.1, Cor 5.3), conjecture H_B | **Partly known** (the parameter); H_B appears new | Dumer–Kovalev–Pryadko 1412.6172 ✓ (they use Υ = y + 2(1−y)√(p(1−p)), the same erasure+Pauli Bhattacharyya parameter); DKLP ✓ | Cite Dumer–Kovalev–Pryadko |
| α as domain-wall tension; path entropy (§5.6, Prop 5.17) | **Known framing** | Chubb–Flammia 1809.10704 ✓; ⚠ English–Roberts–Bartlett–Doherty–Williamson, "Ising on the donut", 2512.10399 ✓ (tension model below threshold); Beverland–Brown–Kastoryano–Marolleau 1812.05117 ✓; Watson–Barrett NJP 2014 (1312.5213) ✓ | Cite all of them; 1812.05117 and Watson–Barrett treat the path entropy that H_B needs |
| ML–MWPM exponent gap Δα with erasures (P1 item 7) | **Appears new** as an exponent gap | Bravyi–Suchara–Vargo 1405.4883 ✓ (prefactor gap); Chubb 2101.04125 †; Higgott et al. 2203.04948 †; AlphaQubit, Nature 2024 † | Quantify the fit-window systematic error first |
| Envelope / Danskin derivatives from one table (Remark 3.13, Lemmas 3.28–3.29) | **Appears new** | Google error budgets 2207.06431 ✓, 2408.13687 ✓ (linearised sensitivities) | -- |
| Peierls bound for MWPM with erasures at code capacity (Thm 4.15) | **Known argument**; erasure and closed-form constants added | DKLP quant-ph/0110143 ✓; Kovalev–Pryadko 1208.2317 ✓ | Credit DKLP for the argument |
| Circuit-level Peierls bound on any balanced stim detector error model (DEM), certified Λ for all d (Thm 4.28, Props 4.29–4.30) | **Appears new** | Fowler 1206.0800 ✓ (one fixed circuit, p < 7.4×10⁻⁴; ours: any DEM, p < 1.4×10⁻³); ⚠ Yoshida–Lake–Yamasaki 2602.20238 ✓ (union-find threshold); Chai–Ng 2207.00217 ✓ (limits for non-Pauli noise); Kovalev–Pryadko 1208.2317 ✓ | Generalises Fowler 2012. **Done (2026-10-03).** Integer-weight gap in Thm 4.28: new part (e). Pymatching 2 minimizes 2 round(κw), so every β_e carries e^{λδ}, with δ = max|w|/(2(2²⁴−1)) ≈ 2×10⁻⁷. Done in the code (`pymatching_rounding_slack`); check N2 now tests minimality up to δ. Still open: interval arithmetic for Props 4.29–4.30.
| Finite-sample certificates valid for all d (Thms 4.5, 4.14, 4.20, Cor 4.16) | **Appears new** (only non-rigorous extrapolation found) | Google Λ fits 2207.06431 ✓, 2408.13687 ✓; Bravyi–Vargo 1308.6270 ✓; Mayer et al. 2509.13678 ✓; ⚠ Takou–Benito–Vezvaee–Lidar–Brown 2606.11496 ✓ (DEM estimated from syndrome data, using Willow data); Regev–Dilley–Bennink 2605.03054 ✓ | -- |
| "Bursts only change the exponent; chip events give a floor" (Lemma 4.6, Thms 4.23–4.24, Cor 4.31) | **Qualitatively known**; the certified quantitative versions and Thm 4.24 (no strength floor) appear new | ⚠ Tan–Pattison–McEwen–Preskill 2406.18897 ✓; Xu et al. 2203.16488 ✓; McEwen et al. 2104.05219 ✓; Suzuki et al. (Q3DE) 2501.00331 ✓; Aharonov–Kitaev–Preskill quant-ph/0510231 ✓; Terhal–Burkard quant-ph/0402104 ✓; Gottesman 2014 ‡ | Lemma 4.6 is not new as a statement |
| Burst shape non-monotonicity (Prop 4.11, Obs 4.25–4.26) | **Appears new** | -- | -- |
| Union bound vacuous as d → ∞ (Prop 5.2(b)) | **Probably folklore** | Forlivesi–Valentini–Chiani 2305.01301 ✓; Valentini et al. 2605.24501 †; Kovalev–Prabhakar–Dumer–Pryadko 1804.01950 ✓ | A remark, not a result |
| Pure-erasure exponent α(0,e) = ln(1/e) − ln 2 + O(e) (Thm 5.16) | **Appears new** as an explicit exponent | Stace–Barrett–Doherty 0904.3556 ✓; Colmenarez–Kim–Müller 2412.16727 ✓ (heuristic (e/e*)^d) | -- |
| Pattern vs counting detection, D_pattern = D_count + Λ₁D(π₁‖π₀), adaptive strong converse (§5.4–5.5) | **The information-theoretic framing appears new for QEC**; the general results have precedents | Chernoff 1959 ✓; Hayashi 2009 (0804.0686) ✓; Nitinawarat–Atia–Veeravalli 2013 ✓; burst detection by counting/thresholds: Q3DE 2501.00331 ✓, Vallero et al. 2506.16834 ✓, ReloQate 2603.00837 †, Chadwick et al. 2405.00146 †, Remm et al. 2502.17722 † | Benchmark against counting baselines (Q3DE, Vallero) |
| Supporting framing: a resource theory with a classical core | Supported | Nielsen quant-ph/9811053 ✓; Gour et al. 1309.6586 ✓; Chitambar–Gour 1806.06107 ✓ | -- |

**Summary.**
- **The strongest new results** are:
  - the exchange-rate bound with its equality theory;
  - the two-scalar preorder;
  - the computable circuit-level Peierls certificate on any DEM;
  - the finite-sample certificates valid for all d;
  - Thm 4.24;
  - Thm 5.16;
  - the burst-shape counterexamples.
- **The main risk** is that a referee reads §0 and the code-universal order as Blackwell, conditional majorization or Shannon inclusion under new names. The papers must say up front that the contribution is the QEC instantiation.

## 2. Public experimental data

| Dataset | DOI | Contents (checked from listings) | Size | Use |
|---|---|---|---|---|
| Google, "Suppressing quantum errors by scaling a surface code logical qubit" (2207.06431) | 10.5281/zenodo.6804040 | `detection_events.b8`, `obs_flips_actual.01`, `circuit_noisy.stim`, `.dem`, p_ij DEMs, decoder predictions; surface code d3 (four patches), d5; repetition code d25 r50 with 500k shots and **a documented high-energy event near shot 57775** | 315 MB | **First demo**: certificates on real DEMs, and burst detection with a known event |
| Google Willow, "QEC below the surface code threshold" (2408.13687) | 10.5281/zenodo.13273331 | 105Q d3/5/7, r up to 250, about 50k shots per config; 72Q repetition code d29, 100 time-ordered samples of 10⁵ shots × 1000 cycles (351 MB each); SI1000 and RL-prior DEMs | 112.5 GB in total (single files can be fetched) | Scale-up; hourly burst events that set the ~10⁻¹⁰ floor |
| Google, "Overcoming leakage" | 10.5281/zenodo.7302032 | Repetition code d21, surface code d3 with leakage | 10.1 GB | Leakage (assumption (L)) |
| Google, decoder priors / colour code / dynamic surface codes / RL control | 10.5281/zenodo.11403595, .14238944, .14238907, .18896801 | Google format | 0.7–11 GB | Breadth (colour code for I-f) |
| ETH, Krinner et al. 2022 | research-collection.ethz.ch (open access; format not checked) | d3 | ? | -- |
| USTC Zuchongzhi, IBM | None found | -- | -- | -- |

## 3. Gaps the project can fill for industry

- **No public tool computes a rigorous logical-error upper bound from a DEM**; sinter only gives sampling confidence intervals. `lcd.analysis.circuit_peierls` is the start of one.
- **Λ is reported as a fit with statistical error bars only** (2408.13687); there is no finite-sample certificate at large d.
- **The top pain point is correlated bursts** (cosmic rays, leakage), which set Willow's floor of about 10⁻¹⁰. Existing detectors from syndrome data use counting or threshold rules; none is Stein/Chernoff optimal.

## 4. Action items for the notes

1. **Done (2026-10-03).** The "deliberately not cited" note is replaced by a positioning paragraph (sec0); citations and status sentences were added in sec1, sec3, sec4 (above §4.9), sec5; new §7 `sec7-related.tex` gives related work by area and a claim → status → prior-work table; the bibliography grew from 13 to 64 entries; the 51 new ones were checked against arXiv/Crossref. Not cited (not verified, or not needed): Brandsen–Geng–Gour, Gottesman 2014, and the remaining † entries.
2. Fix the integer-weight gap in Thm 4.28 (sec4 l. 494): either evaluate at the decoder's rounded weights or carry the rounding error through the minimality step. Redo Props 4.29–4.30 in interval arithmetic.
3. Read the ⚠ references in full: 1506.07124, 1803.02945, 2601.23213, 2512.10399, 2602.20238, 2606.11496, 2406.18897.
4. Demote to remarks: Lemma 1.4, Thm 1.7, Prop 5.2(b), Lemma 4.6 as a statement.
