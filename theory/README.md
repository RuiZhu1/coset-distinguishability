# theory/ -- the theory track

Theory and numerics advance **in parallel**, with theory leading and the numerics testing the theory (including the monotonicity lemma).

```
theory/
├── main.tex                   main file (plain pdflatex); `make pdf` builds main.pdf
├── sec0-constitution.tex      constitutional audit: resource-theory axioms, complete family of monotones, three notions of rate (numbered section 0)
├── sec1-framework.tex         linear decoding structures, simulable reductions, monotonicity under the three free operations
├── sec2-representation.tex    erasure / leakage / burst / chip-scale events in the model; where the argument is valid
├── sec3-exchange.tex          exchange-rate bound, preorder on the Pauli+erasure class, the gap problem, envelope argument
├── sec4-certification.tex     conditional certification: union bound, parameter confidence sets, floor; open lemmas for bursts
├── sec5-information.tex       Chernoff/union sandwich, reference rates, Poisson bound, detection exponents
├── sec6-map.tex               theory <-> numerics map, open problems, schedule in parallel with M0-M5
├── checks/exact_small_codes.py  exact ML checks (no sampling, no approximation) on three codes with n <= 9
├── checks/constitution.py       structural checks of section 0 (LP over simulable reductions; about 4 s)
├── checks/structures.py         exact eps*(p, e) of any linear decoding structure (nested subspaces S < N of F_2^{2n}), enumeration of all structures
├── checks/universal_order_search.py  counterexample search for the single-loss form of Conjecture 3.11 (about 8 min; `make -C theory search`)
├── checks/hardened_structures_search.py  obstruction lemmas on all structures with n <= 3, small-p marginal rates, n = 6 hill-climbing (about 15 min)
├── checks/exponent_criterion.py  channel-coding exponent criterion for large structures (seconds)
└── checks/completion_checks.py   checks K1-K7 of the theory-completion results (genie, burst bound, O1 counterexample, [[5,1,3]] certificate)
```

## Build and run

```bash
make -C theory pdf      # needs a TeX distribution; see below
make -C theory check    # needs numpy; about 35 s; non-zero exit status if an inequality is violated
```

`check` runs both scripts; their last lines should read `ALL INEQUALITY CHECKS PASSED` and `ALL STRUCTURAL CHECKS PASSED`.

**TeX packages.** The notes are plain `pdflatex` (no CJK, no XeLaTeX). On Debian/Ubuntu the minimal set is

```bash
sudo apt-get install -y --no-install-recommends texlive-latex-recommended latexmk
```

(about 120 MB installed; it pulls in `texlive-base`, `texlive-binaries` and `texlive-latex-base`). This set was derived from
the files read by a successful build; it has not been tried on a bare container. In an ephemeral cloud session the packages
disappear with the container; to avoid reinstalling, put the command in the environment's setup script.

## Status tags

| Tag | Meaning |
|---|---|
| `PROVED` | A complete argument is in the notes. **Still a first draft, not independently reviewed**; the author should check it line by line before citing it. |
| `SKETCH` | The framework of the argument is complete; some steps cite standard results (e.g. the Chernoff exponent) or geometric details are only outlined. |
| `NUMERICAL` | Observed in the exact computations of `checks/`; no proof yet. |
| `CONJECTURE` | An explicit conjecture. |
| `PLAN` | Only a statement and a proof route. |

## Status board

| Number | Content | Status | Numerical check |
|---|---|---|---|
| Lemma 0.2, Proposition 0.3 | Kernel form of simulable reductions; they form a category (identity, composition) | PROVED | N1-N3 |
| Proposition 0.4 | Free objects (uniform label independent of the observation), no resource generation, golden rule, faithfulness of $\varepsilon^\star$ | PROVED | N6 |
| Proposition 0.5 | Parallel composition; $-\log_2(1-\varepsilon^\star)$ is additive | PROVED | N4 |
| Theorem 0.6 | Complete family of monotones $\Phi_\Lambda$ (Blackwell--Sherman--Stein with relabeling); an adaptation of the standard separation argument | PROVED | N2, N3 |
| Corollary 0.7 | $\varepsilon^\star$ alone is not complete; the free operations are maximal | PROVED | N5 |
| Propositions 0.8, 0.9 | Level 2 $\subseteq$ Level 1; for the uncoded qubit, local free reachability = arbitrary simulable reduction = $\Phi$-order for all $\varphi$ | PROVED | N1 |
| Remark 0.10 | Reformulation of Conjecture 3.11: all-losses form proved (uncoded qubit); single-loss form for $\varepsilon^\star$ open | -- | `checks/universal_order_search.py` (done; no reversal of trade pairs, Observation 3.17) |
| Proposition 0.11 | Marginal rates of monotones of the local free order fill exactly $[0,c]$; $c$ is attained by the free-order monotone $1-\mu$; $R_B$ and $R_{\rm hash}$ are marginal rates of monotones ($Q_{\rm hash}$: proved, Proposition 0.12) | PROVED | C7, N1, N7 |
| Proposition 0.12 | $H(L\mid Y)$ is a member of the complete family, so $Q_{\rm hash}=1-H(L\mid Y)$ is a monotone (uncoded qubit); $\Phi_\alpha$ multiplicative and $H(L\mid Y)$ additive under $\otimes$: necessary conditions for conversions between tensor powers (R-c) | PROVED; sufficiency for R-c open | N7 |
| Theorem 1.7 | Monotonicity of $\varepsilon^\star$ under simulable reductions (coarse-graining / discarding flags: data processing; superposition: simulation argument) | PROVED | C1, C2 |
| Proposition 1.9 | (F1), (F2), (F2') are simulable reductions; superposing independent noise never decreases $\varepsilon^\star$ | PROVED | C1, C2 |
| Proposition 2.1 | Erasure = flagged completely depolarizing noise (Pauli twirl); the maximally mixed model is conservative for real noise | PROVED | -- |
| Proposition 2.7 | Monotonicity under leakage given assumption (L); where (L) fails | PROVED (under the assumption) | -- |
| Proposition 2.10 | Monotonicity in the event-layer rate directions (Poisson superposition) | PROVED | -- |
| Remark 2.11 / Proposition 4.11 (O1) | Monotonicity in the shape parameters fails for unflagged bursts: exact counterexample in $p_b$ on the $d=3$ surface code ($\varepsilon^\star$ drops by 26% from $p_b=1/2$ to $3/4$); radius and duration untested | PROVED (exact computation) | K4 |
| Theorem 3.1 | Partial flag-discarding inequality $\varepsilon^\star(p,e_0{+}\delta)\le\varepsilon^\star(p'',e_0)$ | PROVED | C3 |
| Theorem 3.2 | Exchange-rate bound $R_{e\to p}\le(3/4-p_0)/(1-e_0)$ | PROVED | C7; **P1 (M2)** |
| Observation 3.5 | The bound is attained by $[[5,1,3]]$ at $e_0{=}0$ (8 significant digits) | NUMERICAL; equality now PROVED (Propositions 3.32, 3.36) | C7, K5 |
| Theorem 3.7 | Necessary and sufficient condition for the local-free-operation preorder on the code-capacity Pauli+erasure class | PROVED | C4 |
| Open problem 3.10 | The gap $c-R_\alpha$: code dependence or incompleteness of the free operations | qualitative part PROVED: the gap is code dependence (Proposition 3.37); quantitative value open ($H_B$) | **P1 (M2)** |
| Remark 3.13 / Observation 3.14 | Envelope argument for ML derivatives (the P1 sampling scheme gets $\partial p_L/\partial p$, $\partial p_L/\partial e$ from one table $f$); exact check on small codes and at d = 5, 7 | PROVED via Lemma 3.29 / NUMERICAL | C8; `envelope_check.py` |
| Proposition 3.15 | The local free order is cut out by two scalars: $p'\ge p$ and $\mu'\le\mu$ | PROVED | N1 (LP), `tests/test_theory_structures.py` |
| Proposition 3.16 | Uncoded qubit: $\varepsilon^\star=\tfrac34(1-\mu)$; two-qubit structure $F_2$: $\varepsilon^\star=\tfrac34(1-\nu)$, $\nu=(1-e^2)\lambda$; $\nu_k=(1-e^k)\lambda$ are free-order monotones separating all unreachable pairs | PROVED | S1 (closed forms to 2e-16) |
| Observation 3.17 | Exhaustive search over all linear structures on $n\le3$ qubits, hill-climbing for $n=4,5$: minimal marginal rate ($0$ at $e_0=0$, positive for $e_0>0$); trade pairs (less Pauli, more erasure) are not reversed by any structure found | NUMERICAL | `checks/universal_order_search.py` (S2-S4) |
| Conjecture 3.11 | Code-universal operational order = local free order. All-losses form: proved (Proposition 0.9, Remark 0.10). Single-loss form ($\varepsilon^\star$ only): proved for pairs with $\mu'>\mu$ or $\nu'>\nu$ (Proposition 3.16); **numerical evidence against it** for the remaining ``trade'' pairs (Observation 3.17) | CONJECTURE (evidence against) | `checks/universal_order_search.py` |
| Conjecture 3.12 | $H_B$: $\alpha$ is a function of the Bhattacharyya parameter $B$, $R_\alpha=R_B$ | CONJECTURE; P1 data consistent within about 1 standard error at $e_0>0$ (Observation 3.26), not confirmed; heuristic leading-order support at small $B$ (§3.9) | **P1 (M2)** |
| Lemma 3.19 | Erasure jump: a logical operator supported on $T$ forces $\varepsilon^\star_A\ge\tfrac12$ for $A\supseteq T$ | PROVED | S1 of `checks/hardened_structures_search.py` (all structures, $n\le3$) |
| Lemma 3.20 | Erasure-only failure $\varepsilon^\star_A(0)=1-2^{-r(A)}$ | PROVED | S2 (all structures, $n\le3$) |
| Lemma 3.21 / Corollary 3.22 | Failure of order $j$ in $p$ needs a logical of weight $\le2j$; hence no erasure hardening beyond order $2j$ | PROVED | S3 ($j=1$, all structures, $n\le3$) |
| Proposition 3.23 | Marginal rate of $\mathrm{Rep}_k$ at $p\to0$ is $c\cdot2e/(2e+(k-1)(1-e))$; the infimum over structures tends to 0 as $p_0\to0$ | PROVED | S4 (all structures, $n\le3$: $0.0523\,c$, $0.2003\,c$ at $p_0=10^{-4}$) |
| Observation 3.24 / Remark 3.25 | Checks of the lemmas; channel-coding exponent criterion for large $S=0$ structures (sufficient for reversal) is not met by any of the six trade pairs | NUMERICAL | `checks/exponent_criterion.py` |
| Lemmas 3.28, 3.29 | One qubit: failure rates affine and $\varepsilon^\star$ concave in its rate, rate $3/4$ = erased; Danskin form of the envelope identity (one-sided derivatives = min over ML rules) | PROVED | K6 |
| Theorem 3.30 | Rate bound with one-sided derivatives, $\partial_e\varepsilon^\star\le c\,\partial_p^+\varepsilon^\star$, no differentiability assumption | PROVED | K6, C7 |
| Theorem 3.31 (O6) | Equality in the rate bound iff one extra flag never changes the ML decision (E1), plus a tie condition (E2) | PROVED | `checks/sharp_codes_scan.py` |
| Proposition 3.32 | $[[5,1,3]]$: (E1) and unique ML decision for all $0<p<3/4$, so $R^{(d)}=c$ exactly at $e_0=0$; Steane and $d=3$ surface violate (E1) | PROVED (computer-assisted, exact integers; also by hand, Proposition 3.36) | K5 |
| Proposition 3.33 | (E1) in polynomial form at $e_0=0$; for small $p$ it is a lexicographic condition on class polynomials | PROVED | -- |
| Lemma 3.34 | Split-logical syndromes: every code has two classes with minimum weights $\lfloor d/2\rfloor,\lceil d/2\rceil$ in one syndrome, so minimum weights never decide (E1) | PROVED | -- |
| Corollary 3.35 | First-order obstruction to (E1) by multiplicities at small $p$ | PROVED | check on a code library (to write) |
| Proposition 3.36 (O6) | $[[5,1,3]]$ by hand: closed-form class polynomials; one flag at a single-error syndrome makes all four classes exactly equally likely; equality $R=c$ for all $p$ | PROVED | K5 (same conclusion) |
| Proposition 3.37 | Gap problem: no code-universally sound enlargement of the free operations lowers $c$; the gap $c-R_\alpha$ is code dependence | PROVED | -- |
| Observation 3.26 / Remark 3.27 | P1/M2: ML exchange rate of the surface code, $d=5$--$11$, six work points: $R^{(d)}\le c$ everywhere; $R_\alpha/c\approx0.3$ at $e_0>0$, $R_\alpha\approx R_B$ | NUMERICAL | `results/p1_exchange_rate.json` |
| Lemma 4.1 / Corollary 4.2 | Union (Bhattacharyya) bound $\varepsilon^\star\le\tfrac12\widetilde W(B)$ | PROVED | C5 |
| Theorem 4.5 | Finite-sample certification at code capacity (independent of $p_L$ data) | PROVED | `experiments/certification/cc_certificate.py`: coverage 0.95-1.00 (target 0.95), certificate >= exact ML at d = 3, 5 |
| Lemma 4.6 | Floor from chip-scale events, $\varepsilon^\star\ge q_c(1-2^{-q})$, and upper bound | PROVED | **P2 (M3-M4)** |
| Lemma 4.7 | Union bound for burst mixtures | PROVED | -- |
| Lemma 4.8 | Tilted Bhattacharyya bound $\mathfrak B(P;G)\le\mathbb E_P\sqrt{Q(X+G)/Q(X)}$ | PROVED | K3 |
| Theorem 4.10 (O3) | Single-sum product bound for unflagged burst mixtures, $\varepsilon^\star\le\tfrac12\widetilde W(B_{\rm burst})$, uniform in the burst strength; no floor | PROVED | K3 |
| Lemma 4.12, Proposition 4.13 (O2) | Limits for $p,e$ robust to contamination; burst-rate limits need a detection efficiency; impossible without a strength floor | PROVED | K7 |
| Theorem 4.14 | Conditional certification with bursts + chip events (code capacity), uniform over burst strengths | PROVED | coverage simulation (to do) |
| Theorem 4.15, Corollary 4.16 (O5) | Peierls bound and finite-sample certificate for MWPM with erasures, $p_L^{\rm MWPM}\le 2dB_M(3B_M)^{d-1}/(1-3B_M)$ | PROVED | MWPM simulation (to do) |
| Definition 4.17, Theorem 4.18 (O3, space-time) | Burst union bound for any linear decoding structure (circuit level, stim DEMs): $\varepsilon^\star\le\tfrac12\widetilde W^{\mathcal L}(\bar B)$, uniform in burst strength | PROVED | small DEM check (to write) |
| Lemma 4.19 (O2, space-time) | Parameter limits from one detector: firing probability $\ge\tfrac12(1-\prod\lambda_\ell)$ under contamination | PROVED | to write |
| Theorem 4.20 | Conditional certification with bursts + chip events for any linear decoding structure (needs $\widetilde W^{\mathcal L}$ of the DEM, not computed) | PROVED | -- |
| Proposition 4.21 (O7) | Closed-form ML certificate for every $d$: $\varepsilon^\star_d\le\min\{\tfrac12\widetilde W_d(B),\ $Peierls$\}$ | PROVED | -- |
| Proposition 4.22 (O7) | Self-avoiding-walk refinement of the Peierls bound: range $4.29\%\to$ about $4.9\%$ at $e=0$ | PROVED (constants from cited $c_K$, hand-computed) | enumerate $c_K$ (to write) |
| Proposition 5.1 | Chernoff upper bound $\alpha_+\le\ln(1/B)$, with explicit finite-$d$ form $\varepsilon_{\rm genie}\ge\tfrac12B^de^{-\sqrt{d\sigma^2}/2}$; Step 1 proved for all odd $d$ | PROVED | C6 ($d{=}3$), K1, K2, `checks/genie_large_d.py` |
| Proposition 5.2 | Union lower bound $\alpha_-\ge-\Phi(B)$ | PROVED | -- |
| Proposition 5.5 | Poisson bound $T\ge\ln((1-a)/\delta)/r$ (pattern methods cannot beat $1/r$) | PROVED | **P2 (M4)** |
| Proposition 5.6 | Chain of exponents $D_{\rm count}\le D_{\rm pattern}\le D_{\rm event}$ | PROVED (chain; ratio of rounds under the marking model, Theorem 5.10) | **P2 (M4)** |
| Open problem 5.7 (O4) | Experimental-design theorem | two-hypothesis part PROVED under (M) (5.8-5.11); composite hypotheses open | -- |
| Lemma 5.8, Theorem 5.9 | Marked Poisson model: $D_{\rm pattern}=D_{\rm count}+\Lambda_1D(\pi_1\|\pi_0)$; weak-signal gain $1+\chi^2$ | PROVED | **P2 (M4)** |
| Theorem 5.10 | Stein form: $T_{\rm count}/T_{\rm pattern}\to D_{\rm pattern}/D_{\rm count}$ (also Chernoff/Bayes) | PROVED | **P2 (M4)** |
| Theorem 5.11 | Adaptive designs do not beat the best single experiment (Stein, weak converse); optimal design maximizes $D_i/c_i$ | PROVED | -- |

Section 0 is the constitutional audit; the axiom list it uses is an assumption of the author of these notes and must be confirmed or replaced (see its first subsection). Numbers refer to the numbering in `main.pdf` (checked against the `.aux` file whenever this table is updated).

## Two hard conventions with the numerics track

1. **Every theorem comes with a falsification test.** A violation means the theorem or the implementation is wrong. Tests live in `checks/` or `tests/`.
2. **MWPM cannot falsify upper bounds for ML.** Monotonicity, the union bound, and the certification theorem are statements about Bayes-optimal (ML) decoding;
   the MWPM failure rate is $\ge\varepsilon^\star$ and can neither confirm nor refute them. Use exact ML (small codes, or `lcd.decoders.ExactTNMLDecoder` for $d\lesssim 11$) or tensor-network ML.
   Lower bounds (floor, Poisson bound, genie bound) hold for any decoder and can be tested with stim + MWPM.

`checks/exact_small_codes.py` is also the oracle for the ML decoder: on the $d=3$ rotated surface code the decoder's $\varepsilon^\star(p,e)$
must agree with it to machine precision (done: see `tests/test_tn_ml.py` and `experiments/p1_erasure_pauli/crosscheck_ml.py`).
