# theory/ -- the theory track

Theory and numerics advance **in parallel**, with theory leading and the numerics testing the theory (including the monotonicity lemma).

```
theory/
├── main.tex                   main file (plain pdflatex); `make pdf` builds main.pdf
├── sec1-framework.tex         linear decoding structures, simulable reductions, monotonicity under the three free operations
├── sec2-representation.tex    erasure / leakage / burst / chip-scale events in the model; where the argument is valid
├── sec3-exchange.tex          exchange-rate bound, preorder on the Pauli+erasure class, the gap problem, envelope argument
├── sec4-certification.tex     conditional certification: union bound, parameter confidence sets, floor; open lemmas for bursts
├── sec5-information.tex       Chernoff/union sandwich, reference rates, Poisson bound, detection exponents
├── sec6-map.tex               theory <-> numerics map, open problems, schedule in parallel with M0-M5
└── checks/exact_small_codes.py  exact ML checks (no sampling, no approximation) on three codes with n <= 9
```

## Build and run

```bash
make -C theory pdf      # needs a TeX distribution; see below
make -C theory check    # needs numpy; about 35 s; non-zero exit status if an inequality is violated
```

The last line of `check` should read `ALL INEQUALITY CHECKS PASSED`.

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
| Theorem 1.7 | Monotonicity of $\varepsilon^\star$ under simulable reductions (coarse-graining / discarding flags: data processing; superposition: simulation argument) | PROVED | C1, C2 |
| Proposition 1.9 | (F1), (F2), (F2') are simulable reductions; superposing independent noise never decreases $\varepsilon^\star$ | PROVED | C1, C2 |
| Proposition 2.1 | Erasure = flagged completely depolarizing noise (Pauli twirl); the maximally mixed model is conservative for real noise | PROVED | -- |
| Proposition 2.7 | Monotonicity under leakage given assumption (L); where (L) fails | PROVED (under the assumption) | -- |
| Proposition 2.10 | Monotonicity in the event-layer rate directions (Poisson superposition) | PROVED | -- |
| Remark 2.11 / Open problem 4.9 (O1) | Monotonicity in the shape parameters $(R,T_b,p_b)$: **not** given by free operations when unflagged | open | -- |
| Theorem 3.1 | Partial flag-discarding inequality $\varepsilon^\star(p,e_0{+}\delta)\le\varepsilon^\star(p'',e_0)$ | PROVED | C3 |
| Theorem 3.2 | Exchange-rate bound $R_{e\to p}\le(3/4-p_0)/(1-e_0)$ | PROVED | C7; **P1 (M2)** |
| Observation 3.5 | The bound is attained by $[[5,1,3]]$ at $e_0{=}0$ (8 significant digits) | NUMERICAL | C7 |
| Theorem 3.7 | Necessary and sufficient condition for the local-free-operation preorder on the code-capacity Pauli+erasure class | PROVED | C4 |
| Open problem 3.10 | The gap $c-R_\alpha$: code dependence or incompleteness of the free operations | open | **P1 (M2)** |
| Remark 3.13 / Observation 3.14 | Envelope argument for ML derivatives (the P1 sampling scheme gets $\partial p_L/\partial p$, $\partial p_L/\partial e$ from one table $f$); exact check on small codes and at d = 5, 7 | SKETCH / NUMERICAL | C8; `envelope_check.py` |
| Conjecture 3.11 | Code-universal operational order = local free order (the "strong" form of the iff) | CONJECTURE | counterexample search (to do) |
| Conjecture 3.12 | $H_B$: $\alpha$ is a function of the Bhattacharyya parameter $B$, $R_\alpha=R_B$ | CONJECTURE | **P1 (M2)** |
| Lemma 4.1 / Corollary 4.2 | Union (Bhattacharyya) bound $\varepsilon^\star\le\tfrac12\widetilde W(B)$ | PROVED | C5 |
| Theorem 4.5 | Finite-sample certification at code capacity (independent of $p_L$ data) | PROVED | coverage simulation (to do) |
| Lemma 4.6 | Floor from chip-scale events, $\varepsilon^\star\ge q_c(1-2^{-q})$, and upper bound | PROVED | **P2 (M3-M4)** |
| Lemma 4.7 | Union bound for burst mixtures | PROVED | -- |
| Theorem 4.8 | Conditional certification with bursts + chip events | PLAN (needs O1-O3) | P2 + exact ML at small $d$ |
| Proposition 5.1 | Chernoff upper bound $\alpha_+\le\ln(1/B)$ | SKETCH | C6 ($d{=}3$) |
| Proposition 5.2 | Union lower bound $\alpha_-\ge-\Phi(B)$ | PROVED | -- |
| Proposition 5.5 | Poisson bound $T\ge\ln((1-a)/\delta)/r$ (pattern methods cannot beat $1/r$) | PROVED | **P2 (M4)** |
| Proposition 5.6 | Chain of exponents $D_{\rm count}\le D_{\rm pattern}\le D_{\rm event}$ | PROVED (chain of inequalities) | **P2 (M4)** |

Numbers refer to the numbering in `main.pdf` (checked against the `.aux` file whenever this table is updated).

## Two hard conventions with the numerics track

1. **Every theorem comes with a falsification test.** A violation means the theorem or the implementation is wrong. Tests live in `checks/` or `tests/`.
2. **MWPM cannot falsify upper bounds for ML.** Monotonicity, the union bound, and the certification theorem are statements about Bayes-optimal (ML) decoding;
   the MWPM failure rate is $\ge\varepsilon^\star$ and can neither confirm nor refute them. Use exact ML (small codes, or `lcd.decoders.ExactTNMLDecoder` for $d\lesssim 11$) or tensor-network ML.
   Lower bounds (floor, Poisson bound, genie bound) hold for any decoder and can be tested with stim + MWPM.

`checks/exact_small_codes.py` is also the oracle for the ML decoder: on the $d=3$ rotated surface code the decoder's $\varepsilon^\star(p,e)$
must agree with it to machine precision (done: see `tests/test_tn_ml.py` and `experiments/p1_erasure_pauli/crosscheck_ml.py`).
