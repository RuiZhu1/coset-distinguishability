# lcd-qec: rigorous logical-error bounds for matching decoders

`lcd-qec` computes **rigorous upper bounds** on the per-shot logical failure probability of minimum-weight matching
(pymatching) on stim detector error models, and therefore also of the maximum-likelihood decoder. A sampled logical
error rate is an estimate with error bars; these bounds are guarantees, valid for every shot of every sample.

```bash
pip install "lcd-qec[sinter]"
lcd-peierls --generated surface_code:rotated_memory_z -d 5 -p 1e-3
# surface_code:rotated_memory_z d=5 r=5 p=0.001: p_L(MWPM) <= 2.000e-03  (Theorem 4.34, lam = 0.500, rho <= 0.4370)
lcd-peierls my_circuit.stim other_model.dem --json
```

Next to a sinter run:

```python
import sinter
from lcd.integrations.sinter_peierls import compare

stats = sinter.collect(num_workers=4, tasks=tasks, decoders=["pymatching"], max_shots=10**6)
for row in compare(tasks, stats):   # sampled rate and interval, rigorous bound, bound / rate, consistency flag
    print(row["json_metadata"], row["rate"], row["bound"], row["ratio"], row["consistent"])
```

On Google's published hardware models (fetched on demand, about 0.4 MB):

```python
from lcd.integrations.zenodo import willow_dem
from lcd.analysis.hardware import certifiability_margin

dem = willow_dem(distance=5, basis="Z", rounds=10, prior="rl_optimized").flattened()
print(certifiability_margin(dem, method="4.34"))   # s* = 1.19: the bound is finite at the fitted noise
```

## What is covered

- Detector error models whose observable part is a balanced matching graph (checked; unbalanced models are refused),
  with one or several logical observables (union bound over observables).
- stim's surface-code memory circuits under circuit-level and phenomenological noise; Google's published Sycamore and
  Willow models.
- Bounds: the circuit-level Peierls bound (Theorem 4.28) and its geodesic (4.32) and gap (4.34) refinements; by default
  the smallest. pymatching's integer weight rounding is accounted for.

How tight: at p = 1e-3 the bound is about 8x (d = 3) and 15x (d = 5) the simulated pymatching rate; it is finite up to
p = 6.9e-3, 4.4e-3, 3.6e-3 at d = 5, 7, 9. On Willow's data-fitted models it is finite at d = 3 and 5 but its value at
d = 5 is still above 1. Correlated matching, belief matching and neural decoders are not covered.

## Background

The theorems, proofs and their numerical checks are in the project repository,
<https://github.com/RuiZhu1/coset-distinguishability> (`theory/main.tex`, section 4.9), together with the research
programme they belong to (logical coset distinguishability as a resource). The proofs are drafts that have not been
independently reviewed. Licence: Apache-2.0.
