"""Several logical observables: per-observable bounds and the union bound on "some observable fails".

Theorems 4.28-4.34 are stated for one observable. For a detector error model with k observables, the decoder's
correction C is one edge set; observable j fails iff its parity on X + C is odd. Projecting the model onto observable j
(dropping the other observables' targets) leaves the graph and the decoder's weights unchanged provided no two parallel
mechanisms differ only in the dropped observables; ``project`` checks this by comparing pymatching's weights. Then each
per-observable bound applies, and P[some observable fails] <= sum_j bound_j, which is what sinter counts as an error.
"""
from __future__ import annotations

import numpy as np
import stim


def project(dem: stim.DetectorErrorModel, j: int) -> stim.DetectorErrorModel:
    """The flattened model with only observable j (renamed L0)."""
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            ts = []
            for t in inst.targets_copy():
                if t.is_logical_observable_id():
                    if t.val == j:
                        ts.append(stim.target_logical_observable_id(0))
                else:
                    ts.append(t)
            out.append("error", inst.args_copy(), ts)
        elif inst.type == "logical_observable":
            if inst.targets_copy()[0].val == j:
                out.append("logical_observable", [], [stim.target_logical_observable_id(0)])
        else:
            out.append(inst)
    return out


def _edge_weights(dem) -> dict:
    import pymatching
    m = pymatching.Matching.from_detector_error_model(dem)
    return {((u,) if v is None else tuple(sorted((u, v)))): a["weight"] for u, v, a in m.edges()}


def bound_any(dem: stim.DetectorErrorModel, fn) -> dict:
    """Apply ``fn`` (a single-observable bound, returning a dict with 'bound') per observable; union bound over them."""
    k = dem.num_observables
    if k == 1:
        r = dict(fn(dem))
        r["per_observable"] = [r["bound"]]
        return r
    full = _edge_weights(dem)
    per, parts = [], []
    for j in range(k):
        pj = project(dem, j)
        wj = _edge_weights(pj)
        for e, w in wj.items():
            if e in full and not np.isclose(w, full[e], rtol=1e-12, atol=0):
                raise NotImplementedError(f"observable {j}: projection changes the decoder's weight on edge {e}")
        r = fn(pj)
        parts.append(r)
        per.append(r["bound"])
    out = dict(parts[int(np.argmax(per))])
    out.update(bound=float(sum(per)), per_observable=per)
    return out
