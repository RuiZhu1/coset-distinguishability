"""The circuit-level Peierls bound (theory, Theorem 4.28) next to sinter's sampled logical error rates.

The bound is a rigorous upper bound, per shot, on the logical failure probability of the decoder sinter calls
``"pymatching"`` (no correlated decoding) on the task's detector error model, and therefore also on that of the
maximum-likelihood decoder. It says nothing about other decoders, which may be worse than ML.

    import sinter
    from lcd.integrations.sinter_peierls import compare

    tasks = [sinter.Task(circuit=c, json_metadata={"d": d, "p": p}) for ...]
    stats = sinter.collect(num_workers=4, tasks=tasks, decoders=["pymatching"], max_shots=10**6)
    for row in compare(tasks, stats):
        print(row["json_metadata"], row["rate"], row["bound"], row["consistent"])

A row is ``consistent`` unless the sampled rate is significantly above the bound (its lower Bayes-factor limit exceeds
the bound), which would falsify the theorem or the implementation.
"""
from __future__ import annotations

from typing import Iterable

import stim

from lcd.analysis.circuit_peierls import peierls_bound

#: sinter decoders that minimize pymatching's weights exactly (up to the rounding of Theorem 4.28(e)).
COVERED_DECODERS = frozenset({"pymatching"})


def task_dem(task) -> stim.DetectorErrorModel:
    """The detector error model sinter decodes for ``task`` (same flags as sinter's collection worker)."""
    if task.detector_error_model is not None:
        return task.detector_error_model
    circuit = task.circuit if task.circuit is not None else stim.Circuit.from_file(task.circuit_path)
    return circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


def bound_for_task(task, **kwargs) -> dict:
    """``peierls_bound`` on the task's detector error model; ``kwargs`` are passed through."""
    return peierls_bound(task_dem(task), **kwargs)


def compare(tasks: Iterable, stats: Iterable, *, max_likelihood_factor: float = 1e3) -> list[dict]:
    """One row per ``sinter.TaskStats`` whose task is in ``tasks`` and whose decoder is covered.

    Each row has the task's metadata, the sampled rate per shot with its likelihood-ratio interval
    (``sinter.fit_binomial``), the bound with its optimal ``lam``, ``ratio = bound / rate``, and ``consistent``.
    The bound is computed once per task.
    """
    import sinter

    # sinter fills in the decoder and the detector error model before hashing a task (its collection worker uses
    # decompose_errors=True, approximate_disjoint_errors=True), so the ids are recomputed the same way here.
    by_id = {}
    for t in tasks:
        circuit = t.circuit if t.circuit is not None else stim.Circuit.from_file(t.circuit_path)
        for dec in ([t.decoder] if t.decoder is not None else sorted(COVERED_DECODERS)):
            full = sinter.Task(circuit=circuit, decoder=dec, detector_error_model=task_dem(t),
                               postselection_mask=t.postselection_mask,
                               postselected_observables_mask=t.postselected_observables_mask,
                               json_metadata=t.json_metadata)
            by_id[full.strong_id()] = full
    cache: dict[str, dict] = {}
    rows = []
    for s in stats:
        if s.decoder not in COVERED_DECODERS or s.strong_id not in by_id:
            continue
        if s.strong_id not in cache:
            cache[s.strong_id] = bound_for_task(by_id[s.strong_id])
        b = cache[s.strong_id]
        kept = s.shots - s.discards
        fit = sinter.fit_binomial(num_shots=kept, num_hits=s.errors, max_likelihood_factor=max_likelihood_factor)
        rate = s.errors / kept if kept else float("nan")
        rows.append(dict(json_metadata=s.json_metadata, decoder=s.decoder, shots=kept, errors=s.errors,
                         rate=rate, rate_low=fit.low, rate_high=fit.high, bound=b["bound"], lam=b["lam"],
                         rho_upper=b["rho_upper"], delta=b["delta"],
                         ratio=b["bound"] / rate if rate > 0 else float("inf"),
                         consistent=bool(fit.low <= b["bound"])))
    return rows
