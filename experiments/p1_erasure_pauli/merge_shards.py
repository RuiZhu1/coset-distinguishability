"""Merge sharded runs of p1_exchange_rate.py back into the stored failure tables.

Every shard started from the same stored table (results/p1_m2/tables/<name>.npz, or none for a new table) and wrote its own
copy with its additions. Independent samples per stratum add, so the merged table is
    fails = base + sum_s (fails_s - base),   N = base + sum_s (N_s - base),
and likewise for the run statistics and the compute seconds.

    python experiments/p1_erasure_pauli/merge_shards.py SHARD_ROOT [--tables results/p1_m2/tables]

SHARD_ROOT is searched recursively for <name>.npz files (one per shard). Exit status non-zero on inconsistent shards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STAT_KEYS = ("refined", "refine_changed", "fallback", "unrepaired")


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["stats"] = json.loads(str(z["stats"]))
    return out


def merge(base: dict | None, shards: list[dict]) -> dict:
    ref = base if base is not None else shards[0]
    for s in shards:
        if not ((s["k"] == ref["k"]).all() and (s["w"] == ref["w"]).all()):
            raise ValueError("strata differ between shards")
        for key in ("p0", "e0", "d", "cutoff"):
            if float(s[key]) != float(ref[key]):
                raise ValueError(f"{key} differs between shards")
    zero = np.zeros_like(ref["N"])
    b_fails = base["fails"] if base is not None else zero
    b_N = base["N"] if base is not None else zero
    b_sec = float(base["seconds"]) if base is not None else 0.0
    b_st = base["stats"] if base is not None else {}
    fails, N, sec = b_fails.copy(), b_N.copy(), b_sec
    stats = {k: int(b_st.get(k, 0)) for k in STAT_KEYS}
    for s in shards:
        dN, dF = s["N"] - b_N, s["fails"] - b_fails
        if (dN < 0).any() or (dF < 0).any() or (dF > dN).any():
            raise ValueError("a shard has fewer samples than the base table, or more failures than samples")
        N, fails, sec = N + dN, fails + dF, sec + float(s["seconds"]) - b_sec
        for k in STAT_KEYS:
            stats[k] += int(s["stats"].get(k, 0)) - int(b_st.get(k, 0))
    return dict(k=ref["k"], w=ref["w"], fails=fails, N=N, p0=ref["p0"], e0=ref["e0"], d=ref["d"], cutoff=ref["cutoff"],
                stats=json.dumps(stats), seconds=sec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--tables", type=Path, default=REPO / "results" / "p1_m2" / "tables")
    a = ap.parse_args()
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(a.root.rglob("*.npz")):
        if not f.name.endswith(".tmp.npz"):
            groups[f.name].append(f)
    if not groups:
        print("no shard tables found under", a.root)
        return 0
    a.tables.mkdir(parents=True, exist_ok=True)
    for name, files in sorted(groups.items()):
        bpath = a.tables / name
        base = load(bpath) if bpath.exists() else None
        shards = [load(f) for f in files]
        out = merge(base, shards)
        added = [int(s["N"].sum() - (base["N"].sum() if base is not None else 0)) for s in shards]
        if sum(added) == 0:
            print(f"{name}: no new decodes in {len(files)} shard files")
            continue
        tmp = bpath.with_suffix(".tmp.npz")
        np.savez(tmp, **out)
        os.replace(tmp, bpath)
        print(f"{name}: base {0 if base is None else int(base['N'].sum())} + {sum(added)} new decodes "
              f"from {len(files)} shards ({min(added)}-{max(added)} each) = {int(out['N'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
