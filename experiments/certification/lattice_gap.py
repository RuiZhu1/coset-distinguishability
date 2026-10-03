"""Distance-independent gap refinement (theory Proposition 4.35) for stim's rotated memory circuit, uniform noise.

  * the per-offset envelopes of q and of pymatching's weights (d = 5, 7, 9, 11, 13; they coincide);
  * hypothesis (A2): d_Go(z, y) <= D+(y - z) for all detector pairs, d = 5 ... 15 (max excess reported);
  * theta* (largest theta with rho(M) < 1, over a (lam, mu) grid) for p in {5e-4, 1e-3, 1.5e-3}, compared with
    Proposition 4.30, and the largest p with theta* > 0;
  * the closed-form bound p_L(d) <= (d+1)^2/2 (x_b + c_b)^2 F exp(-theta (d-2)) at p = 1e-3 for several theta.

Run:  python experiments/certification/lattice_gap.py      (about 30 min, < 3 GB). Writes results/lattice_gap.json.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.lattice_gap import Lattice, bound, check_A2, envelopes  # noqa: E402

L = 16


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main() -> int:
    t0 = time.time()
    out: dict = dict(meta=dict(script="experiments/certification/lattice_gap.py", git_commit=git("rev-parse", "HEAD"),
                               git_dirty_src=bool(git("status", "--porcelain", "src", "experiments")),
                               date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), L=L))
    p = 1e-3
    envs = {d: envelopes(d, p)["raw"] for d in (5, 7, 9, 11, 13)}
    same = all(np.allclose(np.array(list(envs[d].values())), np.array(list(envs[5].values()))) for d in envs)
    out["envelopes"] = dict(by_d=envs, identical=same)
    print("envelopes identical for d = 5..13:", same, flush=True)
    wp = {tuple(int(x) for x in k.strip("()").split(", ")): v[2] for k, v in envs[9].items()}
    out["A2"] = []
    for d in (5, 7, 9, 11, 13, 15):
        r = check_A2(d, p, wp)
        out["A2"].append(r)
        print(f"A2 d={d}: max d_Go - D+ = {r['max_excess']:.2e}", flush=True)
    out["theta_star"] = []
    for pp, prop430 in ((5e-4, 1.24), (1e-3, 0.671), (1.5e-3, None)):
        b = Lattice(pp, L).theta_star()
        b.update(p=pp, proposition_4_30=prop430)
        out["theta_star"].append(b)
        print(f"p={pp:g}: theta* >= {b['theta']:.3f} (Prop 4.30: {prop430})", flush=True)
    lo, hi = 1.4e-3, 2.0e-3
    for _ in range(9):
        m = (lo + hi) / 2
        Lt = Lattice(m, L)
        good = any(Lt.evaluate(lam, mu, 0.0)["rho"] < 1 for lam in (0.46, 0.5, 0.54) for mu in (0.1, 0.2, 0.3))
        lo, hi = (m, hi) if good else (lo, m)
    out["largest_p"] = dict(proposition_4_35=lo, proposition_4_30=1.405e-3)
    print(f"theta* > 0 for p < {lo:.4g} (Prop 4.30: 1.405e-3)", flush=True)
    Lt = Lattice(1e-3, L)
    rows = []
    for th in (0.3, 0.5, 0.6, 0.68):
        rows.append(dict(theta=th, **{f"d{d}": bound(Lt, d, 0.5, 0.1, th) for d in (5, 9, 15, 25, 35)}))
        print("theta", th, {k: f"{v:.2e}" for k, v in rows[-1].items() if k != "theta"}, flush=True)
    out["closed_form_p1e-3"] = rows
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    (REPO / "results" / "lattice_gap.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"done ({out['meta']['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
