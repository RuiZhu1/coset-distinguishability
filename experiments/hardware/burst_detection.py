"""Burst detection on real hardware data: counting versus a space-time scan (theory §5.4-5.5, Lemma 5.8, Theorems 5.9,
5.10, Proposition 5.5). First real-data test of the claim that pattern tests beat counting tests.

Data: Google 2022 (Zenodo 10.5281/zenodo.6804040), data/repetition_code_bZ_d25_r50_center_5_5: 500,000 time-ordered
shots of a distance-25 repetition code, 50 rounds; 1224 detectors = 51 rounds x 24 chain positions (the chain order comes
from the space-like edges of the shipped detector error model). The README documents a high-energy event near shot 57775.

Detectors (src/lcd/analysis/burst_detection.py), both on windows of W consecutive shots (W = 1: single shots):
  count     z-score of the total number of detection events (empirical baseline variance),
  pattern   max over rectangles (chain interval x round window, 9 widths x 11 heights, 61,537 rectangles) of the
            Bernoulli log-likelihood ratio of a common elevated rate against the per-cell baseline rates;
each in three variants: global_raw (per-cell rates from the training shots, as specified), global_masked (the same, but
cells in runs of >= 6 consecutive events of one detector -- the leakage signature that dominates the null tail of the scan
-- are removed before testing), local_masked (masked, rates from the REF = 8192 shots before the window's block: per-
detector rates drift by 3-8% over the run, which a global baseline turns into false alarms once many shots are summed).

Splits (no leakage between estimation and evaluation):
  events     500-shot bins of the per-shot total with robust z > 4 (median / MAD over bins) define event clusters;
             each is excluded with a margin [first bin - 2000, last bin + 5000);
  baseline   global: per-cell rates and count mean / variance from shots [0, 40000) minus excluded shots;
             local: the 8192 shots before the block (null / host), or before onset - 500 (real events, frozen);
  null/host  shots [40000, 500000) minus excluded shots, cut into blocks of 2048 whose 8192 preceding shots are clean;
             even blocks calibrate thresholds (null), odd blocks host the planted bursts (both pools span the whole run,
             so slow drift affects both). Windows of W shots lie inside one block.

Parts:
  (1) background and event characterisation (per-cell rates, over-dispersion, drift, streaks; the event's onset, spatial
      profile, decay);
  (2) thresholds at false-alarm rates 1e-4, 1e-3, 1e-2 per window, from null windows;
  (3) the real events: per-shot scores, empirical p-values, first alarm and alarm duration for W in WS_EVENT;
  (4) planted bursts on real background: an R-position x T-round rectangle (same place in all W shots of a window) in
      which each detector flips with extra probability theta (XOR), R in {3,5,9}, T in {3,10,30}, theta in
      {0.05,0.1,0.2}; detection probability at fixed false-alarm rate vs W, the number of shots W* needed for power
      0.5 / 0.9, and the ratio W*_count / W*_pattern next to the per-shot D_pattern / D_count of Lemma 5.8 and
      Theorem 5.9 (known location) as predicted by Theorem 5.10;
  (5) a search for localized events the count screen misses (pattern, W = 64, stride 32, over all shots).

    python experiments/hardware/burst_detection.py [--quick]

Writes results/burst_detection.json (not with --quick). Peak memory about 1 GB; run as a memory-capped systemd unit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import stim
from scipy.optimize import curve_fit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from lcd.analysis.burst_detection import (B8Reader, Baseline, CountDetector, ScanDetector, bernoulli_divergences,  # noqa: E402
                                          chain_layout, counts_and_trials, marked_poisson_divergences, planted_rates,
                                          streak_mask, thresholds)

DATA = REPO / "data" / "repetition_code_bZ_d25_r50_center_5_5"
SHOTS = 500_000
TRAIN_END = 40_000
BLOCK = 2048
FARS = [1e-4, 1e-3, 1e-2]
VARIANTS = {"global_raw": (None, "global"), "global_masked": (6, "global"), "local_masked": (6, "local")}
REF = 8192
WS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
WS_EVENT = [1, 4, 16, 64]
RS, TS, THETAS = [3, 5, 9], [3, 10, 30], [0.05, 0.1, 0.2]
NULL_CAP = 20_000
NULL_W1_CAP = 120_000
TRIALS = 300
SEED = 20261003


def git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "experiments"], cwd=REPO,
                                             text=True).strip())
    except Exception:
        commit, dirty = "unknown", True
    return commit, dirty


def read01(name: str) -> np.ndarray:
    b = np.frombuffer((DATA / name).read_bytes(), dtype=np.uint8)
    return (b[b != ord("\n")] - ord("0")).astype(np.uint8)


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ (1) characterisation
def screen_events(N: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    """Count screen: 500-shot bins with robust z > 4; clusters of flagged bins (gap <= 2 bins); exclusion intervals."""
    b = N[: len(N) // 500 * 500].reshape(-1, 500).mean(1)
    med = np.median(b)
    sd = 1.4826 * np.median(np.abs(b - med))
    z = (b - med) / sd
    flagged = np.where(z > 4)[0]
    clusters: list[list[int]] = []
    for i in flagged:
        if clusters and i - clusters[-1][-1] <= 2:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    events, excl = [], []
    for c in clusters:
        lo, hi = c[0] * 500, (c[-1] + 1) * 500
        events.append(dict(bins=[lo, hi], max_bin_z=float(z[c].max()), max_bin_mean=float(b[c].max())))
        excl.append((max(0, lo - 2000), min(len(N), hi + 5000)))
    return events, excl, dict(bin_median=float(med), bin_robust_sd=float(sd), iid_bin_sd=float(N.std() / np.sqrt(500)))


def onset(N: np.ndarray, lo: int, hi: int, level: float) -> int:
    """First shot s in [lo, hi) with min(N[s:s+10]) > level."""
    for s in range(lo, hi):
        if N[s:s + 10].min() > level:
            return s
    return -1


def characterise(r: B8Reader, L: np.ndarray, N: np.ndarray, train: np.ndarray, mis: np.ndarray, events: list[dict],
                 bl: Baseline) -> dict:
    p = bl.p
    tot = N[train].astype(float)
    blocks = N[: SHOTS // 10000 * 10000].reshape(-1, 10000).mean(1)
    st = {}
    sample = r.grid(slice(100_000, 120_000), L)
    for mr in (4, 6, 10):
        m = streak_mask(sample, mr)
        st[f"run>={mr}"] = dict(frac_shots=float(m.any((1, 2)).mean()), cells_per_shot=float(m.sum((1, 2)).mean()))
    pi = p.ravel()
    iid6 = float(sum(np.prod([p[t + j] for j in range(6)], axis=0).sum() for t in range(p.shape[0] - 5)))
    bg = dict(per_round_rate=np.round(p.mean(1), 5).tolist(), per_position_rate_bulk=np.round(p[1:-1].mean(0), 5).tolist(),
              mean_rate=float(pi.mean()), rate_range_bulk=[float(p[1:-1].min()), float(p[1:-1].max())],
              per_shot_total=dict(mean=float(tot.mean()), var=float(tot.var()), fano=float(tot.var() / tot.mean()),
                                  poisson_binomial_var=float((pi * (1 - pi)).sum())),
              block10k_mean=dict(min=float(blocks.min()), max=float(blocks.max()), sd=float(blocks.std()),
                                 iid_sd=float(tot.std() / 100)),
              streaks=st, streak_iid_expected_run6_starts_per_shot=iid6,
              masked_baseline_var=bl.count_var, decoder_mismatch_rate=float(mis.mean()))
    out = []
    lvl = tot.mean() + 0.5 * tot.std()
    for ev in events:
        lo, hi = ev["bins"]
        s0 = onset(N, lo - 1000, hi, lvl)
        loc = N[max(0, s0 - 3000):s0 - 100].astype(float)
        ref = loc.mean()
        e = dict(ev, onset=int(s0), onset_rule="first s with min(N[s:s+10]) > mean + 0.5 sd (training shots)",
                 local_mean_before=float(ref), N_from_onset_minus5=N[s0 - 5:s0 + 40].tolist())
        x = r.grid(slice(s0 - 1, s0 + 2), L)
        e["per_round_counts_last20_onset_minus1_to_plus1"] = x.sum(2)[:, -20:].tolist()
        e["onset_shot_last4_rounds"] = dict(events=int(x[1, -4:].sum()), baseline=float(bl.p[-4:].sum()))
        prof = []
        for a, b in [(0, 20), (20, 100), (100, 500), (500, 1500), (1500, 4000), (4000, 6000)]:
            y = r.grid(slice(s0 + a, s0 + b), L).mean(0) - p
            prof.append(dict(shots=[a, b], excess_total=float(y.sum()), excess_per_position=np.round(y.sum(0), 2).tolist(),
                             excess_per_round_mean=float(y.sum(1).mean()), mismatches=int(mis[s0 + a:s0 + b].sum())))
        e["excess_profile"] = prof
        # decay of the excess: 25-shot bins, offsets 10..3000
        off = np.arange(10, 3000, 25)
        ex = np.array([N[s0 + o:s0 + o + 25].mean() - ref for o in off])
        try:
            (A, tau), _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), off + 12.5, ex, p0=(ex[0], 200.0), maxfev=10000)
            e["decay_fit"] = dict(model="A exp(-shots/tau), 25-shot bins, offsets 10-3000", A=float(A), tau_shots=float(tau))
        except Exception as err:  # pragma: no cover
            e["decay_fit"] = dict(error=str(err))
        se = N[train].std() / np.sqrt(200)
        bins200 = np.array([N[s0 + o:s0 + o + 200].mean() - ref for o in range(0, 8000, 200)])
        below = np.where(bins200 < 2 * se)[0]
        e["excess_above_2se_200shot_bins_until"] = int(below[0] * 200) if len(below) else None
        e["mismatches_onset_to_plus300"] = int(mis[s0:s0 + 300].sum())
        out.append(e)
    return dict(background=bg, events=out)


# ------------------------------------------------------------------------------------------------ helpers
class DetBank:
    """Detectors per variant: global baseline (training shots) or local baseline (the REF shots before ``ref_end``)."""

    def __init__(self, r: B8Reader, L: np.ndarray, train: np.ndarray):
        self.r, self.L = r, L
        g = r.grid(train, L)
        self.glob = {mr: Baseline.fit(g, mr) for mr in {mr for mr, _ in VARIANTS.values()}}
        self.cache: dict = {}

    def baseline(self, v: str, ref_end: int) -> Baseline:
        mr, kind = VARIANTS[v]
        if kind == "global":
            return self.glob[mr]
        key = (mr, ref_end)
        if key not in self.cache:
            self.cache[key] = Baseline.fit(self.r.grid(slice(ref_end - REF, ref_end), self.L), mr)
        return self.cache[key]

    def get(self, v: str, ref_end: int) -> dict:
        b = self.baseline(v, ref_end)
        key = ("det", id(b))
        if key not in self.cache:
            self.cache[key] = {"count": CountDetector(b), "pattern": ScanDetector(b)}
        return self.cache[key]


def scores_for(K, M, W, det) -> dict:
    return {"count": det["count"].score(K, M, m=W), "pattern": det["pattern"].score(K, M, m=W)}


def null_scores(r, L, null_blocks, W, bank, cap) -> tuple[dict, int]:
    starts_all = [(b, s) for b in null_blocks for s in range(0, BLOCK - W + 1)]
    if len(starts_all) > cap:
        idx = np.unique(np.linspace(0, len(starts_all) - 1, cap).round().astype(int))
        starts_all = [starts_all[i] for i in idx]
    by_block: dict = {}
    for b, s in starts_all:
        by_block.setdefault(b, []).append(s)
    res = {v: {"count": [], "pattern": []} for v in VARIANTS}
    for b, ss in by_block.items():
        g = r.grid(slice(b, b + BLOCK), L)
        for v, (mr, _) in VARIANTS.items():
            K, M = counts_and_trials(g, mr, W, ss)
            for k, x in scores_for(K, M, W, bank.get(v, b)).items():
                res[v][k].append(x)
    return {v: {k: np.concatenate(x) for k, x in d.items()} for v, d in res.items()}, len(starts_all)


def w_star(Ws, power, target) -> float | None:
    """Smallest W at which the power reaches target, log2-linear interpolation between grid points."""
    for i, (w, pw) in enumerate(zip(Ws, power)):
        if pw >= target:
            if i == 0:
                return float(w)
            w0, p0 = Ws[i - 1], power[i - 1]
            f = (target - p0) / (pw - p0) if pw > p0 else 1.0
            return float(2 ** (np.log2(w0) + f * (np.log2(w) - np.log2(w0))))
    return None


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few windows and configurations; no output file")
    a = ap.parse_args()
    t0 = time.time()
    commit, dirty = git_info()
    ws = [1, 4, 16] if a.quick else WS
    ws_event = [1, 16] if a.quick else WS_EVENT
    rs, ts, thetas = ([5], [10], THETAS) if a.quick else (RS, TS, THETAS)
    cap = 3000 if a.quick else NULL_CAP
    trials = 100 if a.quick else TRIALS
    out: dict = dict(meta=dict(script="experiments/hardware/burst_detection.py", args=vars(a), git_commit=commit,
                               git_dirty_src=dirty, date=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               versions=dict(stim=stim.__version__, numpy=np.__version__, scipy=scipy.__version__),
                               data="data/repetition_code_bZ_d25_r50_center_5_5 (Zenodo 10.5281/zenodo.6804040)",
                               theorem="theory/sec5-information.tex: Lemma 5.8, Theorems 5.9, 5.10, Proposition 5.5",
                               seed=SEED, fars=FARS, reference_shots_local=REF,
                               variants={k: f"{kind} baseline, " + (f"streak mask min_run={mr}" if mr else "no mask")
                                         for k, (mr, kind) in VARIANTS.items()}))

    # ---------------------------------------------------------------- data, layout, screen, splits
    circ = stim.Circuit.from_file(DATA / "circuit_ideal.stim")
    dem = stim.DetectorErrorModel.from_file(DATA / "circuit_detector_error_model.dem")
    L = chain_layout(circ.get_detector_coordinates(), dem)
    r = B8Reader(DATA / "detection_events.b8", circ.num_detectors)
    assert r.shots == SHOTS
    N = np.concatenate([r.events(slice(s, s + 50000)).sum(1) for s in range(0, SHOTS, 50000)]).astype(np.int32)
    mis = read01("obs_flips_actual.01") ^ read01("obs_flips_predicted_by_correlated_matching.01")
    events, excl, screen = screen_events(N)
    log(f"layout {L.shape}; screen found {len(events)} event clusters: {[e['bins'] for e in events]}")
    clean = np.ones(SHOTS, bool)
    for lo, hi in excl:
        clean[lo:hi] = False
    train = np.where(clean[:TRAIN_END])[0]
    blocks, s = [], TRAIN_END
    while s + BLOCK <= SHOTS:
        bad = np.where(~clean[s - REF:s + BLOCK])[0]
        if len(bad) == 0:
            blocks.append(s)
            s += BLOCK
        else:
            s += int(bad[-1]) + 1
    null_blocks, host_blocks = blocks[0::2], blocks[1::2]
    if a.quick:
        null_blocks, host_blocks = null_blocks[:15], host_blocks[:10]
    out["splits"] = dict(screen=screen, exclusions=excl, baseline_shots=int(len(train)), baseline_range=[0, TRAIN_END],
                         block=BLOCK, null_blocks=len(null_blocks), host_blocks=len(host_blocks),
                         null_shots=len(null_blocks) * BLOCK, host_shots=len(host_blocks) * BLOCK)
    log(f"baseline {len(train)} shots; null {len(null_blocks)} blocks, host {len(host_blocks)} blocks of {BLOCK}")

    bank = DetBank(r, L, train)
    braw = bank.glob[None]
    d0 = bank.get("global_raw", 0)["pattern"]
    out["detectors"] = dict(rectangles=d0.regions(), widths=d0.widths, heights=d0.heights,
                            global_baseline={str(mr): dict(count_mean=b.count_mean, count_var=b.count_var)
                                             for mr, b in bank.glob.items()})

    out["characterisation"] = characterise(r, L, N, train, mis, events, bank.glob[6])
    evs = out["characterisation"]["events"]
    if a.quick:
        evs = [e for e in evs if e["bins"][0] <= 57775 < e["bins"][1] + 2000]
    log(f"characterisation done ({time.time() - t0:.0f}s); onsets {[e['onset'] for e in evs]}")

    # ---------------------------------------------------------------- (2) null calibration
    nulls, cal = {}, {}
    for W in sorted(set(ws) | set(ws_event)):
        tw = time.time()
        sc, n = null_scores(r, L, null_blocks, W, bank, (NULL_W1_CAP if W == 1 else cap) if not a.quick else cap)
        nulls[W] = sc
        cal[W] = dict(n_windows=n, n_effective=len(null_blocks) * (BLOCK // W),
                      thresholds={v: {k: thresholds(x, FARS) for k, x in d.items()} for v, d in sc.items()},
                      quantiles={v: {k: np.percentile(x, [50, 90, 99, 99.9]).tolist() for k, x in d.items()}
                                 for v, d in sc.items()})
        log(f"null W={W}: {n} windows ({time.time() - tw:.0f}s); thresholds@1e-3 "
            + ", ".join(f"{v}/{k} {cal[W]['thresholds'][v][k][1e-3]:.2f}" for v in VARIANTS for k in ("count", "pattern")))
    out["calibration"] = {str(W): c for W, c in cal.items()}

    # ---------------------------------------------------------------- (3) real events
    ev_out = []
    for e in evs:
        s0 = e["onset"]
        lo, hi = s0 - 500, s0 + 3000
        rec = dict(onset=s0, range=[lo, hi], local_reference=[lo - REF, lo], by_W={})
        for W in ws_event:
            g = r.grid(slice(lo - W + 1, hi), L)
            ends = np.arange(lo, hi)
            for v, (mr, _) in VARIANTS.items():
                K, M = counts_and_trials(g, mr, W, ends - lo)
                sc = scores_for(K, M, W, bank.get(v, lo))
                for k, x in sc.items():
                    th = cal[W]["thresholds"][v][k]
                    nul = np.sort(nulls[W][v][k])
                    d = dict(score_onset_minus1_to_plus30=np.round(x[500 - 1:500 + 31], 2).tolist())
                    d["null_pvalue_onset_to_plus5"] = [float((len(nul) - np.searchsorted(nul, y, "left")) / len(nul))
                                                       for y in x[500:506]]
                    for f in FARS:
                        al = x > th[f]
                        post = np.where(al[500:])[0]
                        d[f"far{f:g}"] = dict(threshold=th[f], first_alarm=int(ends[500 + post[0]]) if len(post) else None,
                                              delay=int(post[0]) if len(post) else None,
                                              alarms_after_onset=int(al[500:].sum()),
                                              last_alarm=int(ends[500 + post[-1]]) if len(post) else None,
                                              alarms_in_500_before=int(al[:500].sum()))
                    rec["by_W"].setdefault(str(W), {}).setdefault(v, {})[k] = d
            log(f"event {s0} W={W}: delays@1e-4 " + ", ".join(
                f"{v}/{k} {rec['by_W'][str(W)][v][k]['far0.0001']['delay']}"
                f" (n {rec['by_W'][str(W)][v][k]['far0.0001']['alarms_after_onset']})" for v in VARIANTS for k in ("count", "pattern")))
        ev_out.append(rec)
    out["real_events"] = ev_out

    # ---------------------------------------------------------------- (4) planted bursts
    p0 = braw.p
    planted = []
    for R in rs:
        for T in ts:
            prng = np.random.default_rng([SEED, R, T])
            rows = {(th, v): {W: None for W in ws} for th in thetas for v in VARIANTS}
            tconf = time.time()
            for W in ws:
                Ks = {(th, v): ([], [], []) for th in thetas for v in VARIANTS}
                for _ in range(trials):
                    b = host_blocks[prng.integers(len(host_blocks))]
                    s = b + int(prng.integers(0, BLOCK - W + 1))
                    g = r.grid(slice(s, s + W), L)
                    c0 = (int(prng.integers(0, g.shape[1] - T + 1)), int(prng.integers(0, g.shape[2] - R + 1)))
                    u = prng.random((W, T, R))
                    for th in thetas:
                        gp = g.copy()
                        gp[:, c0[0]:c0[0] + T, c0[1]:c0[1] + R] ^= (u < th).astype(np.uint8)
                        for v, (mr, _) in VARIANTS.items():
                            K, M = counts_and_trials(gp, mr, W, [0])
                            Ks[(th, v)][0].append(K)
                            Ks[(th, v)][1].append(M)
                            Ks[(th, v)][2].append(b)
                for (th, v), (kl, ml, bl_) in Ks.items():
                    bl_ = np.array(bl_)
                    sc = {"count": np.zeros(len(kl)), "pattern": np.zeros(len(kl))}
                    for b in np.unique(bl_):  # group trials by host block (local baselines differ per block)
                        idx = np.where(bl_ == b)[0]
                        K = np.concatenate([kl[i] for i in idx])
                        M = None if ml[0] is None else np.concatenate([ml[i] for i in idx])
                        for k, x in scores_for(K, M, W, bank.get(v, int(b))).items():
                            sc[k][idx] = x
                    rows[(th, v)][W] = sc
            for th in thetas:
                # theory: per-shot divergences at known location, averaged over a grid of placements
                corners = [(t, q) for t in np.linspace(0, p0.shape[0] - T, 4).round().astype(int)
                           for q in np.linspace(0, p0.shape[1] - R, 4).round().astype(int)]
                mp = [marked_poisson_divergences(p0, planted_rates(p0, R, T, th, c)) for c in corners]
                be = [bernoulli_divergences(p0, planted_rates(p0, R, T, th, c)) for c in corners]
                delta = float(R * T * th * (1 - 2 * p0).mean())
                theory = dict(D_pattern_M=float(np.mean([x["D_pattern"] for x in mp])),
                              D_count_M=float(np.mean([x["D_count"] for x in mp])),
                              D_pattern_bernoulli=float(np.mean([x["D_pattern"] for x in be])),
                              D_count_bernoulli=float(np.mean([x["D_count"] for x in be])),
                              D_count_gauss_empirical_var=delta ** 2 / (2 * braw.count_var),
                              extra_events_per_shot=delta)
                theory["ratio_M"] = theory["D_pattern_M"] / theory["D_count_M"]
                theory["ratio_bernoulli"] = theory["D_pattern_bernoulli"] / theory["D_count_bernoulli"]
                theory["ratio_bernoulli_pattern_over_empirical_count"] = (theory["D_pattern_bernoulli"]
                                                                          / theory["D_count_gauss_empirical_var"])
                for f in (1e-3, 1e-2):
                    theory[f"stein_W_pattern_far{f:g}"] = float(np.log(1 / f) / theory["D_pattern_bernoulli"])
                    theory[f"stein_W_count_far{f:g}"] = float(np.log(1 / f) / theory["D_count_gauss_empirical_var"])
                res = dict(R=R, T=T, theta=th, theory=theory, variants={})
                for v in VARIANTS:
                    vv = {}
                    for k in ("count", "pattern"):
                        power = {f: [float((rows[(th, v)][W][k] > cal[W]["thresholds"][v][k][f]).mean()) for W in ws]
                                 for f in FARS}
                        nul1 = np.sort(nulls[1][v][k])
                        auc1 = float(np.mean(np.searchsorted(nul1, rows[(th, v)][1][k], "left") / len(nul1))) if 1 in ws else None
                        vv[k] = dict(power={f"far{f:g}": pw for f, pw in power.items()}, auc_W1=auc1,
                                     W_star={f"far{f:g}_power{t}": w_star(ws, power[f], t) for f in (1e-3, 1e-2)
                                             for t in (0.5, 0.9)},
                                     median_score=[float(np.median(rows[(th, v)][W][k])) for W in ws])
                    vv["ratio_W_star_count_over_pattern"] = {
                        key: (vv["count"]["W_star"][key] / vv["pattern"]["W_star"][key]
                              if vv["count"]["W_star"][key] and vv["pattern"]["W_star"][key] else None)
                        for key in vv["count"]["W_star"]}
                    vv["lower_bound_ratio_if_count_never_reached"] = {
                        key: (ws[-1] / vv["pattern"]["W_star"][key]
                              if vv["count"]["W_star"][key] is None and vv["pattern"]["W_star"][key] else None)
                        for key in vv["count"]["W_star"]}
                    res["variants"][v] = vv
                planted.append(res)
                key = "far0.001_power0.5"
                log(f"R={R} T={T} th={th}: W*(1e-3, power 0.5) count/pattern " + " | ".join(
                    f"{v} {res['variants'][v]['count']['W_star'][key]}/{res['variants'][v]['pattern']['W_star'][key]}"
                    for v in VARIANTS) + f" | theory ratio {theory['ratio_bernoulli']:.0f} "
                    f"(vs empirical-var count {theory['ratio_bernoulli_pattern_over_empirical_count']:.0f})")
            log(f"  (R={R}, T={T}) {time.time() - tconf:.0f}s")
    out["planted"] = dict(Ws=ws, trials=trials, results=planted)

    # ---------------------------------------------------------------- (5) search for localized events
    if not a.quick:
        W, stride = 64, 32
        starts = np.arange(0, SHOTS - W + 1, stride)
        starts = starts[starts >= REF]
        v = "local_masked"
        cs, ps = [], []
        for lo in range(0, len(starts), 128):
            st = starts[lo:lo + 128]
            g = r.grid(slice(int(st[0]), int(st[-1]) + W), L)
            K, M = counts_and_trials(g, VARIANTS[v][0], W, st - st[0])
            sc = scores_for(K, M, W, bank.get(v, int(st[0])))
            cs.append(sc["count"])
            ps.append(sc["pattern"])
        cs, ps = np.concatenate(cs), np.concatenate(ps)
        inside = np.array([any(lo_ <= s_ < hi_ for lo_, hi_ in excl) for s_ in starts])
        th = cal[W]["thresholds"][v]
        srch = dict(W=W, stride=stride, variant=v, windows=int(len(starts)), outside_exclusions=int((~inside).sum()))
        for f in (1e-3, 1e-2):
            pa, ca = ps > th["pattern"][f], cs > th["count"][f]
            srch[f"far{f:g}"] = dict(expected_false_alarms_outside=float(f * (~inside).sum()),
                                     pattern_only=int((pa & ~ca & ~inside).sum()), count_only=int((ca & ~pa & ~inside).sum()),
                                     both=int((pa & ca & ~inside).sum()))
        top = np.argsort(np.where(inside, -np.inf, ps / th["pattern"][1e-3]))[::-1][:10]
        srch["top_pattern_outside"] = [dict(start=int(starts[i]), pattern=float(ps[i]), count_z=float(cs[i]),
                                            pattern_over_thr=float(ps[i] / th["pattern"][1e-3]),
                                            count_over_thr=float(cs[i] / th["count"][1e-3])) for i in top]
        out["search"] = srch
        log(f"search: {srch['far0.001']}")

    out["meta"]["seconds"] = round(time.time() - t0, 1)
    out["meta"]["peak_rss_MB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    if not a.quick:
        path = REPO / "results" / "burst_detection.json"
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        log("wrote", path.relative_to(REPO))
    log(f"done ({time.time() - t0:.0f}s, peak RSS {out['meta']['peak_rss_MB']} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
