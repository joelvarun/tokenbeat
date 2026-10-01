"""Measure both estimators, first on real loopback streams, then across a sweep.

Phase 1 opens a real TCP connection per trace and times the frames as they
arrive, so the headline numbers are measured rather than modelled. Phase 2
scores every estimator on those traces. Phase 3 calibrates the simulator to the
measured channel and sweeps the path contamination rate, which is the one axis
the loopback grid cannot cover cheaply.

Writes results/benchmark.json and docs/calibration.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenbeat.baseline import baseline_estimate
from tokenbeat.burst import burst_pmf
from tokenbeat.censored import censoring_mask
from tokenbeat.estimate import analyze
from tokenbeat.harness import run_stream
from tokenbeat.mixture import fit_single_component
from tokenbeat.simulate import StackConfig, calibrated_config, simulate_trace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_STEP_MS = 12.0
SPEC_GRID = [(a, g) for a in (0.5, 0.65, 0.8, 0.9) for g in (2, 3, 5)]
SPEC_REPS = 3
NONSPEC_REPS = 12
N_TOKENS = 300
N_TOKENS_NONSPEC = 200
CONTAM_LEVELS = (0.0, 0.02, 0.04, 0.08)
SWEEP_GRID = [(0.5, 2), (0.65, 3), (0.8, 3), (0.9, 5)]
SWEEP_REPS = 6
GAMMA_MAX = 6
METHODS = ("baseline", "baseline_q99", "baseline_oracle_gamma", "tokenbeat")


def tv_distance(a_hat, g_hat, a_true, g_true) -> float:
    """Total variation between the implied and the true burst-length laws."""
    p, q = burst_pmf(a_hat, g_hat), burst_pmf(a_true, g_true)
    n = max(p.size, q.size)
    pp, qq = np.zeros(n), np.zeros(n)
    pp[: p.size], qq[: q.size] = p, q
    return float(0.5 * np.abs(pp - qq).sum())


def score_trace(arrivals, truth) -> dict:
    a_true, g_true = truth["alpha"], truth["gamma"]
    rep = analyze(arrivals, gamma_max=GAMMA_MAX)
    base = baseline_estimate(arrivals, gamma_max=GAMMA_MAX)
    base_q = baseline_estimate(arrivals, gamma_max=GAMMA_MAX, burst_quantile=0.99)
    base_o = baseline_estimate(arrivals, gamma_max=GAMMA_MAX, oracle_gamma=g_true)

    out = {"truth": {"alpha": a_true, "gamma": g_true, "step_ms": truth["step_ms"]}}
    out["tokenbeat"] = {
        "alpha": rep.alpha, "gamma": rep.gamma,
        "spec": bool(rep.speculative), "step_ms": rep.step_ms,
    }
    for name, est in (
        ("baseline", base), ("baseline_q99", base_q), ("baseline_oracle_gamma", base_o)
    ):
        out[name] = {
            "alpha": est["alpha"], "gamma": est["gamma"],
            "spec": bool(est["gamma"] >= 1), "step_ms": est["step_ms"],
        }
    for name in METHODS:
        est = out[name]
        est["alpha_err"] = abs(est["alpha"] - a_true)
        est["gamma_exact"] = int(est["gamma"] == g_true)
        est["tv"] = tv_distance(est["alpha"], est["gamma"], a_true, g_true)
        est["step_ape"] = abs(est["step_ms"] - truth["step_ms"]) / truth["step_ms"]
    return out


def collect_loopback() -> list:
    rng = np.random.default_rng(20260401)
    records, t0 = [], time.perf_counter()
    jobs = [(a, g, N_TOKENS) for (a, g) in SPEC_GRID for _ in range(SPEC_REPS)]
    jobs += [(0.0, 0, N_TOKENS_NONSPEC) for _ in range(NONSPEC_REPS)]
    for i, (a, g, n) in enumerate(jobs, 1):
        cfg = StackConfig(alpha=a, gamma=g, step_ms=HARNESS_STEP_MS, contam_prob=0.035)
        trace = run_stream(cfg, n, rng)
        records.append(trace)
        print(
            "  [%d/%d] alpha=%.2f gamma=%d tokens=%d elapsed=%.0fs"
            % (i, len(jobs), a, g, trace["arrivals"].size, time.perf_counter() - t0),
            flush=True,
        )
    return records


TRACE_CACHE = os.path.join(ROOT, "results", "loopback_traces.npz")


def _detrend(values, block: int = 10):
    """Remove the local level so step-time jumps do not inflate the dispersion."""
    out = np.empty_like(values)
    for start in range(0, values.size, block):
        chunk = values[start : start + block]
        out[start : start + chunk.size] = chunk - np.median(chunk)
    return out


def _robust_scale(values) -> float:
    """MAD-based scale: a step-time jump inside one block must not dominate."""
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def save_traces(records, path: str = TRACE_CACHE) -> None:
    flat = {}
    for i, tr in enumerate(records):
        flat["arrivals_%d" % i] = tr["arrivals"]
        flat["position_%d" % i] = tr["position"]
        flat["contaminated_%d" % i] = tr["contaminated"]
        flat["truth_%d" % i] = np.array(
            [tr["truth"]["alpha"], tr["truth"]["gamma"], tr["truth"]["step_ms"]]
        )
    np.savez_compressed(path, n=np.array([len(records)]), **flat)


def load_traces(path: str = TRACE_CACHE):
    with np.load(path) as z:
        n = int(z["n"][0])
        out = []
        for i in range(n):
            t = z["truth_%d" % i]
            out.append({
                "arrivals": z["arrivals_%d" % i],
                "position": z["position_%d" % i],
                "contaminated": z["contaminated_%d" % i],
                "truth": {"alpha": float(t[0]), "gamma": int(t[1]), "step_ms": float(t[2])},
            })
    return out


def calibrate(records) -> dict:
    """Measure the realised channel from traces whose burst structure is known.

    Contaminated frames and the frame directly behind one are dropped, so what
    is left describes the clean path, and the two noise sources are kept apart.
    Per-frame delay variation shows up in the gaps between tokens of one step,
    so the measured pool of those gaps carries it already and gets resampled
    whole. What is left in the burst-opening gaps is step-time variation, which
    moves step boundaries but not the frames inside a step, and that becomes the
    simulator's relative step dispersion. Folding both into one per-frame jitter
    term, as an earlier version did, counts the same noise twice and smears a
    structure that real streams keep intact.
    """
    flush, jitter_std, coalesced, pool = [], [], [], []
    for tr in records:
        gaps = np.diff(tr["arrivals"])
        pos = tr["position"][1 : gaps.size + 1]
        cont = tr["contaminated"][1 : gaps.size + 1]
        prev_cont = tr["contaminated"][: gaps.size]
        clean = ~cont & ~prev_cont
        coalesced.append(float((gaps <= 1e-5).mean()))

        w = gaps[(pos > 0) & clean]
        if w.size > 8:
            cen, logg, log_res = censoring_mask(w)
            mu, sigma, _ = fit_single_component(logg, cen, log_res)
            flush.append((mu, sigma))
            pool.append(w)
        b = gaps[(pos == 0) & clean]
        if b.size > 8:
            jitter_std.append(_robust_scale(_detrend(b)))

    flush_pool = np.concatenate(pool)
    np.savez_compressed(os.path.join(ROOT, "docs", "channel.npz"), flush_pool=flush_pool)
    gap_std = float(np.median(jitter_std))
    return {
        "measured_on": "loopback SSE stream, real sockets and real arrival timestamps",
        "n_traces": len(records),
        "coalesced_gap_fraction": float(np.mean(coalesced)),
        "clean_gap_dispersion_ms": gap_std * 1e3,
        "flush_pool_size": int(flush_pool.size),
        "flush_pool_median_ms": float(np.median(flush_pool) * 1e3),
        "flush_log_mean": float(np.median([m for m, _ in flush])),
        "flush_log_sigma": float(np.median([sg for _, sg in flush])),
        "step_log_sigma": gap_std / (HARNESS_STEP_MS * 1e-3),
        "jitter_scale": 0.0,
        "contam_median_steps": 2.0,
        "contam_log_sigma": 0.8,
    }


def aggregate(scored) -> dict:
    spec = [s for s in scored if s["truth"]["gamma"] >= 1]
    non = [s for s in scored if s["truth"]["gamma"] == 0]
    out = {"n_spec": len(spec), "n_nonspec": len(non), "methods": {}}
    for name in METHODS:
        out["methods"][name] = {
            "alpha_mae": float(np.mean([s[name]["alpha_err"] for s in spec])),
            "alpha_rmse": float(np.sqrt(np.mean([s[name]["alpha_err"] ** 2 for s in spec]))),
            "gamma_exact_pct": 100.0 * float(np.mean([s[name]["gamma_exact"] for s in spec])),
            "tv_mean": float(np.mean([s[name]["tv"] for s in spec])),
            "step_mape_pct": 100.0 * float(np.mean([s[name]["step_ape"] for s in spec])),
            "detect_tpr_pct": 100.0 * float(np.mean([s[name]["spec"] for s in spec])),
            "detect_fpr_pct": (
                100.0 * float(np.mean([s[name]["spec"] for s in non])) if non else 0.0
            ),
        }
    return out


def sweep() -> dict:
    rng = np.random.default_rng(777)
    rows = {}
    for cp in CONTAM_LEVELS:
        scored = []
        for a, g in SWEEP_GRID:
            cfg = calibrated_config(a, g, contam_prob=cp, step_ms=HARNESS_STEP_MS)
            for _ in range(SWEEP_REPS):
                tr = simulate_trace(cfg, N_TOKENS, rng)
                scored.append(score_trace(tr["arrivals"], tr["truth"]))
        rows["%.2f" % cp] = aggregate(scored)
        m = rows["%.2f" % cp]["methods"]
        print(
            "  contamination %4.1f%%: baseline alpha MAE %.3f, tokenbeat %.3f"
            % (100 * cp, m["baseline"]["alpha_mae"], m["tokenbeat"]["alpha_mae"]),
            flush=True,
        )
    return rows


def main() -> None:
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)

    refresh = "--refresh" in sys.argv or not os.path.exists(TRACE_CACHE)
    if refresh:
        print("phase 1: streaming real loopback traces", flush=True)
        records = collect_loopback()
        save_traces(records)
    else:
        records = load_traces()
        print("phase 1: reusing %d cached loopback traces" % len(records), flush=True)

    print("phase 1b: calibrating the simulator to the measured channel", flush=True)
    calib = calibrate(records)
    with open(os.path.join(ROOT, "docs", "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
    print(
        "  coalesced gaps %.1f%%, clean dispersion %.3f ms, flush median %.3f ms"
        % (
            100 * calib["coalesced_gap_fraction"],
            calib["clean_gap_dispersion_ms"],
            np.exp(calib["flush_log_mean"]) * 1e3,
        ),
        flush=True,
    )

    print("phase 2: scoring estimators on the real traces", flush=True)
    scored = [score_trace(tr["arrivals"], tr["truth"]) for tr in records]
    loopback = aggregate(scored)
    for name, m in loopback["methods"].items():
        print(
            "  %-24s alpha MAE %.3f  gamma exact %5.1f%%  TV %.3f  FPR %5.1f%%"
            % (name, m["alpha_mae"], m["gamma_exact_pct"], m["tv_mean"], m["detect_fpr_pct"]),
            flush=True,
        )

    print("phase 3: contamination sweep on the calibrated simulator", flush=True)
    sweep_rows = sweep()

    payload = {
        "config": {
            "harness_step_ms": HARNESS_STEP_MS,
            "n_tokens": N_TOKENS,
            "n_tokens_nonspec": N_TOKENS_NONSPEC,
            "spec_grid": SPEC_GRID,
            "spec_reps": SPEC_REPS,
            "nonspec_reps": NONSPEC_REPS,
            "gamma_max": GAMMA_MAX,
            "contam_levels": list(CONTAM_LEVELS),
            "sweep_grid": SWEEP_GRID,
            "sweep_reps": SWEEP_REPS,
        },
        "calibration": calib,
        "loopback": loopback,
        "sweep": sweep_rows,
        "per_trace": scored,
    }
    path = os.path.join(ROOT, "results", "benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote %s" % path, flush=True)


if __name__ == "__main__":
    main()
