"""The unstructured baseline: segment first, then read off the parameters.

Fit the two-component gap mixture, hard-assign every gap to fast or slow, cut
the token stream at every slow gap, and recover the decode parameters from the
resulting burst lengths: gamma from the longest burst seen, alpha by matching
the mean burst length to E[K].

This is what an analyst does when they have the right emission model but no
model of how bursts are generated. It is the strong form of the naive approach:
the split point comes from a fitted mixture rather than a hand-picked constant,
and censored bursts at the two ends of the trace are dropped. What it cannot do
is defend itself against a single merged or split burst, because the longest
burst is a maximum over a hundred of them and the mean is pulled with it.
"""

from __future__ import annotations

import numpy as np

from .burst import invert_expected_burst
from .censored import censoring_mask
from .mixture import fit_lognormal_mixture, single_component_loglik

MIN_GAP_S = 1e-7


def segment_bursts(gaps, threshold_s: float, drop_censored: bool = True) -> np.ndarray:
    """Token counts per burst implied by cutting wherever a gap exceeds threshold."""
    lengths, run = [], 1
    for g in np.asarray(gaps, dtype=float):
        if g > threshold_s:
            lengths.append(run)
            run = 1
        else:
            run += 1
    lengths.append(run)
    if drop_censored and len(lengths) > 2:
        lengths = lengths[1:-1]
    return np.array(lengths, dtype=int)


def baseline_estimate(
    arrivals,
    gamma_max: int = 6,
    oracle_gamma: int | None = None,
    burst_quantile: float | None = None,
) -> dict:
    """Mixture-threshold plus method-of-moments estimate of (gamma, alpha).

    `burst_quantile` replaces the longest observed burst with a high quantile of
    the burst lengths, which is the obvious way to harden the gamma estimate
    against a single merged burst. `oracle_gamma` hands the estimator the true
    draft length so that only alpha is being judged.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    gaps = np.diff(arrivals)
    censored, logg, log_res = censoring_mask(gaps)
    if gaps.size < 8:
        return {
            "gamma": 0, "alpha": 0.0, "threshold_s": 0.0, "fast_weight": 0.0,
            "mean_burst": 1.0, "max_burst": 1, "step_ms": float("nan"),
            "mixture_gain": 0.0,
        }

    fit = fit_lognormal_mixture(logg, censored, log_res)
    thr = float(np.exp(fit["threshold"]))
    lengths = segment_bursts(gaps, thr)
    mean_burst = float(lengths.mean()) if lengths.size else 1.0
    max_burst = int(lengths.max()) if lengths.size else 1

    if burst_quantile is not None and lengths.size:
        top = int(np.ceil(np.quantile(lengths, burst_quantile)))
    else:
        top = max_burst
    gamma_hat = int(np.clip(top - 1, 0, gamma_max))
    if oracle_gamma is not None:
        gamma_hat = int(oracle_gamma)
    alpha_hat = invert_expected_burst(mean_burst, gamma_hat) if gamma_hat >= 1 else 0.0

    slow = gaps[gaps > thr]
    return {
        "gamma": gamma_hat,
        "alpha": float(alpha_hat),
        "threshold_s": thr,
        "fast_weight": float(1.0 - fit["pi_b"]),
        "mean_burst": mean_burst,
        "max_burst": max_burst,
        "top_burst": int(top),
        "step_ms": float(np.median(slow) * 1e3) if slow.size else float(np.median(gaps) * 1e3),
        "mixture_gain": float(fit["loglik"] - single_component_loglik(logg, censored, log_res)),
    }
