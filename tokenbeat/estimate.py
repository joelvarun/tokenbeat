"""Public entry point: fit the burst model at every draft length and pick one.

Model selection by BIC over gamma in 0..gamma_max does three jobs at once. It
decides whether the server speculates at all (gamma = 0 is the no-speculation
model, in the same family and scored on the same likelihood), it reads off the
draft length without using a biased order statistic, and it reports a Bayes
factor instead of asking the caller for a detection threshold.

The reported `speculative` flag also carries an effect-size gate. A path that
delivers isolated long delays produces a handful of near-zero gaps behind them,
and BIC will buy a one-token draft to explain those; the fitted acceptance rate
then comes out near zero, so requiring at least 1.15 tokens per target pass
separates a server that really speculates from a path artefact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baseline import baseline_estimate
from .censored import censoring_mask
from .mixture import fit_lognormal_mixture
from .burst import expected_burst
from .hsmm import HsmmFit, fit_hsmm

LOG10 = float(np.log(10.0))
MIN_EFFECT_TOKENS_PER_STEP = 1.15


@dataclass
class BeatReport:
    speculative: bool
    gamma: int
    alpha: float
    tokens_per_step: float
    flush_ms: float
    step_ms: float
    step_ms_by_window: np.ndarray = field(repr=False)
    log10_bayes_factor: float = 0.0
    bic_table: dict = field(default_factory=dict, repr=False)
    n_tokens: int = 0
    anomaly_rate: float = 0.0
    fit: HsmmFit | None = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"tokens observed          {self.n_tokens}",
            f"speculative decoding     {'yes' if self.speculative else 'no'}"
            f"  (log10 Bayes factor {self.log10_bayes_factor:+.1f})",
        ]
        if self.speculative:
            lines += [
                f"draft length gamma       {self.gamma}",
                f"acceptance rate alpha    {self.alpha:.3f}",
                f"tokens per target pass   {self.tokens_per_step:.2f}",
                f"intra-step flush         {self.flush_ms:.3f} ms",
            ]
        lines.append(f"verification step time   {self.step_ms:.2f} ms")
        drift = self.step_ms_by_window
        if drift.size > 1:
            lines.append(
                f"step time range          {drift.min():.2f} to {drift.max():.2f} ms"
                f" over {drift.size} windows"
            )
        return "\n".join(lines)


def estimate_anomaly_rate(gaps, threshold_s: float, factor: float = 1.8) -> float:
    """Fraction of gaps touched by a queueing delay, counted before any fitting.

    A frame held up in a queue inflates its own gap and, through in-order
    delivery, collapses the gap behind it, so one delayed frame costs two gaps.
    The step level has to come from the median of the slow gaps rather than from
    a fitted location: at high delay rates the fitted slow component absorbs the
    delays themselves and a threshold built on it stops firing. The floor of
    5 percent is what a quiet path needs anyway, and a noisier path raises it.
    """
    gaps = np.asarray(gaps, dtype=float)
    slow = gaps[gaps > threshold_s]
    level = float(np.median(slow)) if slow.size >= 4 else float(np.median(gaps))
    delayed = float(np.mean(gaps > factor * level))
    return float(np.clip(2.0 * delayed, 0.05, 0.35))


def analyze(arrivals, gamma_max: int = 6, window: int = 64, max_iter: int = 40) -> BeatReport:
    """Recover hidden decode parameters from client-side token arrival times."""
    arrivals = np.asarray(arrivals, dtype=float)
    if arrivals.size < 12:
        raise ValueError("need at least 12 token timestamps")
    gaps = np.diff(arrivals)
    censored, logg, log_res = censoring_mask(gaps)

    mix = fit_lognormal_mixture(logg, censored, log_res)
    init = {
        "alpha": float(np.clip(1.0 - mix["pi_b"], 0.1, 0.9)),
        "mu_w": mix["mu_w"], "sigma_w": mix["sigma_w"],
        "mu_b": mix["mu_b"], "sigma_b": mix["sigma_b"],
    }

    anomaly_rate = estimate_anomaly_rate(gaps, float(np.exp(mix["threshold"])))
    fits = {
        g: fit_hsmm(
            gaps, g, window=window, max_iter=max_iter, init=init,
            anomaly_rate=anomaly_rate,
        )
        for g in range(0, gamma_max + 1)
    }

    best = min(fits.values(), key=lambda f: f.bic)
    bic_table = {g: f.bic for g, f in fits.items()}
    lbf = (fits[0].bic - best.bic) / (2.0 * LOG10)
    tps = expected_burst(best.alpha, best.gamma)

    speculative = bool(best.gamma >= 1 and tps >= MIN_EFFECT_TOKENS_PER_STEP)
    step_by_window = best.step_seconds * 1e3
    return BeatReport(
        speculative=speculative,
        gamma=best.gamma if speculative else 0,
        alpha=best.alpha if speculative else 0.0,
        tokens_per_step=tps if speculative else 1.0,
        flush_ms=best.flush_seconds * 1e3 if speculative else float("nan"),
        step_ms=float(np.median(step_by_window)),
        step_ms_by_window=step_by_window,
        log10_bayes_factor=float(lbf),
        bic_table=bic_table,
        n_tokens=int(arrivals.size),
        anomaly_rate=float(anomaly_rate),
        fit=best,
    )
