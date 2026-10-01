"""Generative model of a streaming LLM response as seen by a client.

The server runs verification steps. Each step emits K tokens, where K follows
the speculative burst law, and takes a step time that jumps whenever batch
occupancy changes as other tenants come and go. Tokens inside one step are
flushed back to back. Every frame then picks up a variable one-way network
delay: usually sub-millisecond, but a small fraction of frames are hit by a
queueing delay worth a couple of verification steps. TCP in-order delivery means such a
frame also holds back the frame behind it, so the gap that follows a long delay
collapses toward zero. Those two artefacts, a merged burst and a phantom burst,
are what a single gap threshold cannot survive.

The jitter and flush parameters default to values measured over a real loopback
SSE stream by `tokenbeat.harness`, loaded from docs/calibration.json when it
exists.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from .burst import burst_pmf

MIN_SERIALISATION_S = 2e-6

_DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
_CALIB_PATH = os.path.join(_DOCS, "calibration.json")
_CHANNEL_PATH = os.path.join(_DOCS, "channel.npz")


@dataclass
class StackConfig:
    """Hidden serving-stack parameters a client does not get told."""

    alpha: float = 0.75
    gamma: int = 3
    step_ms: float = 28.0
    step_log_sigma: float = 0.07
    jump_prob: float = 0.012
    jump_log_sigma: float = 0.22
    flush_log_mean: float = float(np.log(4e-4))
    flush_log_sigma: float = 0.6
    net_log_mean: float = float(np.log(8e-4))
    net_log_sigma: float = 0.5
    contam_prob: float = 0.035
    contam_median_steps: float = 2.0
    contam_log_sigma: float = 0.8
    jitter_scale: float = 1.0

    def replace(self, **kw) -> "StackConfig":
        d = asdict(self)
        d.update(kw)
        return StackConfig(**d)


def load_calibration(path: str | None = None) -> dict:
    """Measured loopback timing parameters, if the harness has been run."""
    path = path or _CALIB_PATH
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def load_flush_pool(path: str | None = None):
    """Measured intra-step gaps, for resampling instead of assuming a shape.

    The flush distribution is bounded by the server's send loop and has an atom
    at zero from frame coalescing, so no lognormal fits its tail honestly.
    Bootstrapping the measured gaps keeps the sweep on a real channel.
    """
    path = path or _CHANNEL_PATH
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return z["flush_pool"]


def calibrated_config(alpha: float, gamma: int, jitter_scale: float = 1.0, **kw) -> StackConfig:
    """Build a StackConfig whose channel parameters come from real measurements."""
    calib = load_calibration()
    fields = {"alpha": alpha, "gamma": gamma, "jitter_scale": jitter_scale}
    for key in (
        "flush_log_mean", "flush_log_sigma", "net_log_mean", "net_log_sigma",
        "step_log_sigma", "jitter_scale", "contam_median_steps", "contam_log_sigma",
    ):
        if key in calib:
            fields[key] = calib[key]
    fields.update(kw)
    return StackConfig(**fields)


def draw_burst_lengths(alpha: float, gamma: int, n_tokens: int, rng: np.random.Generator):
    """Draw step burst lengths until they cover n_tokens."""
    if gamma == 0:
        return np.ones(n_tokens, dtype=int)
    pmf = burst_pmf(alpha, gamma)
    support = np.arange(1, gamma + 2)
    out, total = [], 0
    while total < n_tokens:
        block = rng.choice(support, size=max(16, (n_tokens - total) // 2 + 1), p=pmf)
        for k in block:
            out.append(int(k))
            total += int(k)
            if total >= n_tokens:
                break
    return np.array(out, dtype=int)


def simulate_trace(
    cfg: StackConfig, n_tokens: int, rng: np.random.Generator, flush_pool=None
) -> dict:
    """Return client-side arrival timestamps plus the ground truth that made them."""
    lens = draw_burst_lengths(cfg.alpha, cfg.gamma, n_tokens, rng)
    if flush_pool is None:
        flush_pool = load_flush_pool()

    send, burst_id, position = [], [], []
    clock = 0.0
    log_step = np.log(cfg.step_ms * 1e-3)
    for bi, k in enumerate(lens):
        if rng.random() < cfg.jump_prob:
            log_step += float(rng.normal(0.0, cfg.jump_log_sigma))
        clock += float(np.exp(log_step + rng.normal(0.0, cfg.step_log_sigma)))
        send.append(clock)
        burst_id.append(bi)
        position.append(0)
        for pos in range(1, int(k)):
            if flush_pool is not None:
                clock += float(flush_pool[rng.integers(flush_pool.size)])
            else:
                clock += float(
                    np.exp(cfg.flush_log_mean + rng.normal(0.0, cfg.flush_log_sigma))
                )
            send.append(clock)
            burst_id.append(bi)
            position.append(pos)

    send = np.array(send[:n_tokens])
    burst_id = np.array(burst_id[:n_tokens])
    position = np.array(position[:n_tokens])

    delay = cfg.jitter_scale * np.exp(
        cfg.net_log_mean + rng.normal(0.0, cfg.net_log_sigma, size=send.size)
    )
    hit = rng.random(send.size) < cfg.contam_prob
    if hit.any():
        contam_median = cfg.contam_median_steps * cfg.step_ms * 1e-3
        delay[hit] = contam_median * np.exp(
            rng.normal(0.0, cfg.contam_log_sigma, size=int(hit.sum()))
        )

    arrivals = np.empty(send.size)
    prev = -np.inf
    for i in range(send.size):
        t = send[i] + delay[i]
        arrivals[i] = max(t, prev + MIN_SERIALISATION_S)
        prev = arrivals[i]

    return {
        "arrivals": arrivals - arrivals[0],
        "send": send - send[0],
        "burst_id": burst_id,
        "position": position,
        "contaminated": hit,
        "truth": {"alpha": cfg.alpha, "gamma": cfg.gamma, "step_ms": cfg.step_ms},
    }
