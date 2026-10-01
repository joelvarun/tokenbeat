"""Burst-length law induced by draft-and-verify speculative decoding.

One verification step drafts `gamma` tokens, verifies them in a single target
forward pass, accepts a prefix of them, and always emits one extra corrected
token. Under the standard i.i.d. acceptance model with per-draft acceptance
probability `alpha`, the number of tokens K emitted by one step is supported on
1..gamma+1 with a geometric body and an atom at the cap.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def burst_pmf(alpha: float, gamma: int) -> np.ndarray:
    """P(K = k) for k = 1..gamma+1."""
    if gamma == 0:
        return np.array([1.0])
    alpha = float(np.clip(alpha, EPS, 1.0 - EPS))
    k = np.arange(1, gamma + 1)
    pmf = np.empty(gamma + 1)
    pmf[:gamma] = (1.0 - alpha) * alpha ** (k - 1)
    pmf[gamma] = alpha**gamma
    return pmf


def burst_survival(alpha: float, gamma: int) -> np.ndarray:
    """P(K >= j) for j = 1..gamma+1, which collapses to alpha ** (j - 1)."""
    alpha = float(np.clip(alpha, EPS, 1.0 - EPS))
    j = np.arange(1, gamma + 2)
    return alpha ** (j - 1)


def expected_burst(alpha: float, gamma: int) -> float:
    """E[K] = (1 - alpha ** (gamma + 1)) / (1 - alpha)."""
    if gamma == 0:
        return 1.0
    alpha = float(np.clip(alpha, EPS, 1.0 - EPS))
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def position_stationary(alpha: float, gamma: int) -> np.ndarray:
    """Distribution of within-burst position for a uniformly chosen token.

    A renewal argument gives P(position = j) proportional to P(K >= j), so the
    normaliser is exactly E[K].
    """
    surv = burst_survival(alpha, gamma)
    return surv / surv.sum()


def invert_expected_burst(mean_k: float, gamma: int) -> float:
    """Solve E[K](alpha, gamma) = mean_k for alpha by bisection.

    E[K] is strictly increasing in alpha on (0, 1), so bisection is exact to
    machine tolerance. Used by the moment-matching baseline.
    """
    if gamma <= 0:
        return 0.0
    mean_k = float(np.clip(mean_k, 1.0 + 1e-9, gamma + 1 - 1e-9))
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_burst(mid, gamma) < mean_k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tokens_per_forward_pass(alpha: float, gamma: int) -> float:
    """Alias for E[K]: the speedup speculative decoding buys per target pass."""
    return expected_burst(alpha, gamma)
