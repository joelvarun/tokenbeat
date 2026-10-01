"""Left-censored lognormal likelihood and EM moment updates.

When two SSE frames arrive inside one socket read the client stamps them with
the same clock reading, so the gap between them is recorded as zero. Those gaps
are not tiny measurements, they are observations censored at the client's timing
resolution, and treating them as ordinary samples drags the fast component's
log-scale parameters down by several nats and makes an unstructured mixture
prefer splitting the censored atom off instead of finding the fast/slow
boundary. Both estimators in this package share the correct treatment: a
censored gap contributes the component CDF at the resolution floor, and the
M-step imputes its conditional moments the Tobit way.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr

LOG2PI = float(np.log(2.0 * np.pi))
SIGMA_FLOOR = 0.12
RESOLUTION_S = 1e-5


def censoring_mask(gaps, resolution_s: float = RESOLUTION_S):
    """Gaps at or below the client's timing resolution are censored, not observed."""
    gaps = np.asarray(gaps, dtype=float)
    mask = gaps <= resolution_s
    safe = np.where(mask, resolution_s, gaps)
    return mask, np.log(safe), float(np.log(resolution_s))


def log_component(logg, censored, log_res, mu, sigma):
    """Per-observation log-likelihood of one lognormal component."""
    sigma = np.maximum(sigma, SIGMA_FLOOR)
    z = (logg - mu) / sigma
    dens = -logg - np.log(sigma) - 0.5 * LOG2PI - 0.5 * z * z
    tail = log_ndtr((log_res - mu) / sigma)
    return np.where(censored, tail, dens)


def _mills(z):
    """phi(z) / Phi(z), evaluated through log_ndtr so the left tail stays stable."""
    return np.exp(-0.5 * z * z - 0.5 * LOG2PI - log_ndtr(z))


def censored_moments(logg, censored, log_res, w, mu, sigma, fb_mu, fb_sigma):
    """Weighted Tobit M-step for a left-censored lognormal component."""
    tot = float(np.sum(w))
    if tot < 1e-8:
        return fb_mu, fb_sigma
    sigma = max(float(sigma), SIGMA_FLOOR)
    z = (log_res - mu) / sigma
    lam = _mills(z)
    imputed = mu - sigma * lam
    x = np.where(censored, imputed, logg)
    mu_new = float(np.sum(w * x) / tot)

    d = mu - mu_new
    v_cens = sigma * sigma * (1.0 - z * lam) - 2.0 * d * sigma * lam + d * d
    v = np.where(censored, v_cens, (logg - mu_new) ** 2)
    var = float(np.sum(w * np.maximum(v, 0.0)) / tot)
    return mu_new, max(float(np.sqrt(max(var, 0.0))), SIGMA_FLOOR)
