"""Unstructured two-component lognormal mixture over inter-token gaps.

This is the shared front end. Both the baseline and the structured estimator see
the same emission model, the same censoring treatment and the same EM
machinery, so the comparison between them isolates one thing: whether the burst
run-length law is imposed or not.

A single global gap threshold placed by Otsu or by 1D k-means is not good enough
to serve as the baseline, because frame coalescing puts a censored atom at a gap
of zero and both criteria then split off that atom instead of finding the real
fast/slow boundary. Fitting a censored mixture with a separation constraint and
several starts is the strong form of the unstructured approach.
"""

from __future__ import annotations

import numpy as np

from .censored import SIGMA_FLOOR, censored_moments, log_component

MIN_SEPARATION = 0.35
INIT_QUANTILES = (0.45, 0.6, 0.75, 0.88)


def fit_single_component(logg, censored, log_res, n_iter: int = 80):
    """Censored maximum likelihood for one lognormal component."""
    mu, sigma = float(logg.mean()), max(float(logg.std()), SIGMA_FLOOR)
    w = np.ones_like(logg)
    for _ in range(n_iter):
        mu, sigma = censored_moments(logg, censored, log_res, w, mu, sigma, mu, sigma)
    return mu, sigma, float(log_component(logg, censored, log_res, mu, sigma).sum())


def single_component_loglik(logg, censored, log_res) -> float:
    return fit_single_component(logg, censored, log_res)[2]


def _em(logg, censored, log_res, mu_w, sigma_w, mu_b, sigma_b, pi_b, max_iter, tol):
    prev, ll = -np.inf, -np.inf
    rb = None
    for _ in range(max_iter):
        lw = np.log(max(1.0 - pi_b, 1e-9)) + log_component(logg, censored, log_res, mu_w, sigma_w)
        lb = np.log(max(pi_b, 1e-9)) + log_component(logg, censored, log_res, mu_b, sigma_b)
        tot = np.logaddexp(lw, lb)
        ll = float(tot.sum())
        if ll - prev < tol * max(1.0, abs(prev)):
            break
        prev = ll

        rb = np.exp(lb - tot)
        rw = 1.0 - rb
        pi_b = float(np.clip(rb.sum() / logg.size, 1e-4, 1.0 - 1e-4))
        mu_b, sigma_b = censored_moments(logg, censored, log_res, rb, mu_b, sigma_b, mu_b, sigma_b)
        mu_w, sigma_w = censored_moments(logg, censored, log_res, rw, mu_w, sigma_w, mu_w, sigma_w)
        if mu_w > mu_b - MIN_SEPARATION:
            mu_w = mu_b - MIN_SEPARATION
    return dict(
        mu_w=mu_w, sigma_w=sigma_w, mu_b=mu_b, sigma_b=sigma_b, pi_b=pi_b, loglik=ll,
        post_between=rb if rb is not None else np.full(logg.size, 0.5),
    )


def fit_lognormal_mixture(logg, censored, log_res, max_iter=200, tol=1e-10) -> dict:
    """Best of several starts, each seeded by a candidate fast/slow split point."""
    logg = np.asarray(logg, dtype=float)
    best = None
    for q in INIT_QUANTILES:
        cut = float(np.quantile(logg, q))
        lo, hi = logg[logg <= cut], logg[logg > cut]
        if lo.size < 2 or hi.size < 2:
            continue
        cand = _em(
            logg, censored, log_res,
            float(lo.mean()), max(float(lo.std()), SIGMA_FLOOR),
            float(hi.mean()), max(float(hi.std()), SIGMA_FLOOR),
            float(hi.size) / logg.size, max_iter, tol,
        )
        if best is None or cand["loglik"] > best["loglik"]:
            best = cand
    if best is None:
        mu, sigma = float(logg.mean()), max(float(logg.std()), SIGMA_FLOOR)
        best = dict(
            mu_w=mu - MIN_SEPARATION, sigma_w=sigma, mu_b=mu, sigma_b=sigma, pi_b=0.5,
            loglik=single_component_loglik(logg, censored, log_res),
            post_between=np.full(logg.size, 0.5),
        )
    best["threshold"] = _crossing(best)
    return best


def _crossing(fit) -> float:
    """Log-gap value where the slow component's posterior passes one half."""
    grid = np.linspace(fit["mu_w"], fit["mu_b"], 512)
    cen = np.zeros(grid.size, dtype=bool)
    lw = np.log(max(1.0 - fit["pi_b"], 1e-9)) + log_component(grid, cen, -np.inf, fit["mu_w"], fit["sigma_w"])
    lb = np.log(max(fit["pi_b"], 1e-9)) + log_component(grid, cen, -np.inf, fit["mu_b"], fit["sigma_b"])
    above = np.nonzero(lb >= lw)[0]
    return float(grid[above[0]]) if above.size else float(0.5 * (fit["mu_w"] + fit["mu_b"]))
