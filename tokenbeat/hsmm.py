"""Structured latent-variable fit of burst structure to an inter-token gap series.

The hidden state is a token's position inside its verification step. Because the
speculative burst law has survival P(K >= j) = alpha ** (j - 1), the position
chain is a truncated geometric run: continue the current burst with probability
alpha while position <= gamma, and restart with probability 1 at the cap. The
emission is the gap preceding the token, drawn from a fast "same step flush"
lognormal when the position is interior and a slow "next step" lognormal when
the token opens a burst.

Three features carry the method. The run-length constraint ties the
segmentation back to alpha and gamma, so a merged pair of bursts cannot simply
be reported as one long burst. A burst-opening token is allowed, at small fixed
probability, to arrive with a gap drawn from the fast distribution instead of
the slow one: that is the compressed boundary left behind when head-of-line
blocking eats a step gap, and without it the fit escapes by inflating gamma
until long merged bursts become legal. And every gap carries a fixed-weight
contamination component, because real paths deliver isolated long delays and the
near-zero gap that follows one is what makes a hard threshold hallucinate bursts
on a stream that has none.

Both allowances share one weight, set from a robust count of anomalously large
gaps before any model is fitted. It is a plug-in, not a fitted parameter, so
every candidate draft length is scored with the same parameter count and BIC
stays a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .censored import SIGMA_FLOOR, censored_moments, censoring_mask, log_component

MIN_SEPARATION = 0.35
DEFAULT_ANOMALY_RATE = 0.05
CONTAM_SIGMA_MULT = 2.5


@dataclass
class HsmmFit:
    gamma: int
    alpha: float
    mu_w: float
    sigma_w: float
    mu_b: np.ndarray
    sigma_b: float
    loglik: float
    n_params: int
    bic: float
    window: int
    win_idx: np.ndarray = field(repr=False)
    post_between: np.ndarray = field(repr=False)
    n_iter: int = 0

    @property
    def step_seconds(self) -> np.ndarray:
        return np.exp(self.mu_b)

    @property
    def flush_seconds(self) -> float:
        return float(np.exp(self.mu_w))


def _windows(n: int, window: int):
    idx = np.arange(n) // window
    n_win = int(idx.max()) + 1
    if n_win > 1 and int((idx == n_win - 1).sum()) < window // 2:
        idx[idx == n_win - 1] = n_win - 2
        n_win -= 1
    return idx, n_win


def _mix(core_log, contam_log, rho):
    """log of (1 - rho) * core + rho * contamination."""
    return np.logaddexp(np.log1p(-rho) + core_log, np.log(rho) + contam_log)


def _forward_backward(eB, eW, alpha, G):
    """Scaled forward-backward over the within-burst position chain."""
    n = eB.size
    restart = np.full(G, 1.0 - alpha)
    restart[G - 1] = 1.0
    pi = alpha ** np.arange(G)
    pi = pi / pi.sum()

    a = np.empty((n, G))
    scale = np.empty(n)
    first = np.empty(G)
    first[0] = eB[0]
    first[1:] = eW[0]
    cur = pi * first
    scale[0] = max(float(cur.sum()), 1e-300)
    a[0] = cur / scale[0]
    for i in range(1, n):
        prev = a[i - 1]
        cur = np.empty(G)
        cur[0] = eB[i] * float(prev @ restart)
        if G > 1:
            cur[1:] = eW[i] * prev[: G - 1] * alpha
        scale[i] = max(float(cur.sum()), 1e-300)
        a[i] = cur / scale[i]

    b = np.empty((n, G))
    b[n - 1] = 1.0
    xi_cont = np.zeros(n)
    xi_rest = np.zeros(n)
    for i in range(n - 1, 0, -1):
        nxt = b[i]
        cur = restart * (eB[i] * nxt[0])
        if G > 1:
            cur[: G - 1] += alpha * eW[i] * nxt[1:]
        b[i - 1] = cur / scale[i]
        if G > 1:
            prev = a[i - 1]
            xi_cont[i] = float((prev[: G - 1] * alpha * eW[i] * nxt[1:]).sum()) / scale[i]
            xi_rest[i] = (
                float(prev[: G - 1].sum()) * (1.0 - alpha) * eB[i] * nxt[0] / scale[i]
            )

    post = a * b
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
    return post, xi_cont, xi_rest, float(np.log(scale).sum())


def fit_hsmm(
    gaps, gamma, window=64, max_iter=40, tol=1e-7, init=None,
    anomaly_rate: float = DEFAULT_ANOMALY_RATE,
) -> HsmmFit:
    censored, logg, log_res = censoring_mask(gaps)
    n = logg.size
    win_idx, n_win = _windows(n, window)

    mu_c = float(logg.mean())
    sigma_c = max(CONTAM_SIGMA_MULT * float(logg.std()), 1.0)
    contam_log = log_component(logg, censored, log_res, mu_c, sigma_c)

    rho = float(np.clip(anomaly_rate, 0.01, 0.40))
    init = init or {}
    spec = gamma >= 1
    G = gamma + 1
    alpha = float(np.clip(init.get("alpha", 0.6), 0.05, 0.95))
    mu_w = float(init.get("mu_w", np.quantile(logg, 0.15)))
    sigma_w = max(float(init.get("sigma_w", 0.5)), SIGMA_FLOOR)
    mu_b = np.full(n_win, float(init.get("mu_b", np.quantile(logg, 0.85))))
    sigma_b = max(float(init.get("sigma_b", 0.3)), SIGMA_FLOOR)
    if spec and mu_w > mu_b.min() - MIN_SEPARATION:
        mu_w = mu_b.min() - MIN_SEPARATION

    prev_ll, post, loglik, it = -np.inf, None, -np.inf, 0
    log_full, log_comp = np.log1p(-rho), np.log(rho)
    for it in range(1, max_iter + 1):
        step_log = log_component(logg, censored, log_res, mu_b[win_idx], sigma_b)
        if spec:
            fast_log = log_component(logg, censored, log_res, mu_w, sigma_w)
            coreB = np.logaddexp(log_full + step_log, log_comp + fast_log)
            leB, leW = _mix(coreB, contam_log, rho), _mix(fast_log, contam_log, rho)
            offs = np.maximum(leB, leW)
        else:
            fast_log, coreB = step_log, step_log
            leB = leW = _mix(coreB, contam_log, rho)
            offs = leB
        eB, eW = np.exp(leB - offs), np.exp(leW - offs)

        post, xi_cont, xi_rest, ll_scaled = _forward_backward(eB, eW, alpha, G)
        loglik = ll_scaled + float(offs.sum())
        if loglik - prev_ll < tol * max(1.0, abs(prev_ll)):
            prev_ll = loglik
            break
        prev_ll = loglik

        if spec:
            denom = float(xi_cont.sum() + xi_rest.sum())
            if denom > 1e-8:
                alpha = float(np.clip(xi_cont.sum() / denom, 0.02, 0.98))

        w_b = post[:, 0]
        core_b = w_b * np.exp(np.log1p(-rho) + coreB - leB)
        if spec:
            share_full = np.exp(log_full + step_log - coreB)
            w_step = core_b * share_full
            w_fast = (1.0 - w_b) * np.exp(np.log1p(-rho) + fast_log - leW)
            w_fast = w_fast + core_b * (1.0 - share_full)
            mu_w, sigma_w = censored_moments(
                logg, censored, log_res, w_fast, mu_w, sigma_w, mu_w, sigma_w
            )
        else:
            w_step = core_b

        new_mu_b, sig_parts = mu_b.copy(), []
        for w in range(n_win):
            m = win_idx == w
            new_mu_b[w], _ = censored_moments(
                logg[m], censored[m], log_res, w_step[m], mu_b[w], sigma_b, mu_b[w], sigma_b
            )
            _, sg = censored_moments(
                logg[m], censored[m], log_res, w_step[m], new_mu_b[w], sigma_b,
                new_mu_b[w], sigma_b,
            )
            sig_parts.append((float(w_step[m].sum()), sg))
        wsum = sum(c for c, _ in sig_parts)
        if wsum > 1e-8:
            sigma_b = max(float(sum(c * sg for c, sg in sig_parts) / wsum), SIGMA_FLOOR)
        mu_b = new_mu_b
        if spec and mu_w > mu_b.min() - MIN_SEPARATION:
            mu_w = mu_b.min() - MIN_SEPARATION

    n_params = n_win + 4 if spec else n_win + 1
    return HsmmFit(
        gamma=gamma, alpha=alpha if spec else 0.0,
        mu_w=mu_w if spec else float("-inf"), sigma_w=sigma_w if spec else 0.0,
        mu_b=mu_b, sigma_b=sigma_b, loglik=loglik, n_params=n_params,
        bic=-2.0 * loglik + n_params * np.log(max(n, 2)),
        window=window, win_idx=win_idx,
        post_between=post[:, 0] if post is not None else np.ones(n), n_iter=it,
    )
