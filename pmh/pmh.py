"""Marginal Particle Metropolis–Hastings for the univariate SV model."""

import numpy as np
from numba import njit


@njit
def particle_loglik(y, theta, n, rng, guided=True):
    """Notebook filter, using Eq. (26) when guided and Algorithm A1 weights."""
    beta, phi, sigma = theta
    h = rng.standard_normal(n) * sigma / np.sqrt(1 - phi**2)
    log_n = np.log(n)
    log_w = np.full(n, -log_n)
    log_likelihood = 0.0
    for t, observation in enumerate(y):
        scaled_square = (observation / beta)**2
        if t > 0:
            mean = phi * h
            z = rng.standard_normal(n)
            shift = np.zeros(n)
            if guided:
                shift = 0.5 * sigma**2 * (scaled_square * np.exp(-mean) - 1)
            h = mean + shift + sigma * z
            # log f - log q with equal transition and proposal variances.
            log_w -= 0.5 * (shift / sigma) * (2 * z + shift / sigma)
        log_w -= 0.5 * (np.log(2 * np.pi) + h + scaled_square * np.exp(-h))
        log_w -= np.log(beta)
        largest = log_w.max()
        if not np.isfinite(largest):
            return -np.inf
        increment = largest + np.log(np.exp(log_w - largest).sum())
        log_likelihood += increment
        log_w -= increment
        weights = np.exp(log_w)
        if t + 1 < len(y) and np.sum(weights**2) > 2.0 / n:
            cdf = np.cumsum(weights)
            cdf[-1] = 1.0
            positions = (rng.random() + np.arange(n)) / n
            h = h[np.searchsorted(cdf, positions, side="right")]
            log_w.fill(-log_n)
    return log_likelihood


def log_prior(theta, ig_scale=0.25):
    """Eq. (75), with density expressed in (beta, phi, sigma)."""
    beta, phi, sigma = theta
    if not (beta > 0 and -1 < phi < 1 and sigma > 0):
        return -np.inf
    p = (phi + 1) / 2
    return (-np.log(beta) + 19 * np.log(p) + 0.5 * np.log1p(-p)
            - 11 * np.log(sigma) - ig_scale / sigma**2)


def sample_pmh(y, initial, proposal, n, iterations, seed, ig_scale=0.25):
    """Gaussian random walk with fixed Cholesky factor proposal."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(initial, dtype=float)
    score = log_prior(theta, ig_scale) + particle_loglik(y, theta, n, rng)
    if not np.isfinite(score):
        raise ValueError("The initial parameter must have a finite log target.")
    draws = np.empty((iterations, 3))
    accepted = np.zeros(iterations, dtype=bool)
    failures = 0
    for i in range(iterations):
        candidate = theta + proposal @ rng.standard_normal(3)
        candidate_score = log_prior(candidate, ig_scale)
        if np.isfinite(candidate_score):
            candidate_loglik = particle_loglik(y, candidate, n, rng)
            failures += int(not np.isfinite(candidate_loglik))
            candidate_score += candidate_loglik
            if np.log(rng.random()) < candidate_score - score:
                theta, score = candidate, candidate_score
                accepted[i] = True
        # On rejection both theta and the current estimated score are retained.
        draws[i] = theta
    return draws, accepted, failures
