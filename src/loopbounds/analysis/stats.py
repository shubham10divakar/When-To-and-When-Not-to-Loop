"""Statistics helpers (design doc §7.4): Wilson intervals, bootstrap, Brier score,
log-loss, OLS with HC3 standard errors, AIC, logistic regression."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import optimize, stats


def wilson(k, n, z: float = 1.959963984540054):
    """Wilson score interval. Returns (p_hat, lo, hi); works elementwise."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        den = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return p, np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def bootstrap(groups: list, stat: Callable, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap resampling *groups* (e.g. runs) with replacement.

    ``stat`` maps a list of groups to a scalar (or 1-D array). Returns (estimate, lo, hi).
    """
    rng = np.random.default_rng(seed)
    est = np.asarray(stat(groups), float)
    G = len(groups)
    if G == 0:
        return est, np.full_like(est, np.nan), np.full_like(est, np.nan)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, G, size=G)
        draws.append(stat([groups[i] for i in idx]))
    draws = np.asarray(draws, float)
    lo = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)
    return est, lo, hi


def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps: float = 1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ols_hc3(X, y) -> dict:
    """OLS with HC3 heteroskedasticity-robust standard errors."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = resid**2 / np.clip(1 - h, 1e-12, None) ** 2
    cov = XtX_inv @ (X.T * w) @ X @ XtX_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
    dof = max(n - k, 1)
    sigma2 = float((resid**2).sum()) / dof
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return {"coef": beta, "se": se, "cov": cov, "r2": r2, "resid": resid, "n": n, "loglik": loglik,
            "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se}


def aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik


def logistic_fit(X, y) -> dict:
    """Maximum-likelihood logistic regression (with intercept column included in X)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)) + 1e-6 * float(b @ b)

    def grad(b):
        z = X @ b
        return X.T @ (stats.logistic.cdf(z) - y) + 2e-6 * b

    res = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS")
    b = res.x
    pz = stats.logistic.cdf(X @ b)
    H = (X.T * (pz * (1 - pz))) @ X
    se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    return {"coef": b, "se": se, "loglik": -res.fun, "n": len(y), "converged": bool(res.success)}
