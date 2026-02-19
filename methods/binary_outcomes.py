"""
Section 8: Binary Outcomes -- Probit and Logit

Implements logit and probit MLE from scratch, plus average marginal
effects (AME) with delta-method and bootstrap standard errors.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from .utils import ols_fit


def logistic(z):
    """Logistic (sigmoid) CDF: 1 / (1 + exp(-z))."""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def _nll_logit(b, X, y):
    """Negative log-likelihood for logit."""
    p = np.clip(logistic(X @ b), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def _nll_probit(b, X, y):
    """Negative log-likelihood for probit."""
    p = np.clip(stats.norm.cdf(X @ b), 1e-12, 1 - 1e-12)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logit(X, y, start=None):
    """
    Logit MLE via BFGS optimization.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix (with constant).
    y : ndarray, shape (n,)
        Binary outcome (0/1).
    start : ndarray or None
        Starting values for optimization. Defaults to zeros.

    Returns
    -------
    dict with keys:
        beta     : MLE coefficient vector
        p_hat    : predicted probabilities
        nll      : negative log-likelihood at optimum
        converged: bool
    """
    if start is None:
        start = np.zeros(X.shape[1])
    res = minimize(_nll_logit, start, args=(X, y), method="BFGS")
    p_hat = logistic(X @ res.x)
    return dict(beta=res.x, p_hat=p_hat, nll=res.fun, converged=res.success)


def fit_probit(X, y, start=None):
    """
    Probit MLE via BFGS optimization.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix (with constant).
    y : ndarray, shape (n,)
        Binary outcome (0/1).
    start : ndarray or None
        Starting values.

    Returns
    -------
    dict with keys:
        beta     : MLE coefficient vector
        p_hat    : predicted probabilities
        nll      : negative log-likelihood at optimum
        converged: bool
    """
    if start is None:
        start = np.zeros(X.shape[1])
    res = minimize(_nll_probit, start, args=(X, y), method="BFGS")
    p_hat = stats.norm.cdf(X @ res.x)
    return dict(beta=res.x, p_hat=p_hat, nll=res.fun, converged=res.success)


def logit_ame(X, beta, coef_idx=1):
    """
    Average Marginal Effect for a logit model.

    AME = mean( beta_j * p_i * (1 - p_i) )

    Parameters
    ----------
    X : ndarray, shape (n, k)
    beta : ndarray, shape (k,)
    coef_idx : int
        Index of the coefficient for which to compute AME.

    Returns
    -------
    float
        Average marginal effect.
    """
    p = logistic(X @ beta)
    return np.mean(beta[coef_idx] * p * (1 - p))


def probit_ame(X, beta, coef_idx=1):
    """
    Average Marginal Effect for a probit model.

    AME = mean( beta_j * phi(X @ beta) )

    Parameters
    ----------
    X : ndarray, shape (n, k)
    beta : ndarray, shape (k,)
    coef_idx : int
        Index of the coefficient for which to compute AME.

    Returns
    -------
    float
        Average marginal effect.
    """
    return np.mean(beta[coef_idx] * stats.norm.pdf(X @ beta))


def ame_se_delta(X, beta, y):
    """
    Delta-method standard error for the logit AME.

    Uses the observed Fisher information for the MLE covariance
    and computes the gradient of the AME with respect to beta.

    Parameters
    ----------
    X : ndarray, shape (n, k)
    beta : ndarray, shape (k,)
    y : ndarray, shape (n,)
        Binary outcome (used for dimensions).

    Returns
    -------
    float
        Delta-method SE for AME of beta[1].
    """
    n = len(y)
    k = len(beta)
    p = logistic(X @ beta)
    W = p * (1 - p)

    # Observed Fisher info: I(beta) = X' diag(p*(1-p)) X
    fisher = X.T @ (X * W[:, None])
    cov = np.linalg.inv(fisher)

    # Gradient of AME w.r.t. beta
    dpdb = np.zeros(k)
    for i in range(n):
        pi = p[i]
        w = pi * (1 - pi)
        dw_deta = pi * (1 - pi) * (1 - 2 * pi)
        dpdb[0] += beta[1] * dw_deta * X[i, 0] / n
        dpdb[1] += (w + beta[1] * dw_deta * X[i, 1]) / n
        for j in range(2, k):
            dpdb[j] += beta[1] * dw_deta * X[i, j] / n

    return np.sqrt(dpdb @ cov @ dpdb)


def ame_se_bootstrap(X, y, n_boot=500, coef_idx=1):
    """
    Bootstrap standard error for the logit AME.

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)
    n_boot : int
    coef_idx : int

    Returns
    -------
    float
        Bootstrap SE for AME.
    """
    n = len(y)
    boots = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            res = minimize(_nll_logit, np.zeros(X.shape[1]),
                           args=(X[idx], y[idx]), method="BFGS")
            bl = res.x
            pb = logistic(X[idx] @ bl)
            boots[b] = np.mean(bl[coef_idx] * pb * (1 - pb))
        except Exception:
            boots[b] = np.nan

    return np.std(boots[~np.isnan(boots)])


def fit_lpm(X, y):
    """
    Linear Probability Model (OLS on binary outcome).

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)

    Returns
    -------
    dict with keys: beta, se, residuals
    """
    b, se, e, _ = ols_fit(X, y)
    return dict(beta=b, se=se, residuals=e)
