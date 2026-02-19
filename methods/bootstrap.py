"""
Section 10: Bootstrap Inference

Implements the nonparametric bootstrap for computing standard errors
and confidence intervals for any estimator.
"""

import numpy as np
from .utils import ols_fit


def bootstrap_statistic(data_X, data_y, estimator, n_boot=2000, seed=None):
    """
    Nonparametric bootstrap for an arbitrary estimator.

    Parameters
    ----------
    data_X : ndarray, shape (n, k)
        Design matrix.
    data_y : ndarray, shape (n,)
        Outcome vector.
    estimator : callable
        Function (X, y) -> scalar estimate.
    n_boot : int
        Number of bootstrap replications.
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        boot_estimates : array of bootstrap estimates
        se             : bootstrap standard error
        ci_lo, ci_hi   : 2.5th and 97.5th percentile CI
        mean           : mean of bootstrap distribution
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(data_y)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            boots[b] = estimator(data_X[idx], data_y[idx])
        except Exception:
            boots[b] = np.nan

    valid = boots[~np.isnan(boots)]
    ci = np.percentile(valid, [2.5, 97.5])

    return dict(
        boot_estimates=valid,
        se=np.std(valid),
        ci_lo=ci[0],
        ci_hi=ci[1],
        mean=np.mean(valid),
    )


def bootstrap_ols_slope(X, y, n_boot=2000, coef_idx=1, seed=None):
    """
    Bootstrap the OLS slope coefficient.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix (with constant).
    y : ndarray, shape (n,)
        Outcome.
    n_boot : int
        Number of replications.
    coef_idx : int
        Index of coefficient to bootstrap (default 1 = first slope).
    seed : int or None
        Random seed.

    Returns
    -------
    dict with keys:
        boot_estimates : array of bootstrapped coefficients
        se             : bootstrap SE
        ci_lo, ci_hi   : percentile 95% CI
        analytic_se    : analytic OLS SE for comparison
        analytic_ci    : analytic 95% CI [lo, hi]
        beta_hat       : point estimate from full sample
    """
    b_full, se_full, _, _ = ols_fit(X, y)

    def _ols_slope(Xb, yb):
        return ols_fit(Xb, yb)[0][coef_idx]

    bs = bootstrap_statistic(X, y, _ols_slope, n_boot=n_boot, seed=seed)

    analytic_ci = [
        b_full[coef_idx] - 1.96 * se_full[coef_idx],
        b_full[coef_idx] + 1.96 * se_full[coef_idx],
    ]

    return dict(
        boot_estimates=bs["boot_estimates"],
        se=bs["se"],
        ci_lo=bs["ci_lo"],
        ci_hi=bs["ci_hi"],
        analytic_se=se_full[coef_idx],
        analytic_ci=analytic_ci,
        beta_hat=b_full[coef_idx],
    )


def unique_obs_fraction(n):
    """
    Theoretical fraction of unique observations in a bootstrap sample.

    P(observation included) = 1 - (1 - 1/n)^n  ->  1 - 1/e ≈ 0.632

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    float
        Expected fraction of unique observations.
    """
    return 1 - (1 - 1 / n) ** n
