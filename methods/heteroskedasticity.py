"""
Section 3: Heteroskedasticity -- Detection and Robust Standard Errors

Implements the Breusch-Pagan test for heteroskedasticity and
HC1 (Huber-White) robust standard errors from scratch.
"""

import numpy as np
from scipy import stats
from .utils import ols_fit


def breusch_pagan_test(X, residuals):
    """
    Breusch-Pagan test for heteroskedasticity.

    Regresses squared OLS residuals on X. Under H0 (homoskedasticity),
    the F-statistic for the slope(s) should be insignificant.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix used in the original regression.
    residuals : ndarray, shape (n,)
        OLS residuals.

    Returns
    -------
    dict with keys:
        F_stat  : F-statistic
        p_value : p-value (from F distribution)
        reject  : bool, True if p < 0.05
    """
    n = X.shape[0]
    k = X.shape[1]
    esq = residuals ** 2
    bp_b, bp_se, _, _ = ols_fit(X, esq)
    # F-test for all slope coefficients jointly
    if k == 2:
        # Single regressor: scalar F
        F_stat = (bp_b[1] / bp_se[1]) ** 2
        p_value = 1 - stats.f.cdf(F_stat, 1, n - k)
    else:
        t_stats = bp_b[1:] / bp_se[1:]
        F_stat = np.mean(t_stats ** 2)
        p_value = 1 - stats.f.cdf(F_stat, k - 1, n - k)

    return dict(F_stat=F_stat, p_value=p_value, reject=p_value < 0.05)


def hc1_robust_se(X, residuals):
    """
    HC1 (Huber-White) heteroskedasticity-consistent standard errors.

    V_HC1 = (n/(n-k)) * (X'X)^{-1} * [sum_i e_hat_i^2 * x_i x_i'] * (X'X)^{-1}

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix.
    residuals : ndarray, shape (n,)
        OLS residuals.

    Returns
    -------
    se_robust : ndarray, shape (k,)
        HC1 robust standard errors.
    """
    n, k = X.shape
    esq = residuals ** 2
    meat = (X.T * esq) @ X
    bread = np.linalg.inv(X.T @ X)
    V_hc1 = bread @ meat @ bread * (n / (n - k))
    return np.sqrt(np.diag(V_hc1))


def estimate_with_robust_se(X, y):
    """
    OLS estimation with both homoskedastic and HC1 robust SEs.

    Parameters
    ----------
    X : ndarray, shape (n, k)
    y : ndarray, shape (n,)

    Returns
    -------
    dict with keys:
        beta          : coefficient vector
        se_classical  : homoskedastic SEs
        se_robust     : HC1 robust SEs
        residuals     : OLS residuals
        bp_test       : Breusch-Pagan test results
    """
    b, se_classical, e, _ = ols_fit(X, y)
    se_robust = hc1_robust_se(X, e)
    bp = breusch_pagan_test(X, e)

    return dict(
        beta=b,
        se_classical=se_classical,
        se_robust=se_robust,
        residuals=e,
        bp_test=bp,
    )
