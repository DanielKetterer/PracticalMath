"""
Section 4: Instrumental Variables (IV / 2SLS)

Implements two-stage least squares from scratch, including:
- First-stage F-statistic for instrument strength
- Correct 2SLS standard errors (using actual X, not X_hat)
- HC1 robust SEs for 2SLS
- Wald estimator for the just-identified case
"""

import numpy as np
from .utils import ols_fit, add_const


def first_stage(Z, X_endog):
    """
    First stage of 2SLS: regress endogenous X on instruments Z.

    Parameters
    ----------
    Z : ndarray, shape (n, m)
        Instrument matrix (including constant and exogenous controls).
    X_endog : ndarray, shape (n,)
        Endogenous regressor.

    Returns
    -------
    dict with keys:
        X_hat    : fitted values from first stage
        gamma    : first-stage coefficients
        se       : first-stage SEs
        F_stat   : first-stage F-statistic (for instrument relevance)
        residuals: first-stage residuals
    """
    b, se, e, _ = ols_fit(Z, X_endog)
    X_hat = Z @ b
    # F-stat on the excluded instrument(s)
    # For a single instrument: t^2 on the instrument coefficient
    F_stat = (b[1] / se[1]) ** 2
    return dict(X_hat=X_hat, gamma=b, se=se, F_stat=F_stat, residuals=e)


def second_stage(X_hat, X_actual, y, return_naive=False):
    """
    Second stage of 2SLS with correct standard errors.

    Parameters
    ----------
    X_hat : ndarray, shape (n,)
        Fitted values from the first stage.
    X_actual : ndarray, shape (n,)
        Actual (endogenous) regressor values.
    y : ndarray, shape (n,)
        Outcome variable.
    return_naive : bool
        If True, also return the naive (incorrect) SEs.

    Returns
    -------
    dict with keys:
        beta         : 2SLS coefficient vector [intercept, slope]
        se_hom       : correct homoskedastic 2SLS SEs
        se_robust    : HC1 robust 2SLS SEs
        residuals    : correct residuals (y - X_actual @ beta)
        se_naive     : (only if return_naive) incorrect naive SEs
    """
    X_2s = add_const(X_hat)
    X_act = add_const(X_actual)
    n, k = X_2s.shape

    # 2SLS coefficient (from regressing y on X_hat)
    b_2sls = ols_fit(X_2s, y)[0]

    # CORRECT residuals use actual X, not X_hat
    e_correct = y - X_act @ b_2sls
    # Homoskedastic SE
    sigma2 = (e_correct @ e_correct) / (n - k)
    bread = np.linalg.inv(X_2s.T @ X_2s)
    se_hom = np.sqrt(np.diag(sigma2 * bread))

    # Robust HC1 SE
    meat = (X_2s.T * e_correct ** 2) @ X_2s
    V_rob = bread @ meat @ bread * (n / (n - k))
    se_rob = np.sqrt(np.diag(V_rob))

    result = dict(
        beta=b_2sls,
        se_hom=se_hom,
        se_robust=se_rob,
        residuals=e_correct,
    )

    if return_naive:
        e_naive = y - X_2s @ b_2sls
        sigma2_naive = (e_naive @ e_naive) / (n - k)
        se_naive = np.sqrt(np.diag(sigma2_naive * bread))
        result["se_naive"] = se_naive

    return result


def estimate_2sls(Z, X_endog, y):
    """
    Full 2SLS estimation pipeline.

    Parameters
    ----------
    Z : ndarray, shape (n, m)
        Instrument matrix (with constant).
    X_endog : ndarray, shape (n,)
        Endogenous regressor.
    y : ndarray, shape (n,)
        Outcome.

    Returns
    -------
    dict with keys:
        beta        : 2SLS coefficients [intercept, slope]
        se_hom      : correct homoskedastic SEs
        se_robust   : HC1 robust SEs
        se_naive    : naive (incorrect) SEs
        F_stat      : first-stage F-statistic
        X_hat       : first-stage fitted values
        ols_beta    : OLS (biased) coefficients for comparison
    """
    fs = first_stage(Z, X_endog)
    ss = second_stage(fs["X_hat"], X_endog, y, return_naive=True)
    b_ols = ols_fit(add_const(X_endog), y)[0]

    return dict(
        beta=ss["beta"],
        se_hom=ss["se_hom"],
        se_robust=ss["se_robust"],
        se_naive=ss["se_naive"],
        F_stat=fs["F_stat"],
        X_hat=fs["X_hat"],
        ols_beta=b_ols,
        first_stage=fs,
    )


def wald_estimator(y, X_endog, Z_excluded):
    """
    Wald (ratio) IV estimator for the just-identified case.

    beta_IV = Cov(y, Z) / Cov(X, Z)  =  Reduced Form / First Stage

    Parameters
    ----------
    y : ndarray, shape (n,)
    X_endog : ndarray, shape (n,)
    Z_excluded : ndarray, shape (n,)
        The excluded instrument (single variable, no constant).

    Returns
    -------
    dict with keys:
        beta_iv       : IV estimate
        reduced_form  : slope of y on Z
        first_stage   : slope of X on Z
    """
    Z = add_const(Z_excluded)
    b_rf = ols_fit(Z, y)[0]
    b_fs = ols_fit(Z, X_endog)[0]
    return dict(
        beta_iv=b_rf[1] / b_fs[1],
        reduced_form=b_rf[1],
        first_stage=b_fs[1],
    )
