"""
Section 1: OLS -- Ordinary Least Squares

Implements OLS estimation from scratch and provides tools for
diagnosing omitted variable bias via Monte Carlo simulation.
"""

import numpy as np
from .utils import ols_fit, add_const


def estimate(X, y):
    """
    OLS estimation: beta_hat = (X'X)^{-1} X'y.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix (include a constant column for intercept).
    y : ndarray, shape (n,)
        Outcome vector.

    Returns
    -------
    dict with keys:
        beta    : coefficient vector
        se      : standard errors (homoskedastic)
        residuals : OLS residuals
        s2      : estimated error variance
        fitted  : fitted values X @ beta
    """
    b, se, e, s2 = ols_fit(X, y)
    return dict(beta=b, se=se, residuals=e, s2=s2, fitted=X @ b)


def ovb_formula(beta_omitted, cov_included_omitted, var_included):
    """
    Compute the omitted variable bias.

    bias = beta_omitted * Cov(included, omitted) / Var(included)

    Parameters
    ----------
    beta_omitted : float
        True coefficient on the omitted variable.
    cov_included_omitted : float
        Covariance between included and omitted regressors.
    var_included : float
        Variance of the included regressor.

    Returns
    -------
    float
        The OVB -- additive bias in the short-regression coefficient.
    """
    return beta_omitted * cov_included_omitted / var_included


def monte_carlo_ovb(n, n_sims, rho, beta_schooling=2.5, beta_ability=1.5,
                    sigma_eps=3.0, seed=None):
    """
    Monte Carlo demonstration of omitted variable bias.

    Generates data from the DGP:
        schooling = 12 + rho * ability + noise
        wage      = beta_schooling * schooling + beta_ability * ability + eps

    and estimates both the short regression (schooling only) and the long
    regression (schooling + ability) across `n_sims` replications.

    Parameters
    ----------
    n : int
        Sample size per simulation.
    n_sims : int
        Number of Monte Carlo replications.
    rho : float
        Strength of confounding (coefficient of ability in schooling equation).
    beta_schooling : float
        True coefficient on schooling.
    beta_ability : float
        True coefficient on ability.
    sigma_eps : float
        Std dev of the wage equation error.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        mc_short : array of short-regression slope estimates
        mc_long  : array of long-regression slope estimates
        bias     : mean(mc_short) - beta_schooling
    """
    if seed is not None:
        np.random.seed(seed)

    v_sd = np.sqrt(max(4 - rho ** 2, 0.01))
    mc_short = np.empty(n_sims)
    mc_long = np.empty(n_sims)

    for sim in range(n_sims):
        ability = np.random.normal(0, 1, n)
        schooling = 12 + rho * ability + np.random.normal(0, v_sd, n)
        wage = (beta_schooling * schooling
                + beta_ability * ability
                + np.random.normal(0, sigma_eps, n))
        mc_short[sim] = ols_fit(add_const(schooling), wage)[0][1]
        mc_long[sim] = ols_fit(
            add_const(np.column_stack([schooling, ability])), wage
        )[0][1]

    return dict(
        mc_short=mc_short,
        mc_long=mc_long,
        bias=np.mean(mc_short) - beta_schooling,
    )
