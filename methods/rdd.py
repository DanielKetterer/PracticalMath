"""
Section 7: Regression Discontinuity Design (RDD)

Implements sharp RDD via local linear regression, McCrary density test,
and bias-corrected inference from scratch.
"""

import numpy as np
from scipy import stats
from .utils import ols_fit, add_const


def estimate_rdd(y, running_var, cutoff, bandwidth):
    """
    Sharp RDD via local linear regression within a bandwidth window.

    Fits separate linear regressions on each side of the cutoff:
        Below: y = alpha_L + beta_L * (x - c) + eps
        Above: y = alpha_R + beta_R * (x - c) + eps
    tau_hat = alpha_R - alpha_L

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome.
    running_var : ndarray, shape (n,)
        Running (forcing) variable.
    cutoff : float
        Treatment assignment threshold.
    bandwidth : float
        Half-width of the estimation window around the cutoff.

    Returns
    -------
    dict with keys:
        tau        : RDD treatment effect estimate
        beta       : full coefficient vector [const, treated, xc, xc*treated]
        se         : standard errors
        residuals  : OLS residuals
        n_below    : observations below cutoff in window
        n_above    : observations above cutoff in window
    """
    mask = np.abs(running_var - cutoff) <= bandwidth
    xc = running_var[mask] - cutoff
    treated = (running_var[mask] >= cutoff).astype(float)
    yb = y[mask]

    X = np.column_stack([np.ones(mask.sum()), treated, xc, xc * treated])
    b, se, e, _ = ols_fit(X, yb)

    return dict(
        tau=b[1],
        beta=b,
        se=se,
        residuals=e,
        n_below=int((treated == 0).sum()),
        n_above=int((treated == 1).sum()),
    )


def mccrary_density_test(running_var, cutoff, bandwidth=10, alpha=0.001):
    """
    McCrary (2008) density test for manipulation at the cutoff.

    Tests whether there is bunching (discontinuity in density) at the
    cutoff by comparing counts in equal-width bins on each side.

    Parameters
    ----------
    running_var : ndarray, shape (n,)
        Running variable.
    cutoff : float
        Cutoff value.
    bandwidth : float
        Window width for counting on each side.
    alpha : float
        Significance level.

    Returns
    -------
    dict with keys:
        z_stat    : z-test statistic
        p_value   : two-sided p-value
        reject    : bool, True if p < alpha
        n_below   : count below cutoff in window
        n_above   : count above cutoff in window
    """
    below = np.sum((running_var >= cutoff - bandwidth) & (running_var < cutoff))
    above = np.sum((running_var >= cutoff) & (running_var < cutoff + bandwidth))
    z_stat = (above - below) / np.sqrt(above + below)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return dict(
        z_stat=z_stat,
        p_value=p_value,
        reject=p_value < alpha,
        n_below=int(below),
        n_above=int(above),
    )


def bias_corrected_rdd(y, running_var, cutoff, bandwidth, n_boot=500):
    """
    Bias-corrected RDD estimate using quadratic fits on each side.

    Approximates the Calonico, Cattaneo & Titiunik (2014) approach
    by fitting quadratic polynomials and bootstrapping for SEs.

    Parameters
    ----------
    y : ndarray, shape (n,)
    running_var : ndarray, shape (n,)
    cutoff : float
    bandwidth : float
    n_boot : int
        Number of bootstrap replications for SE.

    Returns
    -------
    dict with keys:
        tau_bc  : bias-corrected estimate
        se_bc   : bootstrap SE
        ci_lo   : lower 95% CI bound
        ci_hi   : upper 95% CI bound
    """
    mask = np.abs(running_var - cutoff) <= bandwidth
    xc = running_var[mask] - cutoff
    yb = y[mask]
    below = xc < 0

    def _quad_fit(xc_sub, y_sub):
        X = np.column_stack([np.ones(len(xc_sub)), xc_sub, xc_sub ** 2])
        return np.linalg.lstsq(X, y_sub, rcond=None)[0]

    b_bel = _quad_fit(xc[below], yb[below])
    b_abo = _quad_fit(xc[~below], yb[~below])
    tau_bc = b_abo[0] - b_bel[0]

    # Bootstrap SE
    boots = np.zeros(n_boot)
    for b in range(n_boot):
        idx = np.random.choice(len(yb), len(yb), replace=True)
        xc_b, yb_b = xc[idx], yb[idx]
        bel_b = xc_b < 0
        if bel_b.sum() < 3 or (~bel_b).sum() < 3:
            boots[b] = np.nan
            continue
        b_bel_b = _quad_fit(xc_b[bel_b], yb_b[bel_b])
        b_abo_b = _quad_fit(xc_b[~bel_b], yb_b[~bel_b])
        boots[b] = b_abo_b[0] - b_bel_b[0]

    boots = boots[~np.isnan(boots)]
    se_bc = np.std(boots)

    return dict(
        tau_bc=tau_bc,
        se_bc=se_bc,
        ci_lo=tau_bc - 1.96 * se_bc,
        ci_hi=tau_bc + 1.96 * se_bc,
    )


def global_rdd_fit(y, running_var, cutoff):
    """
    Global linear RDD fit (separate lines on each side of cutoff).

    Useful for the full-sample visualization.

    Parameters
    ----------
    y : ndarray
    running_var : ndarray
    cutoff : float

    Returns
    -------
    dict with keys:
        beta_below, beta_above : OLS coefficients on each side
        jump : estimated discontinuity at the cutoff
    """
    below = running_var < cutoff
    b_bel = ols_fit(add_const(running_var[below]), y[below])[0]
    b_abo = ols_fit(add_const(running_var[~below]), y[~below])[0]
    jump = (b_abo[0] + b_abo[1] * cutoff) - (b_bel[0] + b_bel[1] * cutoff)

    return dict(beta_below=b_bel, beta_above=b_abo, jump=jump)
