"""
Section 6: Difference-in-Differences (DiD)

Implements the 2x2 DiD estimator and event study specifications
from scratch.
"""

import numpy as np
from .utils import ols_fit


def estimate_did(y, treated, post):
    """
    2x2 Difference-in-Differences estimator.

    y = beta_0 + beta_1 * Treated + beta_2 * Post
      + beta_3 * (Treated * Post) + epsilon

    beta_3 is the DiD estimate of the treatment effect.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome.
    treated : ndarray, shape (n,)
        Treatment group indicator (0/1).
    post : ndarray, shape (n,)
        Post-period indicator (0/1).

    Returns
    -------
    dict with keys:
        beta         : full coefficient vector [const, treated, post, interaction]
        tau_did      : DiD treatment effect estimate (beta_3)
        se_classical : homoskedastic SEs
        se_robust    : HC1 robust SEs
        residuals    : OLS residuals
    """
    X = np.column_stack([np.ones(len(y)), treated, post, treated * post])
    b, se, e, _ = ols_fit(X, y)

    # HC1 robust SEs
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = (X.T * e ** 2) @ X
    V_hc1 = bread @ meat @ bread * (n / (n - k))
    se_robust = np.sqrt(np.diag(V_hc1))

    return dict(
        beta=b,
        tau_did=b[3],
        se_classical=se,
        se_robust=se_robust,
        residuals=e,
    )


def event_study(y, treat_unit, time_ids, treatment_time):
    """
    Event study specification: estimate dynamic treatment effects
    at each relative time period.

    For each relative time k (relative to treatment_time), estimates
    a DiD comparing that period to the reference period (k = -1).

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome (stacked panel).
    treat_unit : ndarray, shape (n,)
        Treatment group indicator (0/1).
    time_ids : ndarray, shape (n,)
        Time period identifiers (integer).
    treatment_time : int
        Time period when treatment begins.

    Returns
    -------
    dict with keys:
        rel_times : array of relative time periods
        coefs     : estimated effect at each relative time
        ses       : standard errors
    """
    rel_time = time_ids - treatment_time
    T_min = int(rel_time.min())
    T_max = int(rel_time.max())

    event_coefs = []
    event_ses = []
    event_times = []

    for rt in range(T_min, T_max + 1):
        if rt == -1:
            # Reference period: coefficient normalized to zero
            event_coefs.append(0.0)
            event_ses.append(0.0)
            event_times.append(rt)
            continue

        # DiD for this relative time vs reference period (-1)
        mask = (rel_time == rt) | (rel_time == -1)
        y_rt = y[mask]
        tr_rt = treat_unit[mask]
        post_rt = (rel_time[mask] == rt).astype(float)
        X_rt = np.column_stack([
            np.ones(mask.sum()), tr_rt, post_rt, tr_rt * post_rt
        ])
        b_rt, se_rt, _, _ = ols_fit(X_rt, y_rt)
        event_coefs.append(b_rt[3])
        event_ses.append(se_rt[3])
        event_times.append(rt)

    return dict(
        rel_times=np.array(event_times),
        coefs=np.array(event_coefs),
        ses=np.array(event_ses),
    )


def group_means(y, treated, post):
    """
    Compute the 2x2 group means table for DiD.

    Returns
    -------
    dict with keys:
        treat_pre, treat_post, ctrl_pre, ctrl_post : group means
        delta_treat : change for treated group
        delta_ctrl  : change for control group
        tau_did     : DiD estimate (delta_treat - delta_ctrl)
    """
    t_pre = y[(treated == 1) & (post == 0)].mean()
    t_post = y[(treated == 1) & (post == 1)].mean()
    c_pre = y[(treated == 0) & (post == 0)].mean()
    c_post = y[(treated == 0) & (post == 1)].mean()

    return dict(
        treat_pre=t_pre,
        treat_post=t_post,
        ctrl_pre=c_pre,
        ctrl_post=c_post,
        delta_treat=t_post - t_pre,
        delta_ctrl=c_post - c_pre,
        tau_did=(t_post - t_pre) - (c_post - c_pre),
    )
