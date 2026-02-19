"""
Section 9: Maximum Likelihood Estimation -- from scratch

Provides a generic MLE framework and tools for computing standard
errors via the observed Fisher information (Hessian), profile
likelihoods, and BFGS optimization paths.
"""

import numpy as np
from scipy.optimize import minimize, approx_fprime


def fit_mle(neg_log_lik, start, args=(), method="BFGS", track_path=False):
    """
    Generic MLE via scipy.optimize.minimize.

    Parameters
    ----------
    neg_log_lik : callable
        Negative log-likelihood function: f(beta, *args) -> float.
    start : ndarray
        Starting parameter values.
    args : tuple
        Extra arguments passed to neg_log_lik.
    method : str
        Optimization method (default BFGS).
    track_path : bool
        If True, record the optimization path.

    Returns
    -------
    dict with keys:
        beta      : MLE estimates
        se        : standard errors from observed Fisher info
        nll       : negative log-likelihood at optimum
        hessian   : numerical Hessian at the MLE
        converged : bool
        path      : list of parameter arrays (if track_path)
    """
    path = [np.array(start).copy()]
    callback = (lambda xk: path.append(xk.copy())) if track_path else None

    res = minimize(neg_log_lik, start, args=args, method=method,
                   callback=callback)

    beta = res.x
    # Numerical Hessian for SEs
    hess = numerical_hessian(neg_log_lik, beta, args=args)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(hess)))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)

    result = dict(
        beta=beta,
        se=se,
        nll=res.fun,
        hessian=hess,
        converged=res.success,
    )
    if track_path:
        result["path"] = np.array(path)

    return result


def numerical_hessian(neg_log_lik, beta, args=(), eps=1e-5):
    """
    Numerical Hessian of the negative log-likelihood at beta.

    Parameters
    ----------
    neg_log_lik : callable
    beta : ndarray
    args : tuple
    eps : float

    Returns
    -------
    H : ndarray, shape (k, k)
        Hessian matrix.
    """
    k = len(beta)
    H = np.array([
        approx_fprime(
            beta,
            lambda b, j=j: approx_fprime(b, neg_log_lik, eps, *args)[j],
            eps,
        )
        for j in range(k)
    ])
    return H


def profile_likelihood(neg_log_lik, beta_mle, profile_idx, grid, args=(),
                       method="BFGS"):
    """
    Compute the profile likelihood for a single parameter.

    For each value of beta[profile_idx] on the grid, maximizes
    the log-likelihood over all other parameters.

    Parameters
    ----------
    neg_log_lik : callable
    beta_mle : ndarray
        MLE estimates (used as starting values).
    profile_idx : int
        Index of the parameter to profile.
    grid : ndarray
        Grid of values for the profiled parameter.
    args : tuple

    Returns
    -------
    dict with keys:
        grid        : parameter values
        profile_ll  : profile log-likelihood at each grid point
        ci_95       : (lo, hi) 95% CI based on likelihood ratio
    """
    k = len(beta_mle)
    other_idx = [j for j in range(k) if j != profile_idx]

    profile_ll = np.empty(len(grid))
    for i, val in enumerate(grid):
        def _partial_nll(b_other, _val=val):
            b_full = np.empty(k)
            b_full[profile_idx] = _val
            b_full[other_idx] = b_other
            return neg_log_lik(b_full, *args)

        start_other = beta_mle[other_idx]
        res = minimize(_partial_nll, start_other, method=method,
                       options={"disp": False})
        profile_ll[i] = -res.fun

    # 95% CI: log-likelihood within 1.92 of maximum (chi2_1(0.95)/2)
    ll_max = profile_ll.max()
    in_ci = profile_ll >= (ll_max - 1.92)
    if in_ci.any():
        ci = (grid[in_ci].min(), grid[in_ci].max())
    else:
        ci = (np.nan, np.nan)

    return dict(grid=grid, profile_ll=profile_ll, ci_95=ci)


def log_likelihood_surface(neg_log_lik, beta_grid_0, beta_grid_1, args=()):
    """
    Compute the log-likelihood on a 2-d grid (for contour plots).

    Parameters
    ----------
    neg_log_lik : callable
    beta_grid_0 : ndarray
        Grid for first parameter.
    beta_grid_1 : ndarray
        Grid for second parameter.
    args : tuple

    Returns
    -------
    B0, B1 : meshgrid arrays
    LL : ndarray
        Log-likelihood values on the grid.
    """
    B0, B1 = np.meshgrid(beta_grid_0, beta_grid_1)
    LL = np.array([
        [-neg_log_lik(np.array([B0[i, j], B1[i, j]]), *args)
         for j in range(len(beta_grid_0))]
        for i in range(len(beta_grid_1))
    ])
    return B0, B1, LL
