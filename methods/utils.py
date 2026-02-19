"""
Shared utility functions used across all econometric method modules.
"""

import numpy as np


def ols_fit(X, y):
    """
    OLS estimation via the normal equations.

    Parameters
    ----------
    X : ndarray, shape (n, k)
        Design matrix (should include a constant column if an intercept is desired).
    y : ndarray, shape (n,)
        Outcome vector.

    Returns
    -------
    b : ndarray, shape (k,)
        Coefficient estimates  beta_hat = (X'X)^{-1} X'y.
    se : ndarray, shape (k,)
        Homoskedastic standard errors.
    e : ndarray, shape (n,)
        Residuals  y - X @ b.
    s2 : float
        Estimated error variance  e'e / (n - k).
    """
    n, k = X.shape
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = (e @ e) / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se, e, s2


def add_const(x):
    """
    Prepend a column of ones (intercept) to the design matrix.

    Parameters
    ----------
    x : ndarray
        1-d array or 2-d matrix of regressors.

    Returns
    -------
    X : ndarray, shape (n, k+1)
        Design matrix with leading ones column.
    """
    x = np.atleast_2d(x).T if x.ndim == 1 else x
    return np.column_stack([np.ones(x.shape[0]), x])
