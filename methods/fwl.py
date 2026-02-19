"""
Section 2: Frisch-Waugh-Lovell (FWL) Theorem

Implements the partialing-out procedure:
  beta_hat_1 from y ~ X1 + X2 equals the slope from
  regressing M2*y on M2*X1, where M2 is the annihilator of X2.
"""

import numpy as np
from .utils import ols_fit, add_const


def partial_out(y, X1, X2):
    """
    Partial out X2 from both y and X1 using the annihilator matrix
    M2 = I - X2 (X2'X2)^{-1} X2'.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome vector.
    X1 : ndarray, shape (n,) or (n, k1)
        Variable(s) of interest.
    X2 : ndarray, shape (n, k2)
        Control variable(s) to partial out (should include a constant
        if an intercept is desired in the auxiliary regressions).

    Returns
    -------
    dict with keys:
        resid_y  : M2 * y  -- residuals of y on X2
        resid_X1 : M2 * X1 -- residuals of X1 on X2
        fwl_coef : scalar slope from regressing resid_y on resid_X1
    """
    X1 = np.atleast_1d(X1)
    if X1.ndim == 1:
        X1_col = X1
    else:
        X1_col = X1[:, 0] if X1.shape[1] == 1 else X1

    _, _, resid_y, _ = ols_fit(X2, y)
    if X1.ndim == 1 or X1.shape[1] == 1:
        _, _, resid_X1, _ = ols_fit(X2, X1_col)
        fwl_coef = (resid_X1 @ resid_y) / (resid_X1 @ resid_X1)
    else:
        resid_X1 = np.column_stack([
            ols_fit(X2, X1[:, j])[2] for j in range(X1.shape[1])
        ])
        fwl_coef = ols_fit(resid_X1, resid_y)[0]

    return dict(resid_y=resid_y, resid_X1=resid_X1, fwl_coef=fwl_coef)


def verify_fwl(y, X_full, idx_interest=1):
    """
    Verify the FWL theorem: compare the coefficient from the full
    regression to the partialing-out coefficient.

    Parameters
    ----------
    y : ndarray, shape (n,)
    X_full : ndarray, shape (n, k)
        Full design matrix (including constant).
    idx_interest : int
        Column index of the variable of interest in X_full.

    Returns
    -------
    dict with keys:
        full_coef    : coefficient from full regression
        fwl_coef     : coefficient from partialing-out
        match        : bool, True if they agree to ~1e-10
    """
    b_full, _, _, _ = ols_fit(X_full, y)

    # Partition: X1 = column of interest, X2 = everything else
    k = X_full.shape[1]
    other_cols = [j for j in range(k) if j != idx_interest]
    X1 = X_full[:, idx_interest]
    X2 = X_full[:, other_cols]

    result = partial_out(y, X1, X2)

    return dict(
        full_coef=b_full[idx_interest],
        fwl_coef=result["fwl_coef"],
        match=np.abs(b_full[idx_interest] - result["fwl_coef"]) < 1e-10,
    )
