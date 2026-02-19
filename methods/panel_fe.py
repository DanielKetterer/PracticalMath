"""
Section 5: Panel Data -- Fixed Effects (Within Estimator)

Implements the within (demeaning) estimator for panel data and
Arellano (1987) clustered standard errors from scratch.
"""

import numpy as np
import pandas as pd
from .utils import ols_fit


def within_demean(y, X, unit_ids):
    """
    Demean y and X within each unit (entity) for fixed-effects estimation.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Outcome vector (stacked panel).
    X : ndarray, shape (n,) or (n, k)
        Regressor(s) (stacked panel).
    unit_ids : ndarray, shape (n,)
        Unit identifiers for each observation.

    Returns
    -------
    dict with keys:
        y_demean : demeaned y
        X_demean : demeaned X
    """
    df = pd.DataFrame({"y": y, "u": unit_ids})
    y_mean = df.groupby("u")["y"].transform("mean").values
    y_dm = y - y_mean

    X = np.atleast_1d(X)
    if X.ndim == 1:
        df["x"] = X
        x_mean = df.groupby("u")["x"].transform("mean").values
        X_dm = X - x_mean
    else:
        X_dm = np.empty_like(X)
        for j in range(X.shape[1]):
            df[f"x{j}"] = X[:, j]
            x_mean = df.groupby("u")[f"x{j}"].transform("mean").values
            X_dm[:, j] = X[:, j] - x_mean

    return dict(y_demean=y_dm, X_demean=X_dm)


def estimate_fe(y, X, unit_ids):
    """
    Fixed-effects (within) estimation with clustered standard errors.

    For a single regressor:
        beta_FE = (X_dm' X_dm)^{-1} X_dm' y_dm

    Parameters
    ----------
    y : ndarray, shape (n,)
    X : ndarray, shape (n,)
        Single treatment/regressor variable.
    unit_ids : ndarray, shape (n,)
        Unit identifiers.

    Returns
    -------
    dict with keys:
        beta_fe       : fixed-effects coefficient
        se_homosk     : homoskedastic SE
        se_cluster    : clustered SE (Arellano 1987)
        residuals     : within-estimator residuals
    """
    dm = within_demean(y, X, unit_ids)
    td = dm["X_demean"]
    yd = dm["y_demean"]

    # Within estimator
    b_fe = (td @ yd) / (td @ td)

    # Residuals
    e_fe = yd - b_fe * td

    # Homoskedastic SE
    N = len(td)
    K = 1
    se_homosk = np.sqrt((e_fe @ e_fe) / (N - K) / (td @ td))

    # Clustered SE (Arellano 1987)
    se_cluster = clustered_se_scalar(td, e_fe, unit_ids)

    return dict(
        beta_fe=b_fe,
        se_homosk=se_homosk,
        se_cluster=se_cluster,
        residuals=e_fe,
    )


def clustered_se_scalar(X_dm, residuals, unit_ids):
    """
    Arellano (1987) clustered standard errors for a scalar within estimator.

    V_cluster = (X'X)^{-1} * B * (X'X)^{-1}
    where B = sum_g (X_g' e_g)(X_g' e_g)' with finite-sample correction.

    Parameters
    ----------
    X_dm : ndarray, shape (n,)
        Demeaned regressor (scalar).
    residuals : ndarray, shape (n,)
        Within-estimator residuals.
    unit_ids : ndarray, shape (n,)
        Unit identifiers.

    Returns
    -------
    float
        Clustered standard error.
    """
    unique_units = np.unique(unit_ids)
    G = len(unique_units)
    N = len(X_dm)
    K = 1
    XtX = X_dm @ X_dm

    B = 0.0
    for g in unique_units:
        mask = unit_ids == g
        score = (X_dm[mask] * residuals[mask]).sum()
        B += score ** 2

    # Finite-sample correction: G/(G-1) * (N-1)/(N-K)
    dof_corr = (G / (G - 1)) * ((N - 1) / (N - K))

    return np.sqrt(B / XtX ** 2 * dof_corr)
