"""
Mincer Earnings Equation on CPS Data
======================================

OLS and IV estimation of the return to schooling, using methods
from the PracticalMath methods/ package.

Status: stub -- replace simulated data with CPS microdata.
"""

import numpy as np
import sys
import os

# Add project root to path so methods package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.utils import ols_fit, add_const
from methods import ols as m_ols
from methods import fwl as m_fwl
from methods import heteroskedasticity as m_het
from methods import iv_2sls as m_iv
from methods import bootstrap as m_boot


def simulate_cps_data(n=2000, seed=42):
    """
    Simulate data mimicking CPS microdata for a Mincer equation.

    DGP:
        ability ~ N(0, 1)                       (unobserved)
        schooling = 12 + 0.8*ability + noise
        experience = age - schooling - 6
        log_wage = 0.10*schooling + 0.03*experience
                   - 0.0005*experience^2 + 0.5*ability + eps

    Returns
    -------
    dict with arrays: log_wage, schooling, experience, ability, distance
    """
    np.random.seed(seed)
    ability = np.random.normal(0, 1, n)
    schooling = 12 + 0.8 * ability + np.random.normal(0, 1.5, n)
    schooling = np.clip(schooling, 8, 20)
    age = np.random.uniform(25, 55, n)
    experience = age - schooling - 6
    experience = np.clip(experience, 0, 40)
    # Instrument: distance to nearest college (affects schooling, not wages)
    distance = np.random.uniform(0, 50, n)
    schooling = schooling - 0.04 * distance  # distance lowers schooling
    schooling = np.clip(schooling, 8, 20)

    log_wage = (0.10 * schooling + 0.03 * experience
                - 0.0005 * experience ** 2
                + 0.5 * ability + np.random.normal(0, 0.3, n))

    return dict(
        log_wage=log_wage, schooling=schooling,
        experience=experience, ability=ability,
        distance=distance,
        true_return=0.10,
    )


def main():
    print("=" * 60)
    print("Mincer Earnings Equation -- Return to Schooling")
    print("=" * 60)

    data = simulate_cps_data()
    y = data["log_wage"]
    schooling = data["schooling"]
    experience = data["experience"]
    distance = data["distance"]

    # --- 1) Short regression (schooling only) ---
    X_short = add_const(schooling)
    res_short = m_ols.estimate(X_short, y)
    print(f"\n[OLS short] Return to schooling: {res_short['beta'][1]:.4f}")
    print(f"  SE: {res_short['se'][1]:.4f}")
    print(f"  (Biased upward by omitted ability)")

    # --- 2) Long regression (schooling + experience + experience^2) ---
    X_long = add_const(np.column_stack([
        schooling, experience, experience ** 2
    ]))
    res_long = m_ols.estimate(X_long, y)
    print(f"\n[OLS long] Return to schooling: {res_long['beta'][1]:.4f}")
    print(f"  SE: {res_long['se'][1]:.4f}")

    # --- 3) FWL: verify "controlling for experience" ---
    fwl_check = m_fwl.verify_fwl(y, X_long, idx_interest=1)
    print(f"\n[FWL] Full = {fwl_check['full_coef']:.6f}, "
          f"FWL = {fwl_check['fwl_coef']:.6f}, Match = {fwl_check['match']}")

    # --- 4) Robust SEs ---
    het_result = m_het.estimate_with_robust_se(X_long, y)
    print(f"\n[Robust SE] Classical: {het_result['se_classical'][1]:.4f}, "
          f"HC1: {het_result['se_robust'][1]:.4f}")
    print(f"  Breusch-Pagan p-value: {het_result['bp_test']['p_value']:.4f}")

    # --- 5) IV / 2SLS using distance as instrument ---
    Z = add_const(distance)
    iv_result = m_iv.estimate_2sls(Z, schooling, y)
    print(f"\n[IV/2SLS] Return to schooling: {iv_result['beta'][1]:.4f}")
    print(f"  First-stage F: {iv_result['F_stat']:.1f}")
    print(f"  Robust SE: {iv_result['se_robust'][1]:.4f}")
    print(f"  True return: {data['true_return']}")

    # --- 6) Bootstrap ---
    boot = m_boot.bootstrap_ols_slope(X_short, y, n_boot=1000)
    print(f"\n[Bootstrap] SE: {boot['se']:.4f}, "
          f"CI: [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
