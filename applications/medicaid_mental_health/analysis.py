"""
Medicaid Expansion and Mental Health Outcomes
==============================================

DiD analysis of the effect of state-level Medicaid expansion on
mental health outcomes, using methods from the PracticalMath methods/ package.

Status: stub -- replace simulated data with real data sources.
"""

import numpy as np
import sys
import os

# Add project root to path so methods package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from methods.utils import ols_fit, add_const
from methods import did as m_did
from methods import panel_fe as m_fe
from methods import heteroskedasticity as m_het


def simulate_medicaid_data(n_states=50, n_years=10, treatment_year=5,
                           n_treated=25, seed=42):
    """
    Simulate a state-year panel mimicking Medicaid expansion.

    Returns
    -------
    dict with arrays: y, treated, post, state_ids, year_ids
    """
    np.random.seed(seed)
    N = n_states * n_years
    state_ids = np.repeat(np.arange(n_states), n_years)
    year_ids = np.tile(np.arange(n_years), n_states)

    # State fixed effects
    alpha = np.repeat(np.random.normal(0, 2, n_states), n_years)
    # Time trend
    gamma = np.tile(np.arange(n_years) * 0.3, n_states)
    # Treatment assignment
    treated = np.repeat(
        (np.arange(n_states) < n_treated).astype(float), n_years
    )
    post = (year_ids >= treatment_year).astype(float)
    # True treatment effect = 2.0
    tau_true = 2.0
    effect = tau_true * treated * post
    y = alpha + gamma + effect + np.random.normal(0, 1.5, N)

    return dict(
        y=y, treated=treated, post=post,
        state_ids=state_ids, year_ids=year_ids,
        tau_true=tau_true,
    )


def main():
    print("=" * 60)
    print("Medicaid Expansion & Mental Health -- DiD Analysis")
    print("=" * 60)

    data = simulate_medicaid_data()

    # --- 1) Basic 2x2 DiD ---
    did_result = m_did.estimate_did(data["y"], data["treated"], data["post"])
    print(f"\n[DiD] tau_hat = {did_result['tau_did']:.3f}")
    print(f"  Homoskedastic SE: {did_result['se_classical'][3]:.4f}")
    print(f"  Robust SE:        {did_result['se_robust'][3]:.4f}")
    print(f"  True effect:      {data['tau_true']}")

    # --- 2) Fixed Effects ---
    treatment_indicator = data["treated"] * data["post"]
    fe_result = m_fe.estimate_fe(
        data["y"], treatment_indicator, data["state_ids"]
    )
    print(f"\n[FE] beta_hat = {fe_result['beta_fe']:.3f}")
    print(f"  Clustered SE: {fe_result['se_cluster']:.4f}")

    # --- 3) Event Study ---
    treatment_time = 5
    es_result = m_did.event_study(
        data["y"], data["treated"], data["year_ids"], treatment_time
    )
    print("\n[Event Study] Dynamic treatment effects:")
    for t, c, s in zip(es_result["rel_times"], es_result["coefs"],
                       es_result["ses"]):
        sig = "*" if abs(c) > 1.96 * s and s > 0 else ""
        print(f"  t={t:+d}: tau_hat={c:+.3f}  SE={s:.3f} {sig}")


if __name__ == "__main__":
    main()
