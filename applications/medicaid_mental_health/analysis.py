"""
Medicaid Expansion and Mental Health Outcomes
==============================================

DiD analysis of the effect of state-level Medicaid expansion on
mental health outcomes, using methods from the PracticalMath methods/ package.

Data sources (see load_data.py for details):
  - SAMHSA NSDUH state prevalence estimates (mental health outcomes)
  - CMS Medicaid enrollment data via data.medicaid.gov (enrollment outcome)
  - KFF Medicaid expansion dates (treatment assignment)

Falls back to simulated data if real sources are unavailable.
"""

import argparse
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path so methods package is importable
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from methods.utils import ols_fit, add_const
from methods import did as m_did
from methods import panel_fe as m_fe
from methods import heteroskedasticity as m_het

from load_data import load_real_data, EXPANSION_YEAR


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


def get_treatment_time(data):
    """
    Determine the modal treatment time for event study specification.

    For real data, this is the median expansion year among treated states.
    For simulated data, it is stored in the data dict.
    """
    if "tau_true" in data:
        # Simulated data: treatment_year is encoded in the year_ids
        return 5

    # Real data: use 2014 (first year of ACA Medicaid expansion)
    return 2014


def main():
    parser = argparse.ArgumentParser(
        description="Medicaid Expansion & Mental Health -- DiD Analysis"
    )
    parser.add_argument(
        "--source", choices=["auto", "nsduh", "cms", "simulate"],
        default="auto",
        help="Data source: 'nsduh' for NSDUH mental health outcomes, "
             "'cms' for CMS enrollment data, 'simulate' for synthetic data, "
             "'auto' to try real sources then fall back to simulation "
             "(default: auto)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Medicaid Expansion & Mental Health -- DiD Analysis")
    print("=" * 60)

    # --- Load data ---
    using_real_data = False
    if args.source == "simulate":
        print("\n[Data] Using simulated data")
        data = simulate_medicaid_data()
    elif args.source in ("nsduh", "cms"):
        data = load_real_data(prefer=args.source)
        using_real_data = True
    else:
        # auto: try real data, fall back to simulation
        try:
            data = load_real_data(prefer="nsduh")
            using_real_data = True
        except RuntimeError:
            print("\n[Data] Real data unavailable, using simulated data")
            data = simulate_medicaid_data()

    n_obs = len(data["y"])
    n_treated = int(data["treated"].sum())
    n_post = int(data["post"].sum())
    print(f"\n[Data] N={n_obs}  treated={n_treated}  post={n_post}")
    if using_real_data:
        print(f"[Data] Outcome: {data.get('outcome_label', 'unknown')}")

    # --- 1) Basic 2x2 DiD ---
    did_result = m_did.estimate_did(data["y"], data["treated"], data["post"])
    print(f"\n[DiD] tau_hat = {did_result['tau_did']:.3f}")
    print(f"  Homoskedastic SE: {did_result['se_classical'][3]:.4f}")
    print(f"  Robust SE:        {did_result['se_robust'][3]:.4f}")
    if "tau_true" in data:
        print(f"  True effect:      {data['tau_true']}")

    # --- 2) Fixed Effects ---
    treatment_indicator = data["treated"] * data["post"]
    fe_result = m_fe.estimate_fe(
        data["y"], treatment_indicator, data["state_ids"]
    )
    print(f"\n[FE] beta_hat = {fe_result['beta_fe']:.3f}")
    print(f"  Clustered SE: {fe_result['se_cluster']:.4f}")

    # --- 3) Event Study ---
    treatment_time = get_treatment_time(data)
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
