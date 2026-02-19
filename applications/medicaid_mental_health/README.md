# Medicaid Expansion and Mental Health Outcomes

## Estimand

The causal effect of state-level Medicaid expansion (under the ACA) on
mental health outcomes such as hospitalization rates, employment among
people with serious mental illness, and crisis-service utilization.

## Identification Strategy

**Difference-in-Differences (DiD):** States that expanded Medicaid
(treatment group) vs. states that did not (control group), comparing
outcomes before and after expansion.

- Event study specification to test parallel pre-trends
- Staggered adoption handled via Callaway & Sant'Anna (2021) estimator

## Data Sources

- CMS Medicaid enrollment files
- SAMHSA National Survey on Drug Use and Health (NSDUH)
- State-level hospitalization data (HCUP)

## Methods Used

- `methods.did.estimate_did` -- 2x2 DiD baseline
- `methods.did.event_study` -- dynamic treatment effects
- `methods.panel_fe.estimate_fe` -- state fixed effects
- `methods.heteroskedasticity.hc1_robust_se` -- robust inference

## Status

Stub project -- data collection and analysis pending.
