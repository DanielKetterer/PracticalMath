# Mincer Earnings Equation on CPS Data

## Estimand

The return to schooling: how much does an additional year of education
raise log wages, controlling for experience?

## Identification Strategy

**OLS with controls** (baseline), compared with:
- **IV / 2SLS** using geographic variation as an instrument for schooling
- **Heteroskedasticity diagnostics** and robust standard errors

This application demonstrates why naive OLS overstates the return to
schooling (omitted ability bias) and how IV corrects for it.

## Data Sources

- Current Population Survey (CPS) Annual Social and Economic Supplement
- March CPS microdata (IPUMS)

## Methods Used

- `methods.ols.estimate` -- baseline Mincer regression
- `methods.ols.ovb_formula` -- quantify omitted variable bias
- `methods.fwl.verify_fwl` -- verify "controlling for experience"
- `methods.heteroskedasticity.estimate_with_robust_se` -- robust inference
- `methods.iv_2sls.estimate_2sls` -- IV with geographic instrument
- `methods.bootstrap.bootstrap_ols_slope` -- bootstrap CIs

## Status

Stub project -- data download and analysis pending.
