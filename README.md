# PracticalMath

A small, growing portfolio of "math-first" applied statistics / econometrics code — written for people who are fluent in linear algebra, probability, and calculus, and want to use that toolkit to reason about messy real-world causality (especially in social/economic/political systems).

---

## Repository Structure

```
PracticalMath/
├── README.md
├── requirements.txt
├── methods/                    # From-scratch econometric estimators
│   ├── __init__.py
│   ├── utils.py                # Shared helpers (ols_fit, add_const)
│   ├── ols.py                  # OLS estimation, OVB Monte Carlo
│   ├── fwl.py                  # Frisch-Waugh-Lovell partialing out
│   ├── heteroskedasticity.py   # Breusch-Pagan test, HC1 robust SEs
│   ├── iv_2sls.py              # 2SLS, Wald estimator, correct SEs
│   ├── panel_fe.py             # Within estimator, clustered SEs
│   ├── did.py                  # DiD estimator, event study
│   ├── rdd.py                  # RDD local linear, McCrary, bias-correction
│   ├── binary_outcomes.py      # Logit, Probit MLE, AME + SEs
│   ├── mle.py                  # Generic MLE framework, profile likelihood
│   └── bootstrap.py            # Nonparametric bootstrap inference
├── applications/               # Real-data projects using methods/
│   ├── medicaid_mental_health/
│   │   ├── README.md
│   │   ├── analysis.py         # DiD + FE analysis
│   │   └── figures/
│   └── mincer_cps/
│       ├── README.md
│       ├── analysis.py         # OLS + IV Mincer equation
│       └── figures/
├── ECONOMETRICS PRIMER FOR THE MATH-LITERATE WITH VISUALIZATIONS.py
└── econometrics_visual_primer.pdf
```

---

## What's in here

### 1) `methods/` — Reusable Econometric Estimators

All 10 core methods implemented from scratch using only numpy / scipy.
Each module exposes clean functions that can be imported into any project:

| Module                  | What it provides                                              |
|-------------------------|---------------------------------------------------------------|
| `ols.py`                | OLS estimation, OVB formula, Monte Carlo bias demonstration   |
| `fwl.py`                | FWL partialing out, verification of "controlling for X"       |
| `heteroskedasticity.py` | Breusch-Pagan test, HC1 robust standard errors                |
| `iv_2sls.py`            | Two-stage least squares, Wald estimator, correct 2SLS SEs     |
| `panel_fe.py`           | Within (demeaning) estimator, Arellano clustered SEs          |
| `did.py`                | 2x2 DiD, event study dynamic effects, group means             |
| `rdd.py`                | Local linear RDD, McCrary density test, bias-corrected CI     |
| `binary_outcomes.py`    | Logit/Probit MLE, AME with delta-method & bootstrap SEs       |
| `mle.py`                | Generic MLE optimizer, Hessian SEs, profile likelihood         |
| `bootstrap.py`          | Nonparametric bootstrap for any estimator                      |

**Example usage:**

```python
from methods.utils import add_const
from methods import ols, iv_2sls, bootstrap

# OLS
X = add_const(schooling)
result = ols.estimate(X, wages)
print(result["beta"], result["se"])

# IV / 2SLS
Z = add_const(distance)
iv_result = iv_2sls.estimate_2sls(Z, schooling, wages)
print(iv_result["beta"], iv_result["F_stat"])

# Bootstrap
boot = bootstrap.bootstrap_ols_slope(X, wages, n_boot=2000)
print(boot["se"], boot["ci_lo"], boot["ci_hi"])
```

### 2) `applications/` — Real-Data Projects

Applied projects that import from `methods/` and demonstrate the
estimators on (simulated, pending real) data:

- **`medicaid_mental_health/`** — DiD analysis of Medicaid expansion
  on mental health outcomes (state-year panel)
- **`mincer_cps/`** — Mincer earnings equation estimating the return
  to schooling via OLS, IV, and bootstrap on CPS microdata

### 3) Econometrics Primer (code + PDF)

- **Script:** `ECONOMETRICS PRIMER FOR THE MATH-LITERATE WITH VISUALIZATIONS.py`
- **Output PDF (already included):** `econometrics_visual_primer.pdf`

The primer imports from `methods/` and serves as both a teaching tool
and integration test. It simulates data, runs all 10 methods, generates
figures, and compiles a polished PDF.

The primer is intentionally "math-forward":
- brief economic story → formal setup → estimator/inference → simulation + plots
- estimators implemented via linear algebra / optimization (no black-box econometrics packages)
- visual explanations (including DAG-style intuition where useful)

**Covered methods (current sections):**
1. OLS (and omitted variable bias)
2. Frisch–Waugh–Lovell (partial regression / partialing out)
3. Heteroskedasticity (detection + robust standard errors)
4. Instrumental Variables (IV / 2SLS)
5. Panel Data (Fixed Effects / within estimator)
6. Difference-in-Differences (DiD) + event study intuition
7. Regression Discontinuity Design (RDD)
8. Binary outcomes (Logit / Probit) + AME intuition
9. Maximum Likelihood Estimation (from scratch)
10. Bootstrap inference

---

## Quick start

### Option A — read the PDF
Open `econometrics_visual_primer.pdf` directly in GitHub.

### Option B — regenerate everything locally

```bash
# 1) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 2) Install pinned dependencies
pip install -r requirements.txt

# 3) Run the primer (generates figures + PDF)
python "ECONOMETRICS PRIMER FOR THE MATH-LITERATE WITH VISUALIZATIONS.py"

# 4) Run an application
python applications/mincer_cps/analysis.py
python applications/medicaid_mental_health/analysis.py
```

This will generate:
- A set of `fig*.png` visualizations in the repo directory
- `econometrics_visual_primer.pdf` compiled from interleaved text + figures

### Option C — use methods as a library

```python
import sys; sys.path.insert(0, "/path/to/PracticalMath")
from methods import ols, did, bootstrap
# ... use in your own analysis
```
