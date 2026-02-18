# PracticalMath

A small, growing portfolio of “math-first” applied statistics / econometrics code — written for people who are fluent in linear algebra, probability, and calculus, and want to use that toolkit to reason about messy real-world causality (especially in social/economic/political systems).

Right now the repo centers on an **Econometrics Primer for the Math-Literate (with visualizations)**: a single Python script that **simulates data, implements estimators from scratch, produces figures, and compiles a polished PDF**.

---

## What’s in here

### 1) Econometrics Primer (code + PDF)

- **Script:** `ECONOMETRICS PRIMER FOR THE MATH-LITERATE WITH VISUALIZATIONS.py`  
- **Output PDF (already included):** `econometrics_visual_primer.pdf`

The primer is intentionally “math-forward”:
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

#### 1) Create a virtual environment
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1
2) Install pinned dependencies
pip install -r requirements.txt
3) Run the script
python "ECONOMETRICS PRIMER FOR THE MATH-LITERATE WITH VISUALIZATIONS.py"
This will generate:

a set of fig*.png visualizations in the repo directory

econometrics_visual_primer.pdf compiled from interleaved text + figures
