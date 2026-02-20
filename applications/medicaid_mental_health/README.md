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

### 1. SAMHSA NSDUH State Prevalence Estimates (mental health outcomes)

The National Survey on Drug Use and Health (NSDUH) provides state-level
prevalence estimates for mental health measures via Small Area Estimation (SAE).

**Measures available:**
- Any Mental Illness (AMI) among adults 18+
- Serious Mental Illness (SMI) among adults 18+
- Major Depressive Episode (MDE) among adults 18+
- Received Mental Health Treatment among adults 18+

**Download instructions:**
1. Visit the NSDUH State Releases page:
   https://www.samhsa.gov/data/nsduh/state-reports
2. Download the "State Prevalence Tables: CSV (ZIP)" for each year-pair
3. Place the ZIP files (or extracted CSVs) in `data/nsduh/`

Available releases (2-year rolling windows):
- 2023-2024: https://www.samhsa.gov/data/nsduh/state-reports
- 2022-2023: https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/state-releases/2022-2023
- 2021-2022: https://www.samhsa.gov/data/nsduh/state-reports-NSDUH-2022

Note: Methodology changed in 2021-2022; earlier releases use a different
SAE model and are not directly comparable to 2021+ estimates.

### 2. CMS Medicaid & CHIP Enrollment Data (enrollment outcomes)

Monthly state-level Medicaid enrollment counts from CMS, available via the
data.medicaid.gov open-data API. The code in `load_data.py` fetches this
automatically.

**Source:** https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360

**Manual download (if API is unavailable):**
- MBES enrollment files: https://www.medicaid.gov/medicaid/national-medicaid-chip-program-information/medicaid-chip-enrollment-data/medicaid-enrollment-data-collected-through-mbes
- Download CSV and save to `data/cms_enrollment.csv`

### 3. Medicaid Expansion Dates by State (treatment assignment)

Hard-coded in `load_data.py` from KFF tracking of state expansion decisions.

**Source:** https://www.kff.org/affordable-care-act/state-indicator/state-activity-around-expanding-medicaid-under-the-affordable-care-act/

41 states (including DC) have adopted expansion as of 2025. Expansion
years range from 2014 (first wave, 27 states) through 2023 (South Dakota,
North Carolina).

### 4. Additional / Alternative Sources

- **CDC BRFSS** -- "Poor mental health days" by state from the Behavioral
  Risk Factor Surveillance System. Annual microdata available at:
  https://www.cdc.gov/brfss/annual_data/annual_data.htm
  Pre-aggregated state estimates via the BRFSS Prevalence & Trends tool:
  https://www.cdc.gov/brfss/brfssprevalence/index.html

- **HCUP / HCUPnet** -- State-level hospitalization statistics by diagnosis
  (including mental health). Free query tool (aggregated statistics):
  https://datatools.ahrq.gov/hcupnet/
  Note: Full microdata (State Inpatient Databases) require purchase.

- **KFF State Health Facts** -- Medicaid enrollment, spending, and other
  state-level health indicators with CSV export:
  https://www.kff.org/statedata/

## Methods Used

- `methods.did.estimate_did` -- 2x2 DiD baseline
- `methods.did.event_study` -- dynamic treatment effects
- `methods.panel_fe.estimate_fe` -- state fixed effects
- `methods.heteroskedasticity.hc1_robust_se` -- robust inference

## Usage

```bash
# Auto-detect available data sources (falls back to simulation)
python analysis.py

# Use specific data source
python analysis.py --source nsduh      # NSDUH mental health outcomes
python analysis.py --source cms        # CMS enrollment data
python analysis.py --source simulate   # Simulated data (for testing)
```

## File Structure

```
medicaid_mental_health/
├── README.md          # This file
├── analysis.py        # Main analysis script (DiD, FE, event study)
├── load_data.py       # Data loaders for NSDUH, CMS, and expansion dates
└── data/              # Local data directory (not checked in)
    ├── nsduh/         # NSDUH SAE CSV/ZIP files (manual download)
    └── cms_enrollment.csv  # Cached CMS enrollment data (auto-downloaded)
```
