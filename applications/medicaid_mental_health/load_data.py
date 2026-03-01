"""
Real data loaders for the Medicaid Expansion & Mental Health analysis.
=====================================================================

Replaces simulated data with publicly available government data sources:

1. **Medicaid expansion dates** -- Hard-coded from KFF tracking of state
   expansion decisions (public knowledge, sourced from CMS/state records).

2. **Mental-health outcomes (NSDUH)** -- SAMHSA National Survey on Drug Use
   and Health, Small Area Estimation (SAE) state prevalence tables.
   CSV ZIP files downloaded from:
   https://www.samhsa.gov/data/nsduh/state-reports

3. **Medicaid enrollment (CMS)** -- Monthly Medicaid & CHIP enrollment
   counts by state, fetched from the data.medicaid.gov DKAN API.

Usage
-----
Option A -- Fully automated (enrollment data only, no manual download):

    data = load_cms_enrollment_panel()

Option B -- Full analysis with NSDUH mental health outcomes:

    1. Download NSDUH SAE CSVs (see README for URLs)
    2. Place ZIP or extracted CSVs in  data/nsduh/
    3. Call:  data = load_nsduh_panel()

Both return a dict compatible with the methods/ package:
    {y, treated, post, state_ids, year_ids}
"""

import csv
import io
import json
import os
import re
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import numpy as np

# ---------------------------------------------------------------------------
# Medicaid expansion dates (year coverage began) -- from KFF
# https://www.kff.org/affordable-care-act/state-indicator/
#   state-activity-around-expanding-medicaid-under-the-affordable-care-act/
#
# States NOT listed here have not expanded as of 2025.
# ---------------------------------------------------------------------------
EXPANSION_YEAR = {
    "Arizona": 2014, "Arkansas": 2014, "California": 2014,
    "Colorado": 2014, "Connecticut": 2014, "Delaware": 2014,
    "District of Columbia": 2014, "Hawaii": 2014, "Illinois": 2014,
    "Iowa": 2014, "Kentucky": 2014, "Maryland": 2014,
    "Massachusetts": 2014, "Minnesota": 2014, "Nevada": 2014,
    "New Jersey": 2014, "New Mexico": 2014, "New York": 2014,
    "North Dakota": 2014, "Ohio": 2014, "Oregon": 2014,
    "Rhode Island": 2014, "Vermont": 2014, "Washington": 2014,
    "West Virginia": 2014,
    "Michigan": 2014,          # April 2014
    "New Hampshire": 2014,     # August 2014
    "Pennsylvania": 2015,      # January 2015
    "Indiana": 2015,           # February 2015
    "Alaska": 2015,            # September 2015
    "Montana": 2016,           # January 2016
    "Louisiana": 2016,         # July 2016
    "Virginia": 2019,          # January 2019
    "Maine": 2019,             # January 2019
    "Idaho": 2020,             # January 2020
    "Utah": 2020,              # January 2020
    "Nebraska": 2020,          # October 2020
    "Missouri": 2021,          # July 2021
    "Oklahoma": 2021,          # July 2021
    "South Dakota": 2023,      # July 2023
    "North Carolina": 2023,    # December 2023
}

ALL_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California",
    "Colorado", "Connecticut", "Delaware", "District of Columbia",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana",
    "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin",
    "Wyoming",
]

DATA_DIR = Path(__file__).parent / "data"
ACA_EXPANSION_START_YEAR = min(EXPANSION_YEAR.values())


def _resolve_nsduh_dir(nsduh_dir=None):
    """Resolve the NSDUH directory from common project locations."""
    if nsduh_dir is not None:
        p = Path(nsduh_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        DATA_DIR / "nsduh",
        repo_root / "data" / "nsduh",
        DATA_DIR,
        repo_root / "data",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Auto-create default directory so downstream loader can populate it.
    default_dir = repo_root / "data" / "nsduh"
    default_dir.mkdir(parents=True, exist_ok=True)
    return default_dir

# ---------------------------------------------------------------------------
# NSDUH State Prevalence Estimates -- CSV loader
# ---------------------------------------------------------------------------

# NSDUH SAE CSV ZIP download pages (manual download required):
NSDUH_DOWNLOAD_URLS = {
    "2021-2022": "https://www.samhsa.gov/data/nsduh/state-reports-NSDUH-2022",
    "2022-2023": "https://www.samhsa.gov/data/data-we-collect/nsduh-national-survey-drug-use-and-health/state-releases/2022-2023",
    "2023-2024": "https://www.samhsa.gov/data/nsduh/state-reports",
}

NSDUH_STATE_RELEASES_INDEX = (
    "https://www.samhsa.gov/data/data-we-collect/"
    "nsduh-national-survey-drug-use-and-health/state-releases"
)

# Table numbers for key mental health measures in NSDUH SAE files.
# These table numbers correspond to CSV filenames inside the ZIP.
# Table mappings based on the 2021-2022 and later SAE releases:
NSDUH_MENTAL_HEALTH_TABLES = {
    "ami": "NSDUHsaePercents2022Tab25",       # Any Mental Illness 18+
    "smi": "NSDUHsaePercents2022Tab27",       # Serious Mental Illness 18+
    "mde": "NSDUHsaePercents2022Tab29",       # Major Depressive Episode 18+
    "mh_treatment": "NSDUHsaePercents2022Tab33",  # Received MH Treatment 18+
}

NSDUH_MEASURE_MAP = {
    "ami": "AMI",
    "smi": "SMI",
    "mde": "MDE",
    "mh_treatment": "MHT",
    "mht": "MHT",
}


def _build_panel_from_records(records, state_list, outcome_label):
    """Build methods-compatible arrays from a (state, year, value) record list."""
    arr = np.array(records)
    state_ids = arr[:, 0].astype(int)
    year_ids = arr[:, 1].astype(int)
    y = arr[:, 2]

    treated = np.zeros(len(y))
    post = np.zeros(len(y))
    for i, (sid, yr) in enumerate(zip(state_ids, year_ids)):
        state_name = state_list[sid]
        exp_year = EXPANSION_YEAR.get(state_name)
        if exp_year is not None:
            treated[i] = 1.0
            post[i] = float(yr >= exp_year)

    return dict(
        y=y,
        treated=treated,
        post=post,
        state_ids=state_ids,
        year_ids=year_ids,
        state_names=state_list,
        years=np.array(sorted(set(year_ids))),
        outcome_label=outcome_label,
    )


def _load_fetcher_derived_panel(nsduh_dir, outcome, year_range):
    """
    Load output produced by fetch_nsduh_state_prevalence.py.

    Expected files include:
      - derived/nsduh_state_mental_health_prevalence_long.csv
      - nsduh_state_mental_health_prevalence_long.csv
    """
    nsduh_path = Path(nsduh_dir)
    candidates = [
        nsduh_path / "derived" / "nsduh_state_mental_health_prevalence_long.csv",
        nsduh_path / "nsduh_state_mental_health_prevalence_long.csv",
    ]

    outcome_key = NSDUH_MEASURE_MAP.get(outcome.lower(), outcome.upper())
    start_yr, end_yr = year_range

    state_list = sorted(ALL_STATES)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    for path in candidates:
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = {name.lower().strip(): name for name in (reader.fieldnames or [])}
            if not {"state", "measure", "estimate"}.issubset(fields):
                continue

            state_col = fields["state"]
            measure_col = fields["measure"]
            estimate_col = fields["estimate"]
            year_col = fields.get("period_end_year") or fields.get("year")

            records = []
            for row in reader:
                state = row.get(state_col, "").strip()
                if state not in state_to_idx:
                    continue

                if row.get(measure_col, "").strip().upper() != outcome_key:
                    continue

                try:
                    val = float(row.get(estimate_col, "").strip().replace(",", ""))
                except ValueError:
                    continue

                if year_col:
                    try:
                        yr = int(row.get(year_col, "").strip())
                    except ValueError:
                        continue
                else:
                    # Fallback: use second year from a year_pair like 2023-2024.
                    year_pair = row.get(fields.get("year_pair", ""), "").strip()
                    try:
                        yr = int(year_pair.split("-")[-1])
                    except Exception:
                        continue

                if start_yr <= yr <= end_yr:
                    records.append((state_to_idx[state], yr, val))

        if records:
            print(f"  [NSDUH] Loaded derived prevalence panel: {path}")
            return _build_panel_from_records(records, state_list, outcome)

    return None


def _find_nsduh_csv(nsduh_dir, table_prefix):
    """Search for an NSDUH SAE CSV by table prefix in the given directory."""
    nsduh_path = Path(nsduh_dir)
    # Try direct CSVs anywhere under the directory tree.
    for f in sorted(nsduh_path.rglob("*.csv")):
        if table_prefix.lower() in f.stem.lower():
            return f

    # Try extracting from ZIP
    for z in sorted(nsduh_path.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if table_prefix.lower() in name.lower() and name.endswith(".csv"):
                    target = nsduh_path / "extracted" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, open(target, "wb") as dst:
                        dst.write(src.read())
                    return target
    return None


def _normalize_col_name(name):
    name = (name or "").strip().lower()
    name = re.sub(r"[%\(\)\[\]\-]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _find_column(columns, patterns):
    for col in columns:
        norm = _normalize_col_name(col)
        for pat in patterns:
            if re.search(pat, norm):
                return col
    return None


def _to_float(val):
    text = str(val or "").strip()
    if text == "" or text.lower() in {"nan", "na", "n/a", "suppressed", "not available"}:
        return None
    text = text.replace("%", "")
    text = re.sub(r"[^\d\.\-]+", "", text)
    if text in {"", "-", "."}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_nsduh_sae_csv(csv_path):
    """
    Parse an NSDUH SAE prevalence CSV file into a dict of {state: estimate}.

    NSDUH SAE CSVs typically have columns:
      State, 18+ Estimate, 18+ 95% CI (Lower), 18+ 95% CI (Upper), ...
    with some header rows. This parser is flexible and looks for the
    state name column and the first numeric estimate column.

    Returns
    -------
    dict : {state_name: float} -- prevalence percentage for each state
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read().replace("\x00", "")

    lines = text.splitlines()

    # Find the header row (first row that contains "State" or "Order")
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if "state" in low and ("estimate" in low or "percent" in low or "total" in low):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"Could not find header row in {csv_path}")

    sample = "\n".join(lines[header_idx: header_idx + 10])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except Exception:
        delimiter = ","

    reader = csv.DictReader(lines[header_idx:], delimiter=delimiter)
    fieldnames = reader.fieldnames or []

    # Identify columns robustly (NSDUH layouts vary by release)
    state_col = _find_column(fieldnames, [r"\bstate\b", r"\bgeography\b"])
    estimate_col = _find_column(
        fieldnames,
        [
            r"\bestimate\b",
            r"\bpercent\b",
            r"\bpercentage\b",
            r"\bvalue\b",
        ],
    )

    if state_col is None and fieldnames:
        state_col = fieldnames[0]
    if estimate_col is None and len(fieldnames) > 1:
        estimate_col = fieldnames[1]
    if state_col is None or estimate_col is None:
        raise ValueError(f"Could not detect state/estimate columns in {csv_path}")

    results = {}
    for row in reader:
        state = row[state_col].strip()
        val = _to_float(row.get(estimate_col))
        if state in ALL_STATES or state == "District of Columbia":
            if val is not None:
                results[state] = val
    return results


def _parse_generated_nsduh_panel_csv(csv_path):
    """
    Parse a generated NSDUH panel CSV with columns like:
    state, year, value/outcome/estimate.
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        state_col = None
        year_col = None
        value_col = None
        for col in fieldnames:
            cl = col.lower().strip()
            if state_col is None and "state" in cl:
                state_col = col
            if year_col is None and "year" in cl:
                year_col = col
            if value_col is None and any(k in cl for k in ("value", "outcome", "estimate", "percent")):
                value_col = col

        if state_col is None or year_col is None or value_col is None:
            return None

        state_list = sorted(ALL_STATES)
        state_to_idx = {s: i for i, s in enumerate(state_list)}
        records = []
        for row in reader:
            state = row.get(state_col, "").strip()
            if state not in state_to_idx:
                continue

            try:
                yr = int(row.get(year_col, "").strip())
            except ValueError:
                continue

            val_str = row.get(value_col, "").strip().replace(",", "")
            try:
                val = float(val_str)
            except ValueError:
                continue

            records.append((state_to_idx[state], yr, val))

    if not records:
        return None

    return _build_panel_from_records(records, state_list, "nsduh_generated_csv")


def _auto_prepare_nsduh_data(nsduh_dir):
    """Placeholder for automatic NSDUH data preparation (not yet implemented)."""
    pass


def load_nsduh_panel(
    nsduh_dir=None,
    outcome="ami",
    year_range=(2010, 2022),
):
    """
    Build a state-year panel from NSDUH SAE CSV files stored locally.

    Expects a directory containing subdirectories or ZIP files for each
    two-year release (e.g., ``NSDUHsaePercents2014.csv``). Each file maps
    to the *second* year of the two-year window (2013-2014 -> year 2014).

    Parameters
    ----------
    nsduh_dir : str or Path, optional
        Directory with NSDUH SAE CSV or ZIP files. Defaults to data/nsduh/.
    outcome : str
        Mental health measure key (see NSDUH_MENTAL_HEALTH_TABLES).
    year_range : tuple
        (start_year, end_year) inclusive.

    Returns
    -------
    dict with arrays: y, treated, post, state_ids, year_ids, state_names,
                      years, outcome_label
    """
    nsduh_dir = _resolve_nsduh_dir(nsduh_dir)

    # If no local NSDUH CSVs are detected, auto-download and extract them.
    _auto_prepare_nsduh_data(nsduh_dir)

    # Support pre-generated panel CSV files for direct analysis.
    for generated_name in (
        "nsduh_panel.csv",
        "nsduh_generated.csv",
        "nsduh.csv",
    ):
        generated_path = nsduh_dir / generated_name
        if generated_path.exists():
            panel = _parse_generated_nsduh_panel_csv(generated_path)
            if panel is not None:
                print(f"  [NSDUH] Loaded generated panel CSV: {generated_path}")
                return panel

    start_yr, end_yr = year_range

    # Prefer tidy panel output generated by fetch_nsduh_state_prevalence.py.
    derived_panel = _load_fetcher_derived_panel(nsduh_dir, outcome, year_range)
    if derived_panel is not None:
        return derived_panel

    years = list(range(start_yr, end_yr + 1))

    # Collect state-year observations
    state_list = sorted(ALL_STATES)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    records = []  # (state_idx, year, y_value)

    for yr in years:
        # NSDUH SAE files are named like NSDUHsaePercents{YEAR}Tab25.csv
        # Try to find a file matching this year and the requested table
        table_base = NSDUH_MENTAL_HEALTH_TABLES.get(outcome, outcome)
        # Replace the year suffix
        table_search = table_base.replace("2022", str(yr))

        csv_path = _find_nsduh_csv(nsduh_dir, table_search)
        if csv_path is None:
            # Also try a generic search by year
            csv_path = _find_nsduh_csv(nsduh_dir, f"saePercents{yr}")

        if csv_path is None:
            print(f"  [NSDUH] No data found for year {yr}, skipping")
            continue

        state_vals = parse_nsduh_sae_csv(csv_path)
        for state, val in state_vals.items():
            if state in state_to_idx:
                records.append((state_to_idx[state], yr, val))

    if not records:
        raise FileNotFoundError(
            "No NSDUH data files found. See README for download instructions."
        )

    return _build_panel_from_records(records, state_list, outcome)


# ---------------------------------------------------------------------------
# CMS Medicaid Enrollment -- data.medicaid.gov API
# ---------------------------------------------------------------------------

CMS_ENROLLMENT_DATASET = "6165f45b-ca93-5bb5-9d06-db29c692a360"
CMS_API_BASE = "https://data.medicaid.gov/api/1/datastore/query"
CMS_CSV_URL = (
    f"{CMS_API_BASE}/{CMS_ENROLLMENT_DATASET}/0/download?format=csv"
)
# Fallback: Socrata-style endpoint (older)
CMS_SOCRATA_CSV_URL = (
    "https://data.medicaid.gov/api/views/n5ce-jxme/rows.csv"
    "?accessType=DOWNLOAD"
)


def fetch_cms_enrollment_csv(cache_path=None):
    """
    Download Medicaid & CHIP monthly enrollment data from data.medicaid.gov.

    Parameters
    ----------
    cache_path : str or Path, optional
        If provided, cache the downloaded CSV here and reuse on next call.

    Returns
    -------
    str : raw CSV text
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8")

    for url in [CMS_CSV_URL, CMS_SOCRATA_CSV_URL]:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "PracticalMath-Research/1.0"
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(raw, encoding="utf-8")
                return raw
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  [CMS] Failed to fetch from {url}: {e}")
            continue

    raise ConnectionError(
        "Could not download CMS enrollment data. Check your network "
        "connection or download manually from:\n"
        "  https://data.medicaid.gov/dataset/"
        f"{CMS_ENROLLMENT_DATASET}\n"
        "and save to data/cms_enrollment.csv"
    )


def parse_cms_enrollment(csv_text):
    """
    Parse CMS enrollment CSV into {(state, year): enrollment_count}.

    The CSV has columns like: State, Year, Month, Total Medicaid Enrollment,
    etc. We aggregate to annual averages per state.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames

    # Find relevant columns (names vary across dataset versions)
    state_col = None
    year_col = None
    month_col = None
    enrollment_col = None

    for col in fieldnames:
        cl = col.lower().strip()
        if "state" in cl and "name" in cl:
            state_col = col
        elif cl == "state":
            state_col = state_col or col
        elif "year" in cl and year_col is None:
            year_col = col
        elif "month" in cl and month_col is None:
            month_col = col
        elif ("total" in cl and "enroll" in cl) or ("medicaid" in cl and "enroll" in cl):
            enrollment_col = col

    if state_col is None:
        # Try first column
        state_col = fieldnames[0]

    # Collect monthly values -> aggregate annually
    monthly = {}  # (state, year) -> [values]
    for row in reader:
        state = row.get(state_col, "").strip()
        if state not in ALL_STATES and state != "District of Columbia":
            continue

        yr_str = row.get(year_col, "").strip() if year_col else ""
        try:
            yr = int(yr_str)
        except ValueError:
            continue

        val_str = (
            row.get(enrollment_col, "0").strip().replace(",", "")
            if enrollment_col else "0"
        )
        try:
            val = float(val_str)
        except ValueError:
            continue

        key = (state, yr)
        monthly.setdefault(key, []).append(val)

    # Average monthly values to get annual enrollment
    annual = {}
    for (state, yr), vals in monthly.items():
        annual[(state, yr)] = np.mean(vals)

    return annual


def load_cms_enrollment_panel(
    cache_path=None,
    year_range=None,
    log_enrollment=True,
):
    """
    Build a state-year panel using CMS Medicaid enrollment as the outcome.

    The outcome variable is log(annual average enrollment), which is
    appropriate for studying the *level* effect of expansion on take-up.

    Parameters
    ----------
    cache_path : str or Path, optional
        Path to cache downloaded CSV. Defaults to data/cms_enrollment.csv.
    year_range : tuple, optional
        (start_year, end_year) inclusive. Defaults to range in data.
    log_enrollment : bool
        If True, use log(enrollment) as outcome. Default True.

    Returns
    -------
    dict with arrays: y, treated, post, state_ids, year_ids, state_names,
                      years, outcome_label
    """
    if cache_path is None:
        cache_path = DATA_DIR / "cms_enrollment.csv"

    csv_text = fetch_cms_enrollment_csv(cache_path)
    annual = parse_cms_enrollment(csv_text)

    if not annual:
        raise ValueError(
            "No enrollment data parsed. The CMS CSV format may have changed. "
            "Check the raw file at: " + str(cache_path)
        )

    # Determine year range from data if not specified
    all_years = sorted(set(yr for _, yr in annual.keys()))
    if year_range:
        all_years = [y for y in all_years if year_range[0] <= y <= year_range[1]]

    state_list = sorted(ALL_STATES)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    records = []
    for state in state_list:
        for yr in all_years:
            val = annual.get((state, yr))
            if val is not None and val > 0:
                outcome = np.log(val) if log_enrollment else val
                records.append((state_to_idx[state], yr, outcome))

    records = np.array(records)
    state_ids = records[:, 0].astype(int)
    year_ids = records[:, 1].astype(int)
    y = records[:, 2]

    treated = np.zeros(len(y))
    post = (year_ids >= ACA_EXPANSION_START_YEAR).astype(float)
    for i, (sid, yr) in enumerate(zip(state_ids, year_ids)):
        state_name = state_list[sid]
        exp_year = EXPANSION_YEAR.get(state_name)
        if exp_year is not None:
            treated[i] = 1.0

    label = "log_medicaid_enrollment" if log_enrollment else "medicaid_enrollment"
    return dict(
        y=y,
        treated=treated,
        post=post,
        state_ids=state_ids,
        year_ids=year_ids,
        state_names=state_list,
        years=np.array(all_years),
        outcome_label=label,
    )


# ---------------------------------------------------------------------------
# BRFSS State-Year Panel -- from brfss_extract_with_states_v3.py output
# ---------------------------------------------------------------------------

BRFSS_OUTCOME_COLUMNS = {
    "freq_mental_distress": "freq_mental_distress_mean",
    "mh_days_not_good": "mh_days_not_good_mean",
    "any_mh_days": "any_mh_days_mean",
    "depression_dx": "depression_dx_mean",
    "has_insurance": "has_insurance_mean",
    "cost_barrier": "cost_barrier_mean",
    "has_medicaid": "has_medicaid_mean",
}


def _resolve_brfss_panel_path(brfss_dir=None):
    """Find brfss_state_year_panel.csv in common project locations."""
    if brfss_dir is not None:
        return Path(brfss_dir) / "brfss_state_year_panel.csv"

    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parents[1]
    candidates = [
        app_dir / "brfss_output" / "brfss_state_year_panel.csv",
        repo_root / "brfss_output" / "brfss_state_year_panel.csv",
        DATA_DIR / "brfss" / "brfss_state_year_panel.csv",
        repo_root / "data" / "brfss" / "brfss_state_year_panel.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Default path (will trigger FileNotFoundError downstream)
    return candidates[0]


def load_brfss_panel(
    brfss_dir=None,
    outcome="freq_mental_distress",
    year_range=(2010, 2023),
):
    """
    Build a state-year panel from BRFSS data extracted by
    brfss_extract_with_states_v3.py.

    Parameters
    ----------
    brfss_dir : str or Path, optional
        Directory containing brfss_state_year_panel.csv.
        Searched automatically if not specified.
    outcome : str
        Outcome variable. One of: freq_mental_distress, mh_days_not_good,
        any_mh_days, depression_dx, has_insurance, cost_barrier, has_medicaid.
    year_range : tuple
        (start_year, end_year) inclusive.

    Returns
    -------
    dict with arrays: y, treated, post, state_ids, year_ids, state_names,
                      years, outcome_label
    """
    panel_path = _resolve_brfss_panel_path(brfss_dir)

    if not panel_path.exists():
        raise FileNotFoundError(
            f"BRFSS state-year panel not found at {panel_path}. "
            "Run brfss_extract_with_states_v3.py first to generate the data."
        )

    outcome_col = BRFSS_OUTCOME_COLUMNS.get(outcome)
    if outcome_col is None:
        raise ValueError(
            f"Unknown BRFSS outcome '{outcome}'. "
            f"Available: {list(BRFSS_OUTCOME_COLUMNS.keys())}"
        )

    start_yr, end_yr = year_range

    state_set = set()
    raw_records = []

    with open(panel_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if outcome_col not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{outcome_col}' not found in {panel_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            state_name = row.get("state_name", "").strip()
            if not state_name:
                continue
            # Normalize: accept both ALL_STATES entries and DC variants
            if state_name not in ALL_STATES and state_name != "District of Columbia":
                continue

            try:
                yr = int(row.get("year", "").strip())
            except ValueError:
                continue

            if yr < start_yr or yr > end_yr:
                continue

            val_str = row.get(outcome_col, "").strip()
            if not val_str or val_str.lower() in ("nan", "na", ""):
                continue

            try:
                val = float(val_str)
            except ValueError:
                continue

            state_set.add(state_name)
            raw_records.append((state_name, yr, val))

    if not raw_records:
        raise FileNotFoundError(
            f"No valid records found in {panel_path} for outcome '{outcome}' "
            f"in year range {year_range}."
        )

    state_list = sorted(state_set)
    state_to_idx = {s: i for i, s in enumerate(state_list)}

    records = []
    for state_name, yr, val in raw_records:
        records.append((state_to_idx[state_name], yr, val))

    print(f"  [BRFSS] Loaded {len(records)} state-year observations "
          f"from {panel_path.name}")
    return _build_panel_from_records(records, state_list, f"brfss_{outcome}")


# ---------------------------------------------------------------------------
# Convenience: build a panel from whatever data sources are available
# ---------------------------------------------------------------------------

def load_real_data(prefer="nsduh", **kwargs):
    """
    Attempt to load real data, trying available sources in order.

    Parameters
    ----------
    prefer : str
        Which source to try first: "brfss", "nsduh", or "cms".
    **kwargs : dict
        Passed through to the chosen loader.

    Returns
    -------
    dict : panel data compatible with methods/ package
    """
    loaders = {
        "brfss": load_brfss_panel,
        "nsduh": load_nsduh_panel,
        "cms": load_cms_enrollment_panel,
    }
    order = [prefer] + [k for k in loaders if k != prefer]

    last_error = None
    for source in order:
        try:
            print(f"[load_data] Trying {source.upper()} source...")
            data = loaders[source](**kwargs)
            n = len(data["y"])
            n_states = len(set(data["state_ids"]))
            n_years = len(data["years"])
            print(f"[load_data] Loaded {n} observations "
                  f"({n_states} states x {n_years} years) "
                  f"from {source.upper()}")
            return data
        except (FileNotFoundError, ConnectionError, ValueError, TypeError) as e:
            print(f"[load_data] {source.upper()} unavailable: {e}")
            last_error = e

    raise RuntimeError(
        "No real data sources available. Either:\n"
        "  1. Run brfss_extract_with_states_v3.py to generate BRFSS panel\n"
        "  2. Download NSDUH SAE CSVs to data/nsduh/ (see README)\n"
        "  3. Ensure network access for CMS API download\n"
        "  4. Use simulate_medicaid_data() as a fallback\n"
        f"Last error: {last_error}"
    )
