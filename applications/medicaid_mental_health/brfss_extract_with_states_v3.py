"""
BRFSS Data Extraction Pipeline for Medicaid Expansion & Mental Health Research
===============================================================================

Purpose: Downloads, harmonizes, and aggregates BRFSS survey data (2010-2023)
         for studying the causal effects of Medicaid expansion on mental health
         outcomes, organized by state and year.

Data source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
    https://www.cdc.gov/brfss/annual_data/annual_data.htm

Output: Two datasets:
    1. brfss_individual_panel.parquet  — individual-level records (millions of rows)
    2. brfss_state_year_panel.csv      — state-year aggregated panel (ready for DiD)

Key variables extracted:
    - Mental health outcomes (poor MH days, frequent mental distress, depression dx)
    - Insurance status (any coverage, Medicaid specifically where available)
    - Demographics for covariate adjustment (age, sex, race, income, education, employment)
    - Medicaid expansion status (merged from policy data)
    - Survey weights for proper population inference

Dependencies: pandas, requests, zipfile, io, os, numpy
    pip install pandas requests numpy pyarrow --break-system-packages

Usage:
    python brfss_extract.py                      # full pipeline, all years
    python brfss_extract.py --years 2012 2016    # specific years only
    python brfss_extract.py --skip-download       # re-process already-downloaded files
    python brfss_extract.py --states OH MI PR   # only selected states / territories

Author: Daniel (PhD thesis pipeline)
Last updated: 2026-02-28
"""

import os
import sys
import io
import zipfile
import argparse
import warnings
from pathlib import Path
from typing import Optional, Iterable, Set

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Years spanning pre-expansion (2010-2013) through post-expansion (2014-2023)
YEARS = list(range(2010, 2024))

# Output directories
DATA_DIR = Path("./brfss_raw")
OUTPUT_DIR = Path("./brfss_output")

# CDC BRFSS download URLs
# Format shifted slightly across years; this mapping handles known variations
def get_brfss_url(year: int) -> str:
    """
    Construct the CDC download URL for a given BRFSS survey year.
    
    The naming convention for the SAS transport (XPT) files has varied:
      - 2010: CDBRFS10XPT.zip
      - 2011+: LLCP{YEAR}XPT.zip (the "calculated variables" version)
    
    We use the LLCP (calculated variables) version when available because it
    includes pre-computed variables like _MENTHLTH, _RFMNTL, etc.
    """
    base = "https://www.cdc.gov/brfss/annual_data"
    if year == 2010:
        return f"{base}/2010/files/CDBRFS10XPT.zip"
    else:
        yr_short = str(year)
        return f"{base}/{year}/files/LLCP{yr_short}XPT.zip"


# =============================================================================
# VARIABLE HARMONIZATION MAP
# =============================================================================
# BRFSS variable names shift across survey years. This map defines the
# canonical variable name → list of possible source names (tried in order).
# The first match found in the dataset is used.
#
# Sources:
#   - BRFSS codebooks: https://www.cdc.gov/brfss/annual_data/annual_data.htm
#   - Variable layouts by year (PDF/HTML on CDC site)

VARIABLE_MAP = {
    # --- Identifiers ---
    "state_fips": ["_STATE"],
    
    # --- Mental health outcomes ---
    # MENTHLTH: "Now thinking about your mental health, which includes stress,
    #            depression, and problems with emotions, for how many days during
    #            the past 30 days was your mental health not good?"
    # Coded: 1-30 = days, 88 = none, 77 = don't know, 99 = refused
    "mental_health_days_raw": ["MENTHLTH"],
    
    # ADDEPEV3/ADDEPEV2: "Has a doctor, nurse, or other health professional
    #                     ever told you that you had a depressive disorder?"
    # Coded: 1 = Yes, 2 = No, 7 = DK, 9 = Refused
    # Name changed from ADDEPEV2 to ADDEPEV3 around 2020
    "depression_ever_raw": ["ADDEPEV3", "ADDEPEV2"],
    
    # _MENT14D: Calculated - mental health not good 14+ days (frequent distress)
    # 1 = 0 days not good, 2 = 1-13 days, 3 = 14+ days, 9 = DK/missing
    "freq_mental_distress_calc": ["_MENT14D"],
    
    # --- Insurance / coverage ---
    # HLTHPLN1: "Do you have any kind of health care coverage?"
    # 1 = Yes, 2 = No, 7 = DK, 9 = Refused
    "has_insurance_raw": ["HLTHPLN1", "HLTHPLN"],
    
    # PRIMINSR: Primary source of health insurance (available 2014+)
    # Codes vary but Medicaid is typically 4 or 5
    "primary_insurance_raw": ["PRIMINSR"],
    
    # MEDCOST1/MEDCOST: "Was there a time in the past 12 months when you needed 
    #                    to see a doctor but could not because of cost?"
    # 1 = Yes, 2 = No
    "cost_barrier_raw": ["MEDCOST1", "MEDCOST"],
    
    # _HCVU651: Calculated - respondents aged 18-64 who have any health coverage
    # 1 = Have coverage, 2 = Do not, 9 = DK/missing
    "has_coverage_18_64_calc": ["_HCVU651"],
    
    # --- Demographics (for covariate adjustment) ---
    # Age
    "age_raw": ["_AGE80", "_AGE_G", "_AGEG5YR", "AGE"],
    "age_group_calc": ["_AGE_G", "_AGEG5YR"],
    
    # Sex / gender
    # Note: BRFSS changed from SEX to SEXVAR in 2022, added BIRTHSEX
    "sex_raw": ["SEXVAR", "SEX1", "SEX", "_SEX"],
    
    # Race / ethnicity
    # _IMPRACE: Imputed race/ethnicity (available 2013+, preferred for completeness)
    # _RACEGR3: Calculated race groups
    # _RACE: Five-level race
    "race_ethnicity_calc": ["_IMPRACE", "_RACEGR3", "_RACE"],
    
    # Income
    # INCOME3 (2021+) replaced INCOME2 (prior years): household income brackets
    "income_raw": ["INCOME3", "INCOME2"],
    # _INCOMG1 (2021+) replaced _INCOMG: calculated income categories  
    "income_group_calc": ["_INCOMG1", "_INCOMG"],
    
    # Education
    "education_raw": ["EDUCA"],
    "education_group_calc": ["_EDUCAG"],
    
    # Employment
    "employment_raw": ["EMPLOY1", "EMPLOY"],
    
    # Marital status
    "marital_raw": ["MARITAL"],
    
    # --- Geography ---
    "county_code": ["_COUNFIP", "COUNFIP", "CNTY"],
    "metro_status_calc": ["_METSTAT", "MSCODE"],
    "urban_rural_calc": ["_URBSTAT"],
    
    # --- Survey design ---
    # Primary sampling unit and stratum for variance estimation
    "psu": ["_PSU"],
    "stratum": ["_STSTR"],
    
    # Final weight: _LLCPWT (2011+), _FINALWT (2010 and earlier)
    "weight": ["_LLCPWT", "_FINALWT"],
}


# =============================================================================
# MEDICAID EXPANSION STATUS BY STATE
# =============================================================================
# Source: KFF Status of State Medicaid Expansion Decisions
# https://www.kff.org/medicaid/issue-brief/status-of-state-medicaid-expansion-decisions/
#
# Format: state FIPS → year of expansion (January implementation)
# States not listed here had NOT expanded as of end-2023.

MEDICAID_EXPANSION = {
    # 2014 original expansion states (+ DC)
    1: None,     # Alabama - not expanded
    2: 2015,     # Alaska
    4: 2014,     # Arizona
    5: 2014,     # Arkansas
    6: 2014,     # California
    8: 2014,     # Colorado
    9: 2014,     # Connecticut
    10: 2014,    # Delaware
    11: 2014,    # DC
    12: None,    # Florida - not expanded
    13: None,    # Georgia - not expanded
    15: 2014,    # Hawaii
    16: 2020,    # Idaho
    17: 2014,    # Illinois
    18: 2015,    # Indiana
    19: 2014,    # Iowa
    20: None,    # Kansas - not expanded (as of 2023)
    21: 2014,    # Kentucky
    22: 2016,    # Louisiana
    23: 2019,    # Maine (approved 2017, implemented 2019)
    24: 2014,    # Maryland
    25: 2014,    # Massachusetts
    26: 2014,    # Michigan
    27: 2014,    # Minnesota
    28: None,    # Mississippi - not expanded
    29: 2021,    # Missouri
    30: 2016,    # Montana
    31: 2020,    # Nebraska
    32: 2014,    # Nevada
    33: 2014,    # New Hampshire (technically mid-2014)
    34: 2014,    # New Jersey
    35: 2014,    # New Mexico
    36: 2014,    # New York
    37: 2024,    # North Carolina (Dec 2023 / early 2024 — treat as post-period)
    38: 2014,    # North Dakota
    39: 2014,    # Ohio
    40: 2024,    # Oklahoma (2024 implementation)
    41: 2014,    # Oregon
    42: 2015,    # Pennsylvania
    44: 2014,    # Rhode Island
    45: None,    # South Carolina - not expanded
    46: 2024,    # South Dakota (2024)
    47: None,    # Tennessee - not expanded
    48: None,    # Texas - not expanded
    49: 2020,    # Utah
    50: 2014,    # Vermont
    51: 2019,    # Virginia
    53: 2014,    # Washington
    54: 2014,    # West Virginia
    55: None,    # Wisconsin - partial, not counted as full expansion
    56: None,    # Wyoming - not expanded
    66: None,    # Guam
    72: None,    # Puerto Rico
    78: None,    # Virgin Islands
}

# State FIPS to name mapping (for readable output)
STATE_NAMES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
    53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
    66: "Guam", 72: "Puerto Rico", 78: "Virgin Islands",
}

# Postal abbreviations / common aliases for user-friendly CLI state filters
STATE_ABBREV_TO_FIPS = {
    "AL": 1, "AK": 2, "AZ": 4, "AR": 5, "CA": 6, "CO": 8, "CT": 9,
    "DE": 10, "DC": 11, "FL": 12, "GA": 13, "HI": 15, "ID": 16,
    "IL": 17, "IN": 18, "IA": 19, "KS": 20, "KY": 21, "LA": 22,
    "ME": 23, "MD": 24, "MA": 25, "MI": 26, "MN": 27, "MS": 28,
    "MO": 29, "MT": 30, "NE": 31, "NV": 32, "NH": 33, "NJ": 34,
    "NM": 35, "NY": 36, "NC": 37, "ND": 38, "OH": 39, "OK": 40,
    "OR": 41, "PA": 42, "RI": 44, "SC": 45, "SD": 46, "TN": 47,
    "TX": 48, "UT": 49, "VT": 50, "VA": 51, "WA": 53, "WV": 54,
    "WI": 55, "WY": 56, "GU": 66, "PR": 72, "VI": 78,
}

STATE_NORMALIZED_TO_FIPS = {}
for fips, name in STATE_NAMES.items():
    STATE_NORMALIZED_TO_FIPS[name.strip().lower()] = fips

STATE_NORMALIZED_TO_FIPS.update({
    abbr.lower(): fips for abbr, fips in STATE_ABBREV_TO_FIPS.items()
})

STATE_NORMALIZED_TO_FIPS.update({
    "district of columbia": 11,
    "district of columbia (dc)": 11,
    "washington dc": 11,
    "washington, dc": 11,
    "d.c.": 11,
    "puerto rico": 72,
    "virgin islands": 78,
    "u.s. virgin islands": 78,
    "us virgin islands": 78,
})


def normalize_state_token(token: str) -> str:
    """Normalize CLI state tokens like 'New_York', 'new-york', ' DC '."""
    return token.strip().replace("_", " ").replace("-", " ").lower()


def parse_state_filters(states: Optional[Iterable[str]]) -> Optional[Set[int]]:
    """
    Parse a user-supplied list of states / territories into a set of BRFSS FIPS codes.

    Accepts full names (e.g. 'Ohio'), postal abbreviations ('OH'), and numeric FIPS
    strings ('39'). Returns None if no filter was supplied.
    """
    if not states:
        return None

    selected_fips: Set[int] = set()
    unknown = []

    for raw_state in states:
        token = normalize_state_token(raw_state)
        compact = raw_state.strip()

        if compact.isdigit():
            fips = int(compact)
            if fips in STATE_NAMES:
                selected_fips.add(fips)
            else:
                unknown.append(raw_state)
            continue

        fips = STATE_NORMALIZED_TO_FIPS.get(token)
        if fips is None and compact.upper() in STATE_ABBREV_TO_FIPS:
            fips = STATE_ABBREV_TO_FIPS[compact.upper()]

        if fips is None:
            unknown.append(raw_state)
        else:
            selected_fips.add(fips)

    if unknown:
        valid_examples = "OH Ohio 39 PR Puerto_Rico Guam VI"
        raise ValueError(
            "Unrecognized state / territory: "
            + ", ".join(unknown)
            + f". Examples of valid inputs: {valid_examples}"
        )

    return selected_fips


def filter_to_states(df: pd.DataFrame, selected_fips: Optional[Set[int]]) -> pd.DataFrame:
    """Filter a cleaned BRFSS DataFrame to a selected set of state FIPS codes."""
    if selected_fips is None or "state_fips" not in df.columns:
        return df
    return df[df["state_fips"].isin(selected_fips)].copy()



# =============================================================================
# DOWNLOAD AND READ
# =============================================================================

def find_existing_xpt(year: int, data_dir: Path) -> Optional[Path]:
    """Best-effort lookup for an already-downloaded BRFSS transport file."""
    patterns = [
        f"*{year}*.XPT", f"*{year}*.xpt",
        f"*{str(year)[-2:]}*.XPT", f"*{str(year)[-2:]}*.xpt",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(data_dir.glob(pattern))

    canonical_prefixes = [
        f"LLCP{year}",
        f"CDBRFS{str(year)[-2:]}",
        f"LLCP{str(year)[-2:]}",
    ]
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        name_upper = path.name.upper()
        if name_upper.endswith(".XPT") and any(name_upper.startswith(prefix.upper()) for prefix in canonical_prefixes):
            candidates.append(path)

    if not candidates:
        return None

    candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
    return candidates[0]



def download_brfss_year(year: int, data_dir: Path) -> Optional[Path]:
    """
    Download the BRFSS XPT file for a given year.
    Returns the path to the extracted .XPT file, or None on failure.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = find_existing_xpt(year, data_dir)
    if existing is not None:
        print(f"  [✓] {year}: already downloaded → {existing.name}")
        return existing

    url = get_brfss_url(year)
    print(f"  [↓] {year}: downloading from {url} ...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        # BRFSS files usually come as a zip containing a single SAS transport file,
        # but CDC packaging is not fully consistent about member filenames.
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if not members:
                print(f"  [!] {year}: zip was empty")
                return None

            # Prefer explicit .xpt members; otherwise fall back to the largest file.
            xpt_members = [m for m in members if m.filename.upper().endswith(".XPT")]
            chosen = max(xpt_members or members, key=lambda m: m.file_size)

            out_name = "CDBRFS10.XPT" if year == 2010 else f"LLCP{year}.XPT"
            out_path = data_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(chosen) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

            print(f"  [✓] {year}: extracted {chosen.filename} → {out_name} "
                  f"({out_path.stat().st_size / 1e6:.0f} MB)")
            return out_path

    except requests.exceptions.RequestException as e:
        print(f"  [✗] {year}: download failed — {e}")
        return None
    except zipfile.BadZipFile:
        # Some years may serve the transport file directly instead of a zip.
        out_name = "CDBRFS10.XPT" if year == 2010 else f"LLCP{year}.XPT"
        out_path = data_dir / out_name
        out_path.write_bytes(resp.content)
        print(f"  [✓] {year}: saved raw XPT as {out_name} ({out_path.stat().st_size / 1e6:.0f} MB)")
        return out_path


def read_brfss_xpt(xpt_path: Path, year: int) -> pd.DataFrame:
    """
    Read a BRFSS .XPT (SAS transport) file and extract + harmonize variables.

    Returns a DataFrame with canonical column names from VARIABLE_MAP.
    Only columns that exist in the source file are returned.
    """
    print(f"  [→] Reading {xpt_path.name} ...")

    # Older files can contain non-UTF-8 labels/strings; latin-1 is a safe fallback.
    last_err = None
    for enc in ("utf-8", "latin1"):
        try:
            raw = pd.read_sas(xpt_path, format="xport", encoding=enc)
            if enc != "utf-8":
                print(f"      using fallback encoding: {enc}")
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise last_err

    # Normalize column names to uppercase (some years differ in casing)
    raw.columns = raw.columns.str.upper()
    available_cols = set(raw.columns)

    print(f"      {len(raw):,} records, {len(raw.columns)} variables")

    # Extract variables using the harmonization map
    extracted = pd.DataFrame(index=raw.index)
    extracted["year"] = year

    matched_vars = []
    for canonical_name, candidates in VARIABLE_MAP.items():
        for candidate in candidates:
            if candidate.upper() in available_cols:
                extracted[canonical_name] = raw[candidate.upper()].values
                matched_vars.append(f"    {canonical_name} ← {candidate}")
                break

    print(f"      Matched {len(matched_vars)} / {len(VARIABLE_MAP)} variables")

    return extracted


# =============================================================================
# CLEANING AND RECODING
# =============================================================================

def clean_individual_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recode raw BRFSS responses into analysis-ready variables.
    
    BRFSS uses various sentinel values:
        77 = Don't know / Not sure
        88 = None (for count variables like MENTHLTH)
        99 = Refused
        BLANK = Missing
    
    We recode these to proper numeric values or NaN.
    """
    out = df.copy()
    year = out["year"].iloc[0]
    
    # --- Mental health days (0-30 scale) ---
    if "mental_health_days_raw" in out.columns:
        mh = out["mental_health_days_raw"].copy()
        mh = mh.replace({88: 0})                    # 88 = "None" → 0 days
        mh[(mh == 77) | (mh == 99)] = np.nan         # DK / Refused
        mh[(mh < 0) | (mh > 30)] = np.nan            # out of range
        out["mh_days_not_good"] = mh
        
        # Binary: frequent mental distress (14+ days)
        out["freq_mental_distress"] = (mh >= 14).astype(float)
        out.loc[mh.isna(), "freq_mental_distress"] = np.nan
        
        # Binary: any poor mental health days
        out["any_mh_days"] = (mh > 0).astype(float)
        out.loc[mh.isna(), "any_mh_days"] = np.nan
    
    # --- Depression diagnosis (binary) ---
    if "depression_ever_raw" in out.columns:
        dep = out["depression_ever_raw"].copy()
        out["depression_dx"] = np.where(dep == 1, 1.0,
                               np.where(dep == 2, 0.0, np.nan))
    
    # --- Insurance status (binary) ---
    if "has_insurance_raw" in out.columns:
        ins = out["has_insurance_raw"].copy()
        out["has_insurance"] = np.where(ins == 1, 1.0,
                               np.where(ins == 2, 0.0, np.nan))
    
    # --- Cost barrier to care (binary) ---
    if "cost_barrier_raw" in out.columns:
        cost = out["cost_barrier_raw"].copy()
        out["cost_barrier"] = np.where(cost == 1, 1.0,
                              np.where(cost == 2, 0.0, np.nan))
    
    # --- Medicaid as primary insurance (binary, 2014+) ---
    if "primary_insurance_raw" in out.columns:
        pri = out["primary_insurance_raw"].copy()
        # Code 4 or 5 = Medicaid depending on year; 
        # check codebook, but typically:
        #   2014-2019: 4 = Medicaid
        #   2020+: may shift — verify against year-specific codebook
        out["has_medicaid"] = np.where(pri.isin([4, 5]), 1.0,
                              np.where(pri.isin([1, 2, 3, 6, 7, 8, 9, 10]), 0.0,
                              np.nan))
    
    # --- Sex (binary: 1=male, 0=female) ---
    if "sex_raw" in out.columns:
        sex = out["sex_raw"].copy()
        out["male"] = np.where(sex == 1, 1.0,
                     np.where(sex == 2, 0.0, np.nan))
    
    # --- Race/ethnicity (categorical) ---
    if "race_ethnicity_calc" in out.columns:
        race = out["race_ethnicity_calc"].copy()
        # _IMPRACE coding (2013+):
        # 1=White NH, 2=Black NH, 3=Asian NH, 4=AI/AN NH, 5=Hispanic, 6=Other
        # _RACEGR3 coding (earlier):
        # 1=White NH, 2=Black NH, 3=Other NH, 4=Multiracial NH, 5=Hispanic
        # We harmonize to: White_NH, Black_NH, Hispanic, Other
        race_map = {1: "White_NH", 2: "Black_NH", 3: "Other_NH",
                    4: "Other_NH", 5: "Hispanic", 6: "Other_NH"}
        out["race_eth"] = race.map(race_map)
    
    # --- Income group ---
    if "income_group_calc" in out.columns:
        inc = out["income_group_calc"].copy()
        # _INCOMG / _INCOMG1: 1 = <$15K, 2 = $15-25K, 3 = $25-35K,
        #                      4 = $35-50K, 5 = $50K+, 9 = DK/Missing
        inc[inc == 9] = np.nan
        out["income_group"] = inc
    elif "income_raw" in out.columns:
        # INCOME2: 1-8 scale from <$10K to $75K+, 77=DK, 99=Refused
        inc = out["income_raw"].copy()
        inc[(inc == 77) | (inc == 99)] = np.nan
        # Collapse to 5 groups matching _INCOMG
        out["income_group"] = pd.cut(inc, bins=[0, 2, 4, 5, 6, 8],
                                     labels=[1, 2, 3, 4, 5]).astype(float)
    
    # --- Education group ---
    if "education_group_calc" in out.columns:
        edu = out["education_group_calc"].copy()
        # 1 = < HS, 2 = HS grad, 3 = some college, 4 = college grad, 9 = DK
        edu[edu == 9] = np.nan
        out["education_group"] = edu
    
    # --- Employment ---
    if "employment_raw" in out.columns:
        emp = out["employment_raw"].copy()
        # 1=employed, 2=self-employed, 3=unemployed 1yr+, 4=unemployed <1yr,
        # 5=homemaker, 6=student, 7=retired, 8=unable to work, 9=refused
        out["employed"] = np.where(emp.isin([1, 2]), 1.0,
                          np.where(emp.isin([3, 4, 5, 6, 7, 8]), 0.0, np.nan))
        out["unemployed"] = np.where(emp.isin([3, 4]), 1.0,
                            np.where(emp.isin([1, 2, 5, 6, 7, 8]), 0.0, np.nan))
        out["unable_to_work"] = np.where(emp == 8, 1.0,
                                np.where(emp.isin([1,2,3,4,5,6,7]), 0.0, np.nan))
    
    # --- Age (continuous where available) ---
    if "age_raw" in out.columns:
        age = out["age_raw"].copy()
        # _AGE80: top-coded at 80; values 7/9 = DK/refused in some years
        age[(age < 18) | (age > 99)] = np.nan
        out["age"] = age
    
    # --- State FIPS (ensure integer) ---
    if "state_fips" in out.columns:
        out["state_fips"] = out["state_fips"].astype(float).astype("Int64")
    
    # --- Medicaid expansion status ---
    if "state_fips" in out.columns:
        out["expansion_year"] = out["state_fips"].map(MEDICAID_EXPANSION)
        out["expanded"] = (
            out["expansion_year"].notna() & 
            (out["year"] >= out["expansion_year"])
        ).astype(int)
        # Years since expansion (for event study)
        out["years_since_expansion"] = np.where(
            out["expansion_year"].notna(),
            out["year"] - out["expansion_year"],
            np.nan
        )
    
    # State name
    if "state_fips" in out.columns:
        out["state_name"] = out["state_fips"].map(STATE_NAMES)
    
    return out


# =============================================================================
# AGGREGATION: STATE × YEAR PANEL
# =============================================================================

def aggregate_state_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual-level data to a state × year panel.
    
    Computes weighted means (using survey weights) for all outcome and
    covariate variables. This is the dataset you'd use for state-level DiD.
    
    For individual-level DiD (e.g., DR-DiD with Callaway-Sant'Anna),
    use the individual panel instead.
    """
    
    # Define which columns to aggregate
    outcome_vars = [
        "mh_days_not_good", "freq_mental_distress", "any_mh_days",
        "depression_dx", "has_insurance", "cost_barrier", "has_medicaid",
    ]
    covariate_vars = [
        "male", "age", "employed", "unemployed", "unable_to_work",
    ]
    all_agg_vars = [v for v in outcome_vars + covariate_vars if v in df.columns]
    
    has_weights = "weight" in df.columns and df["weight"].notna().any()
    
    def safe_weighted_average(values, weights):
        """Return a weighted average, or NaN if weights are unusable."""
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not mask.any():
            return np.nan
        values = values[mask]
        weights = weights[mask]
        weight_sum = weights.sum()
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            return np.nan
        return np.average(values, weights=weights)

    def weighted_mean(group, var, weight_col="weight"):
        """Compute weighted mean, handling NaN in both var and weights."""
        mask = group[var].notna()
        if has_weights:
            mask = mask & group[weight_col].notna() & (group[weight_col] > 0)
        subset = group[mask]
        if len(subset) == 0:
            return np.nan
        if has_weights:
            return safe_weighted_average(subset[var], subset[weight_col])
        else:
            return subset[var].mean()
    
    def weighted_se(group, var, weight_col="weight"):
        """Approximate standard error of weighted mean."""
        mask = group[var].notna()
        if has_weights:
            mask = mask & group[weight_col].notna() & (group[weight_col] > 0)
        subset = group[mask]
        n = len(subset)
        if n < 2:
            return np.nan
        if has_weights:
            weights = pd.to_numeric(subset[weight_col], errors="coerce").to_numpy(dtype=float)
            values = pd.to_numeric(subset[var], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            if valid.sum() < 2:
                return np.nan
            values = values[valid]
            weights = weights[valid]
            wm = safe_weighted_average(values, weights)
            if not np.isfinite(wm):
                return np.nan
            variance = safe_weighted_average((values - wm)**2, weights)
            if not np.isfinite(variance):
                return np.nan
            return np.sqrt(variance / len(values))
        else:
            return subset[var].std() / np.sqrt(n)
    
    print("\n  Aggregating to state × year panel ...")
    groups = df.groupby(["state_fips", "year"])
    
    records = []
    for (state, year), group in groups:
        rec = {
            "state_fips": state,
            "year": year,
            "state_name": STATE_NAMES.get(state, f"Unknown ({state})"),
            "n_respondents": len(group),
        }
        
        # Weighted means for all variables
        for var in all_agg_vars:
            if var in group.columns:
                rec[f"{var}_mean"] = weighted_mean(group, var)
                rec[f"{var}_se"] = weighted_se(group, var)
                rec[f"{var}_n"] = group[var].notna().sum()
        
        # Race/ethnicity shares (weighted)
        if "race_eth" in group.columns:
            for race_cat in ["White_NH", "Black_NH", "Hispanic", "Other_NH"]:
                indicator = (group["race_eth"] == race_cat).astype(float)
                indicator[group["race_eth"].isna()] = np.nan
                mask = indicator.notna()
                if has_weights:
                    mask = mask & group["weight"].notna() & (group["weight"] > 0)
                subset_ind = indicator[mask]
                if has_weights:
                    w = pd.to_numeric(group.loc[mask, "weight"], errors="coerce")
                    rec[f"share_{race_cat.lower()}"] = safe_weighted_average(subset_ind, w)
                else:
                    rec[f"share_{race_cat.lower()}"] = subset_ind.mean()
        
        # Income distribution shares
        if "income_group" in group.columns:
            for ig in [1, 2, 3, 4, 5]:
                indicator = (group["income_group"] == ig).astype(float)
                indicator[group["income_group"].isna()] = np.nan
                mask = indicator.notna()
                if has_weights:
                    mask = mask & group["weight"].notna() & (group["weight"] > 0)
                subset_ind = indicator[mask]
                if len(subset_ind) > 0:
                    if has_weights:
                        w = pd.to_numeric(group.loc[mask, "weight"], errors="coerce")
                        rec[f"share_income_{ig}"] = safe_weighted_average(subset_ind, w)
                    else:
                        rec[f"share_income_{ig}"] = subset_ind.mean()
        
        # Education distribution
        if "education_group" in group.columns:
            for eg in [1, 2, 3, 4]:
                indicator = (group["education_group"] == eg).astype(float)
                indicator[group["education_group"].isna()] = np.nan
                mask = indicator.notna()
                if has_weights:
                    mask = mask & group["weight"].notna() & (group["weight"] > 0)
                subset_ind = indicator[mask]
                if len(subset_ind) > 0:
                    if has_weights:
                        w = pd.to_numeric(group.loc[mask, "weight"], errors="coerce")
                        rec[f"share_educ_{eg}"] = safe_weighted_average(subset_ind, w)
                    else:
                        rec[f"share_educ_{eg}"] = subset_ind.mean()
        
        # Expansion status
        exp_yr = MEDICAID_EXPANSION.get(state)
        rec["expansion_year"] = exp_yr
        rec["expanded"] = 1 if (exp_yr is not None and year >= exp_yr) else 0
        rec["ever_expanded"] = 1 if exp_yr is not None else 0
        rec["years_since_expansion"] = (year - exp_yr) if exp_yr else np.nan
        
        records.append(rec)
    
    panel = pd.DataFrame(records)
    panel = panel.sort_values(["state_fips", "year"]).reset_index(drop=True)
    
    print(f"  Panel: {len(panel)} state-year observations "
          f"({panel['state_fips'].nunique()} states × "
          f"{panel['year'].nunique()} years)")
    
    return panel


# =============================================================================
# SUBGROUP PANELS (for heterogeneity analysis / causal forests)
# =============================================================================

def aggregate_subgroup_panels(df: pd.DataFrame) -> dict:
    """
    Create state × year panels stratified by key subgroups.
    Returns a dict of DataFrames keyed by subgroup name.
    
    These are useful for:
      - Pre-specified heterogeneity analysis (by race, income, age)
      - Validating causal forest heterogeneity findings
      - Equity analysis (comparing expansion effects across groups)
    """
    panels = {}
    
    # By income group (low vs. higher income — the policy-relevant margin)
    if "income_group" in df.columns:
        low_income = df[df["income_group"].isin([1, 2])]  # <$25K
        higher_income = df[df["income_group"].isin([3, 4, 5])]  # $25K+
        if len(low_income) > 0:
            panels["low_income"] = aggregate_state_year(low_income)
        if len(higher_income) > 0:
            panels["higher_income"] = aggregate_state_year(higher_income)
    
    # By race/ethnicity
    if "race_eth" in df.columns:
        for race in ["White_NH", "Black_NH", "Hispanic"]:
            subset = df[df["race_eth"] == race]
            if len(subset) > 1000:  # minimum sample for meaningful aggregation
                panels[f"race_{race.lower()}"] = aggregate_state_year(subset)
    
    return panels


# =============================================================================
# DIAGNOSTICS AND VALIDATION
# =============================================================================

def run_diagnostics(individual_df: pd.DataFrame, panel_df: pd.DataFrame):
    """
    Print diagnostic checks to verify data quality and flag issues.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTICS")
    print("=" * 70)
    
    # 1. Coverage by year
    print("\n1. Records per year:")
    yearly = individual_df.groupby("year").size()
    for yr, n in yearly.items():
        print(f"   {yr}: {n:>10,}")
    
    # 2. Variable availability by year
    key_vars = ["mh_days_not_good", "depression_dx", "has_insurance",
                "cost_barrier", "has_medicaid"]
    print(f"\n2. Variable availability (% non-missing) by year:")
    print(f"   {'Year':<6}", end="")
    for v in key_vars:
        print(f" {v[:15]:>16}", end="")
    print()
    for yr in sorted(individual_df["year"].unique()):
        yr_data = individual_df[individual_df["year"] == yr]
        print(f"   {int(yr):<6}", end="")
        for v in key_vars:
            if v in yr_data.columns:
                pct = yr_data[v].notna().mean() * 100
                print(f" {pct:>15.1f}%", end="")
            else:
                print(f" {'N/A':>16}", end="")
        print()
    
    # 3. Expansion status check
    print(f"\n3. Expansion status in panel:")
    if "expanded" in panel_df.columns:
        exp_check = panel_df.groupby("year")["expanded"].mean()
        for yr, pct in exp_check.items():
            print(f"   {int(yr)}: {pct:.1%} of states expanded")
    
    # 4. Key outcome trends (unweighted, for quick sanity check)
    print(f"\n4. Unweighted outcome means by expansion status and year:")
    if all(v in panel_df.columns for v in 
           ["expanded", "freq_mental_distress_mean", "year"]):
        trend = panel_df.groupby(["year", "expanded"])[
            "freq_mental_distress_mean"].mean().unstack()
        trend.columns = ["Non-expansion", "Expansion"]
        print(trend.to_string(float_format=lambda x: f"{x:.4f}"))
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BRFSS data extraction for Medicaid expansion & mental health")
    parser.add_argument("--years", nargs="+", type=int, default=YEARS,
                       help="Specific years to process (default: 2010-2023)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download, process existing .XPT files")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                       help="Directory for raw .XPT files")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                       help="Directory for output files")
    parser.add_argument("--no-individual", action="store_true",
                       help="Skip saving the large individual-level file")
    parser.add_argument("--subgroups", action="store_true",
                       help="Also produce subgroup-stratified panels")
    parser.add_argument("--states", nargs="+",
                       help="Optional state / territory filter. Accepts names, postal abbreviations, or FIPS codes (e.g. OH Ohio 39 PR)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        selected_fips = parse_state_filters(args.states)
    except ValueError as e:
        parser.error(str(e))
    
    years = sorted(args.years)
    print(f"BRFSS Extraction Pipeline")
    print(f"Years: {years[0]}–{years[-1]} ({len(years)} years)")
    print(f"Raw data: {data_dir}")
    print(f"Output: {output_dir}")
    if selected_fips is None:
        print("States: all states / territories in source files")
    else:
        selected_names = [STATE_NAMES[fips] for fips in sorted(selected_fips)]
        print(f"States: {selected_names}")
    print("=" * 70)
    
    # Step 1: Download
    if not args.skip_download:
        print("\n[1/4] DOWNLOADING BRFSS FILES")
        xpt_paths = {}
        for year in years:
            path = download_brfss_year(year, data_dir)
            if path:
                xpt_paths[year] = path
    else:
        print("\n[1/4] SCANNING EXISTING FILES")
        xpt_paths = {}
        for year in years:
            candidates = list(data_dir.glob(f"*{year}*.XPT")) + \
                        list(data_dir.glob(f"*{year}*.xpt"))
            if candidates:
                xpt_paths[year] = candidates[0]
                print(f"  [✓] {year}: {candidates[0].name}")
            else:
                print(f"  [✗] {year}: no file found")
    
    if not xpt_paths:
        print("\n[ERROR] No data files available. Exiting.")
        sys.exit(1)
    
    # Step 2: Read and harmonize
    print(f"\n[2/4] READING AND HARMONIZING ({len(xpt_paths)} files)")
    all_years = []
    for year in sorted(xpt_paths.keys()):
        raw = read_brfss_xpt(xpt_paths[year], year)
        cleaned = clean_individual_data(raw)
        cleaned = filter_to_states(cleaned, selected_fips)
        if len(cleaned) == 0:
            print(f"  [!] {year}: no records remain after state filter")
        all_years.append(cleaned)
        # Free memory — these DataFrames are large
        del raw
    
    # Combine all years
    print("\n  Concatenating all years ...")
    if not any(len(df) > 0 for df in all_years):
        print("\n[ERROR] No records matched the requested state filter. Exiting.")
        sys.exit(1)
    individual_df = pd.concat(all_years, ignore_index=True)
    del all_years
    print(f"  Total: {len(individual_df):,} individual records")
    
    # Step 3: Aggregate to state × year
    print(f"\n[3/4] AGGREGATING")
    state_year_panel = aggregate_state_year(individual_df)
    
    # Optional subgroup panels
    subgroup_panels = {}
    if args.subgroups:
        print("\n  Building subgroup panels ...")
        subgroup_panels = aggregate_subgroup_panels(individual_df)
    
    # Step 4: Save outputs
    print(f"\n[4/4] SAVING OUTPUTS")
    
    # State × year panel (CSV — small, human-readable, ready for R/Stata/Python)
    panel_path = output_dir / "brfss_state_year_panel.csv"
    state_year_panel.to_csv(panel_path, index=False, float_format="%.6f")
    print(f"  [✓] State-year panel → {panel_path}")
    print(f"      {len(state_year_panel)} rows × {len(state_year_panel.columns)} cols")
    
    # Individual-level panel (Parquet — efficient for large data)
    if not args.no_individual:
        # Select columns for the individual file (drop raw columns)
        keep_cols = [
            "year", "state_fips", "state_name",
            "mh_days_not_good", "freq_mental_distress", "any_mh_days",
            "depression_dx",
            "has_insurance", "cost_barrier", "has_medicaid",
            "male", "age", "race_eth", "income_group", "education_group",
            "employed", "unemployed", "unable_to_work",
            "expanded", "expansion_year", "years_since_expansion",
            "weight", "psu", "stratum",
        ]
        keep_cols = [c for c in keep_cols if c in individual_df.columns]
        
        indiv_path = output_dir / "brfss_individual_panel.parquet"
        individual_df[keep_cols].to_parquet(indiv_path, index=False,
                                            engine="pyarrow")
        print(f"  [✓] Individual panel → {indiv_path}")
        print(f"      {len(individual_df):,} rows × {len(keep_cols)} cols")
    
    # Subgroup panels
    for name, panel in subgroup_panels.items():
        sub_path = output_dir / f"brfss_state_year_{name}.csv"
        panel.to_csv(sub_path, index=False, float_format="%.6f")
        print(f"  [✓] Subgroup panel ({name}) → {sub_path}")
    
    # Diagnostics
    run_diagnostics(individual_df, state_year_panel)
    
    # Summary
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Individual records: {len(individual_df):,}")
    print(f"  State-year obs:     {len(state_year_panel):,}")
    print(f"  States:             {state_year_panel['state_fips'].nunique()}")
    print(f"  Years:              {state_year_panel['year'].min():.0f}–"
          f"{state_year_panel['year'].max():.0f}")
    print(f"\nOutputs in: {output_dir.resolve()}")
    if not args.no_individual:
        print(f"  • brfss_individual_panel.parquet  (for individual-level DiD)")
        print(f"  • brfss_state_year_panel.csv      (for state-level DiD)")
    print(f"\nNext steps:")
    print(f"  1. Verify diagnostics above (variable availability, sample sizes)")
    print(f"  2. Check MENTHLTH and ADDEPEV coverage for your target years")
    print(f"  3. For R analysis: read the CSV directly or convert parquet with arrow")
    print(f"  4. For Stata: use 'import delimited' on the CSV")
    

if __name__ == "__main__":
    main()
