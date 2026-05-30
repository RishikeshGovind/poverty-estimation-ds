"""
Shared config for the Phase 1 satellite extraction pipeline.
Country list matches useWorldBank.ts exactly (same ISO3 codes).
gee_name matches the USDOS/LSIB_SIMPLE/2017 `country_na` property.
"""

import os
from pathlib import Path

# ── Output paths ──────────────────────────────────────────────────────────────
PIPELINE_DIR = Path(__file__).parent
OUTPUT_DIR   = PIPELINE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

VIIRS_OUT    = OUTPUT_DIR / "viirs_ntl.json"
SENTINEL_OUT = OUTPUT_DIR / "sentinel2_ndvi_ndbi.json"
LANDSAT_OUT  = OUTPUT_DIR / "landsat_ndvi.json"
MERGED_OUT   = OUTPUT_DIR / "satellite_features.json"

# ── GEE project (set via env var or replace with your project ID) ─────────────
GEE_PROJECT = os.environ.get("GEE_PROJECT", "")   # e.g. "ee-yourname"

# ── SSA country list ──────────────────────────────────────────────────────────
# gee_name must match USDOS/LSIB_SIMPLE/2017 "country_na" property exactly
SSA_COUNTRIES = [
    {"iso3": "NGA", "name": "Nigeria",       "gee_name": "Nigeria"},
    {"iso3": "ETH", "name": "Ethiopia",      "gee_name": "Ethiopia"},
    {"iso3": "COD", "name": "DR Congo",      "gee_name": "Congo, Democratic Republic of the"},
    {"iso3": "KEN", "name": "Kenya",         "gee_name": "Kenya"},
    {"iso3": "TZA", "name": "Tanzania",      "gee_name": "United Republic of Tanzania"},
    {"iso3": "MOZ", "name": "Mozambique",    "gee_name": "Mozambique"},
    {"iso3": "GHA", "name": "Ghana",         "gee_name": "Ghana"},
    {"iso3": "UGA", "name": "Uganda",        "gee_name": "Uganda"},
    {"iso3": "CMR", "name": "Cameroon",      "gee_name": "Cameroon"},
    {"iso3": "AGO", "name": "Angola",        "gee_name": "Angola"},
    {"iso3": "ZMB", "name": "Zambia",        "gee_name": "Zambia"},
    {"iso3": "ZWE", "name": "Zimbabwe",      "gee_name": "Zimbabwe"},
    {"iso3": "MWI", "name": "Malawi",        "gee_name": "Malawi"},
    {"iso3": "SEN", "name": "Senegal",       "gee_name": "Senegal"},
    {"iso3": "MLI", "name": "Mali",          "gee_name": "Mali"},
    {"iso3": "BFA", "name": "Burkina Faso",  "gee_name": "Burkina Faso"},
    {"iso3": "RWA", "name": "Rwanda",        "gee_name": "Rwanda"},
    {"iso3": "NER", "name": "Niger",         "gee_name": "Niger"},
    {"iso3": "TCD", "name": "Chad",          "gee_name": "Chad"},
    {"iso3": "MDG", "name": "Madagascar",    "gee_name": "Madagascar"},
    {"iso3": "ZAF", "name": "South Africa",  "gee_name": "South Africa"},
    {"iso3": "SDN", "name": "Sudan",         "gee_name": "Sudan"},
    {"iso3": "SOM", "name": "Somalia",       "gee_name": "Somalia"},
    {"iso3": "GIN", "name": "Guinea",        "gee_name": "Guinea"},
    {"iso3": "BWA", "name": "Botswana",      "gee_name": "Botswana"},
    {"iso3": "NAM", "name": "Namibia",       "gee_name": "Namibia"},
    {"iso3": "SLE", "name": "Sierra Leone",  "gee_name": "Sierra Leone"},
    {"iso3": "TGO", "name": "Togo",          "gee_name": "Togo"},
    {"iso3": "BEN", "name": "Benin",         "gee_name": "Benin"},
    {"iso3": "HTI", "name": "Haiti",         "gee_name": "Haiti"},
]

GEE_NAMES = [c["gee_name"] for c in SSA_COUNTRIES]
ISO3_BY_GEE_NAME = {c["gee_name"]: c["iso3"] for c in SSA_COUNTRIES}

# ── Extraction settings ───────────────────────────────────────────────────────
VIIRS_YEARS    = list(range(2014, 2025))   # 2014–2024
SENTINEL_YEARS = list(range(2019, 2025))   # 2019–2024 (S2 archive starts 2017)
LANDSAT_YEARS  = list(range(2014, 2025))   # 2014–2024

SCALE_VIIRS    = 500    # metres — native VIIRS resolution
SCALE_SENTINEL = 1000   # metres — coarser for country-level stats (faster)
SCALE_LANDSAT  = 1000   # metres

MAX_CLOUD_PCT  = 30     # % — maximum cloud cover for Sentinel-2 scenes
