"""
Task 2 — Extract VIIRS nighttime lights (NTL) mean radiance per SSA country.

Source : NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG  (Google Earth Engine)
Band   : avg_rad  (nW/cm²/sr, gap-filled monthly composites)
Output : pipeline/outputs/viirs_ntl.json

Schema:
  {
    "NGA": {"2014": 1.23, "2015": 1.45, ...},
    "KEN": {"2014": 0.87, ...},
    ...
  }

Usage:
    python pipeline/extract_viirs.py
    python pipeline/extract_viirs.py --years 2020 2021 2022   # subset of years
"""

import argparse
import json
import sys
import time

import ee

from config import (
    GEE_PROJECT, GEE_NAMES, ISO3_BY_GEE_NAME,
    VIIRS_YEARS, SCALE_VIIRS, VIIRS_OUT,
)


def init_gee():
    key_file = __import__("os").environ.get("GEE_KEY_FILE", "")
    if key_file and __import__("os.path").path.exists(key_file):
        creds = ee.ServiceAccountCredentials(email=None, key_file=key_file)
        ee.Initialize(credentials=creds, project=GEE_PROJECT or None)
    else:
        ee.Initialize(project=GEE_PROJECT or None)


def get_country_fc():
    """Load LSIB boundaries filtered to our SSA country list."""
    all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    return all_countries.filter(ee.Filter.inList("country_na", GEE_NAMES))


def extract_annual_ntl(year: int, country_fc: ee.FeatureCollection) -> dict:
    """
    Compute mean annual NTL radiance for each country.
    Returns {iso3: mean_radiance} for the given year.
    """
    collection = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("avg_rad")
    )

    if collection.size().getInfo() == 0:
        print(f"  [warn] No VIIRS data for {year} — skipping")
        return {}

    annual_mean = collection.mean()

    stats = annual_mean.reduceRegions(
        collection=country_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_VIIRS,
    )

    features = stats.getInfo().get("features", [])
    result = {}
    for feat in features:
        props   = feat.get("properties", {})
        gee_name = props.get("country_na", "")
        iso3     = ISO3_BY_GEE_NAME.get(gee_name)
        mean_val = props.get("mean")
        if iso3 and mean_val is not None:
            result[iso3] = round(float(mean_val), 4)

    return result


def main(years: list[int]):
    print("[viirs] Initialising GEE…")
    init_gee()

    print("[viirs] Loading country boundaries…")
    country_fc = get_country_fc()
    n = country_fc.size().getInfo()
    print(f"[viirs] Found {n} country features")

    output = {}   # {iso3: {year_str: mean_ntl}}

    for year in years:
        print(f"[viirs] Extracting {year}…", end=" ", flush=True)
        t0 = time.time()
        try:
            year_data = extract_annual_ntl(year, country_fc)
            elapsed   = time.time() - t0
            print(f"got {len(year_data)} countries in {elapsed:.1f}s")
            for iso3, val in year_data.items():
                output.setdefault(iso3, {})[str(year)] = val
        except Exception as exc:
            print(f"\n  [error] {year}: {exc}")

    VIIRS_OUT.parent.mkdir(exist_ok=True)
    with open(VIIRS_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[viirs] Saved → {VIIRS_OUT}")
    print(f"[viirs] Countries with data: {len(output)}")
    if output:
        sample_iso3 = next(iter(output))
        print(f"[viirs] Sample ({sample_iso3}): {output[sample_iso3]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=VIIRS_YEARS,
                        help="Years to extract (default: 2014–2024)")
    args = parser.parse_args()
    main(args.years)
