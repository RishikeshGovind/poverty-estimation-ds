"""
Task 3 — Extract Sentinel-2 NDVI and NDBI per SSA country.

Source : COPERNICUS/S2_SR_HARMONIZED  (Google Earth Engine)
Bands  : B4 (Red), B8 (NIR), B11 (SWIR)
Indices:
  NDVI = (B8 - B4) / (B8 + B4)   → vegetation health [−1, 1]
  NDBI = (B11 - B8) / (B11 + B8) → built-up / urban index [−1, 1]
Output : pipeline/outputs/sentinel2_ndvi_ndbi.json

Schema:
  {
    "NGA": {
      "ndvi": {"2019": 0.42, "2020": 0.40, ...},
      "ndbi": {"2019": −0.08, "2020": −0.09, ...}
    },
    ...
  }

Usage:
    python pipeline/extract_sentinel2.py
    python pipeline/extract_sentinel2.py --years 2022 2023
"""

import argparse
import json
import time

import ee

from config import (
    GEE_PROJECT, GEE_NAMES, ISO3_BY_GEE_NAME,
    SENTINEL_YEARS, SCALE_SENTINEL, MAX_CLOUD_PCT, SENTINEL_OUT,
)


def init_gee():
    key_file = __import__("os").environ.get("GEE_KEY_FILE", "")
    if key_file and __import__("os.path").path.exists(key_file):
        creds = ee.ServiceAccountCredentials(email=None, key_file=key_file)
        ee.Initialize(credentials=creds, project=GEE_PROJECT or None)
    else:
        ee.Initialize(project=GEE_PROJECT or None)


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Use Sentinel-2 Scene Classification Layer (SCL) to mask clouds and shadows."""
    scl = image.select("SCL")
    # SCL values: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=thin cirrus
    cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(cloud_mask)


def add_indices(image: ee.Image) -> ee.Image:
    """Add NDVI and NDBI bands. Sentinel-2 reflectance is scaled by 10000."""
    nir  = image.select("B8").toFloat()
    red  = image.select("B4").toFloat()
    swir = image.select("B11").toFloat()

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndbi = swir.subtract(nir).divide(swir.add(nir)).rename("NDBI")

    return image.addBands([ndvi, ndbi])


def get_country_fc():
    all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    return all_countries.filter(ee.Filter.inList("country_na", GEE_NAMES))


def extract_annual_indices(year: int, country_fc: ee.FeatureCollection) -> dict:
    """
    Compute median annual NDVI and NDBI per country, cloud-filtered.
    Returns {iso3: {"ndvi": float, "ndbi": float}}.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
        .map(add_indices)
        .select(["NDVI", "NDBI"])
    )

    if collection.size().getInfo() == 0:
        print(f"  [warn] No S2 data for {year} — skipping")
        return {}

    # Median composite reduces cloud/shadow residuals
    composite = collection.median()

    stats = composite.reduceRegions(
        collection=country_fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_SENTINEL,
    )

    features  = stats.getInfo().get("features", [])
    result    = {}
    for feat in features:
        props    = feat.get("properties", {})
        gee_name = props.get("country_na", "")
        iso3     = ISO3_BY_GEE_NAME.get(gee_name)
        ndvi_val = props.get("NDVI")
        ndbi_val = props.get("NDBI")
        if iso3 and ndvi_val is not None and ndbi_val is not None:
            result[iso3] = {
                "ndvi": round(float(ndvi_val), 4),
                "ndbi": round(float(ndbi_val), 4),
            }

    return result


def main(years: list[int]):
    print("[sentinel2] Initialising GEE…")
    init_gee()

    print("[sentinel2] Loading country boundaries…")
    country_fc = get_country_fc()

    # Output structure: {iso3: {ndvi: {year: val}, ndbi: {year: val}}}
    output: dict[str, dict] = {}

    for year in years:
        print(f"[sentinel2] Extracting {year}…", end=" ", flush=True)
        t0 = time.time()
        try:
            year_data = extract_annual_indices(year, country_fc)
            elapsed   = time.time() - t0
            print(f"got {len(year_data)} countries in {elapsed:.1f}s")
            for iso3, vals in year_data.items():
                entry = output.setdefault(iso3, {"ndvi": {}, "ndbi": {}})
                entry["ndvi"][str(year)] = vals["ndvi"]
                entry["ndbi"][str(year)] = vals["ndbi"]
        except Exception as exc:
            print(f"\n  [error] {year}: {exc}")

    SENTINEL_OUT.parent.mkdir(exist_ok=True)
    with open(SENTINEL_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[sentinel2] Saved → {SENTINEL_OUT}")
    print(f"[sentinel2] Countries with data: {len(output)}")
    if output:
        sample = next(iter(output))
        print(f"[sentinel2] Sample ({sample}): {output[sample]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=SENTINEL_YEARS)
    args = parser.parse_args()
    main(args.years)
