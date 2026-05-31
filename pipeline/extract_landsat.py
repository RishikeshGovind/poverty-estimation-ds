"""
Task 4 — Extract Landsat 8/9 NDVI per SSA country (2014–2024).

Longer historical record than Sentinel-2 (which starts 2017).
Landsat 8 launched 2013, Landsat 9 in 2021 — merged as a single collection.

Source : LANDSAT/LC08/C02/T1_L2  (Landsat 8, 2013–present)
         LANDSAT/LC09/C02/T1_L2  (Landsat 9, 2021–present)
Bands  : SR_B5 (NIR), SR_B4 (Red)  — Collection 2 Level-2 Surface Reflectance
Scale  : SR value = raw_value × 0.0000275 − 0.2
NDVI   = (SR_B5 - SR_B4) / (SR_B5 + SR_B4)

Output : pipeline/outputs/landsat_ndvi.json

Schema:
  {
    "NGA": {"2014": 0.38, "2015": 0.40, ...},
    ...
  }

Usage:
    python pipeline/extract_landsat.py
    python pipeline/extract_landsat.py --years 2014 2015 2016
"""

import argparse
import json
import time

import ee

from config import (
    GEE_PROJECT, GEE_NAMES, ISO3_BY_GEE_NAME,
    LANDSAT_YEARS, SCALE_LANDSAT, LANDSAT_OUT,
)

_SCALE  = 0.0000275
_OFFSET = -0.2


def init_gee():
    key_file = __import__("os").environ.get("GEE_KEY_FILE", "")
    if key_file and __import__("os.path").path.exists(key_file):
        creds = ee.ServiceAccountCredentials(email=None, key_file=key_file)
        ee.Initialize(credentials=creds, project=GEE_PROJECT or None)
    else:
        ee.Initialize(project=GEE_PROJECT or None)


def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    """
    Use QA_PIXEL band to mask clouds, cloud shadows, and saturated pixels.
    Bits: 3 = cloud shadow, 4 = snow, 5 = cloud.
    """
    qa = image.select("QA_PIXEL")
    cloud_bit        = 1 << 5
    cloud_shadow_bit = 1 << 3
    clear = qa.bitwiseAnd(cloud_bit).eq(0).And(
            qa.bitwiseAnd(cloud_shadow_bit).eq(0))
    return image.updateMask(clear)


def apply_scale_factors(image: ee.Image) -> ee.Image:
    """Apply Collection 2 Level-2 reflectance scale + offset, then compute NDVI."""
    nir = image.select("SR_B5").multiply(_SCALE).add(_OFFSET).toFloat()
    red = image.select("SR_B4").multiply(_SCALE).add(_OFFSET).toFloat()
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return image.addBands(ndvi)


def get_country_fc():
    all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    return all_countries.filter(ee.Filter.inList("country_na", GEE_NAMES))


def get_landsat_collection(year: int) -> ee.ImageCollection:
    """
    Merge Landsat 8 and 9 C2 L2 collections for a given year.
    Landsat 9 data is only available from 2022 onward.
    """
    date_start = f"{year}-01-01"
    date_end   = f"{year}-12-31"

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(date_start, date_end)
            .map(mask_landsat_clouds)
            .map(apply_scale_factors)
            .select("NDVI"))

    if year >= 2022:
        l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
                .filterDate(date_start, date_end)
                .map(mask_landsat_clouds)
                .map(apply_scale_factors)
                .select("NDVI"))
        return l8.merge(l9)

    return l8


def extract_country_year(country: dict, year: int) -> float | None:
    """Process one country × one year to stay under GEE memory limits."""
    all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    feat = all_countries.filter(ee.Filter.eq("country_na", country["gee_name"])).first()
    geom = feat.geometry()

    collection = get_landsat_collection(year).filterBounds(geom)
    if collection.size().getInfo() == 0:
        return None

    composite = collection.median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=SCALE_LANDSAT,
        maxPixels=1e9,
        bestEffort=True,
    )
    val = stats.getInfo().get("NDVI")
    return round(float(val), 4) if val is not None else None


def main(years: list[int]):
    print("[landsat] Initialising GEE…")
    init_gee()

    from config import SSA_COUNTRIES

    output: dict[str, dict] = {}
    total = len(SSA_COUNTRIES) * len(years)
    done  = 0

    for country in SSA_COUNTRIES:
        iso3 = country["iso3"]
        for year in years:
            done += 1
            print(f"[landsat] ({done}/{total}) {iso3} {year}…", end=" ", flush=True)
            t0 = time.time()
            try:
                val = extract_country_year(country, year)
                elapsed = time.time() - t0
                if val is not None:
                    output.setdefault(iso3, {})[str(year)] = val
                    print(f"NDVI={val} ({elapsed:.1f}s)")
                else:
                    print(f"no data ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"error: {exc}")

    LANDSAT_OUT.parent.mkdir(exist_ok=True)
    with open(LANDSAT_OUT, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[landsat] Saved → {LANDSAT_OUT}")
    print(f"[landsat] Countries with data: {len(output)}")
    if output:
        sample = next(iter(output))
        print(f"[landsat] Sample ({sample}): {output[sample]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=LANDSAT_YEARS)
    args = parser.parse_args()
    main(args.years)
