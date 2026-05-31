"""
Task 3 — Extract MODIS NDVI and Landsat NDBI per SSA country.

Switched from Sentinel-2 to MODIS for NDVI: MODIS MOD13A3 is a pre-built
monthly 1km composite — no per-scene cloud masking needed, no memory issues.
NDBI uses Landsat (has SWIR band; MODIS NDBI is unreliable at 500m).

Sources:
  NDVI : MODIS/061/MOD13A3  — monthly 1km NDVI composites, band "_1_km_monthly_NDVI"
  NDBI : LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2 — band SR_B6 (SWIR), SR_B5 (NIR)

Output : pipeline/outputs/sentinel2_ndvi_ndbi.json  (kept same filename for merge compat)
"""

import argparse
import json
import time

import ee

from config import GEE_PROJECT, SENTINEL_YEARS, SENTINEL_OUT, SSA_COUNTRIES

_LS_SCALE  = 0.0000275
_LS_OFFSET = -0.2


def init_gee():
    key_file = __import__("os").environ.get("GEE_KEY_FILE", "")
    if key_file and __import__("os.path").path.exists(key_file):
        creds = ee.ServiceAccountCredentials(email=None, key_file=key_file)
        ee.Initialize(credentials=creds, project=GEE_PROJECT or None)
    else:
        ee.Initialize(project=GEE_PROJECT or None)


def get_geom(gee_name: str) -> ee.Geometry:
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    return fc.filter(ee.Filter.eq("country_na", gee_name)).first().geometry()


def extract_modis_ndvi(geom: ee.Geometry, year: int) -> float | None:
    """MODIS MOD13A3 monthly 1km NDVI — pre-built composite, very memory-efficient."""
    col = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(geom)
        .select("NDVI")
    )
    composite = col.mean()   # annual mean of monthly composites
    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=1000,
        maxPixels=1e8,
        bestEffort=True,
    )
    val = stats.getInfo().get("NDVI")
    if val is None:
        return None
    return round(float(val) * 0.0001, 4)   # MOD13A3 scale factor


def mask_ls_clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA_PIXEL")
    clear = qa.bitwiseAnd(1 << 5).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
    return image.updateMask(clear)


def extract_landsat_ndbi(geom: ee.Geometry, year: int) -> float | None:
    """Landsat NDBI = (SWIR - NIR) / (SWIR + NIR). Process one country at coarse scale."""
    def apply_scale(img):
        nir  = img.select("SR_B5").multiply(_LS_SCALE).add(_LS_OFFSET)
        swir = img.select("SR_B6").multiply(_LS_SCALE).add(_LS_OFFSET)
        return img.addBands(swir.subtract(nir).divide(swir.add(nir)).rename("NDBI"))

    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(geom)
            .map(mask_ls_clouds).map(apply_scale).select("NDBI"))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .filterBounds(geom)
            .map(mask_ls_clouds).map(apply_scale).select("NDBI")) if year >= 2022 else ee.ImageCollection([])

    col = l8.merge(l9)
    composite = col.median()
    stats = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=5000,
        maxPixels=1e8,
        bestEffort=True,
    )
    val = stats.getInfo().get("NDBI")
    return round(float(val), 4) if val is not None else None


def main(years: list[int]):
    print("[ndvi/ndbi] Initialising GEE…")
    init_gee()

    output: dict[str, dict] = {}
    total = len(SSA_COUNTRIES) * len(years)
    done  = 0

    for country in SSA_COUNTRIES:
        iso3 = country["iso3"]
        geom = get_geom(country["gee_name"])
        for year in years:
            done += 1
            print(f"[ndvi/ndbi] ({done}/{total}) {iso3} {year}…", end=" ", flush=True)
            t0 = time.time()
            try:
                ndvi = extract_modis_ndvi(geom, year)
                ndbi = extract_landsat_ndbi(geom, year)
                elapsed = time.time() - t0
                entry = output.setdefault(iso3, {"ndvi": {}, "ndbi": {}})
                if ndvi is not None:
                    entry["ndvi"][str(year)] = ndvi
                if ndbi is not None:
                    entry["ndbi"][str(year)] = ndbi
                print(f"NDVI={ndvi} NDBI={ndbi} ({elapsed:.1f}s)")
            except Exception as exc:
                print(f"error: {exc}")

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
