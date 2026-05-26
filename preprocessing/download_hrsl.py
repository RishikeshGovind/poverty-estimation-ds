"""
Phase 2 — Meta High-Resolution Settlement Layer (HRSL) downloader.

HRSL provides 30m population density for 160+ countries, built by Meta/CIESIN
from satellite imagery + census data.  Population density is a strong proxy
for both urbanisation and poverty access to services.

Free on AWS S3 (no credentials needed) and Humanitarian Data Exchange (HDX).

AWS path: s3://dataforgood-fb-data/hrsl-data/

Run:
    python -m preprocessing.download_hrsl
"""

import os
import subprocess
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# HDX direct download URLs for Africa (no sign-in needed)
# Full continent file — 28 COG tiffs split by region
_HDX_AFRICA_URL = (
    "https://data.humdata.org/dataset/"
    "highresolutionpopulationdensitymaps-eth"  # Ethiopia example
    "/resource/population_eth_2018-10-01.tif"
)

# AWS S3 path (no credentials) — country-level files
# Pattern: s3://dataforgood-fb-data/hrsl-data/hrsl_<country_iso3>.zip
_S3_BASE = "s3://dataforgood-fb-data/hrsl-data/"

# ISO3 codes for the DHS countries we target
_COUNTRY_ISO3 = {
    "Kenya":    "ken",
    "Nigeria":  "nga",
    "Ethiopia": "eth",
    "Ghana":    "gha",
    "Tanzania": "tza",
    "Rwanda":   "rwa",
    "Malawi":   "mwi",
    "Uganda":   "uga",
}


def download_hrsl_country(country: str, output_dir: str) -> str | None:
    """
    Download HRSL GeoTIFF for a given country ISO3 code.
    Returns local path or None if unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)
    iso3 = _COUNTRY_ISO3.get(country, country.lower()[:3])

    # Try population_<iso3>_2018-10-01.tif naming
    tif_name  = f"population_{iso3}_2018-10-01.tif"
    local_path = os.path.join(output_dir, tif_name)

    if os.path.exists(local_path):
        logger.info("HRSL %s already exists.", tif_name)
        return local_path

    s3_url = f"{_S3_BASE}hrsl_{iso3}/{tif_name}"

    # Try aws s3 cp
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_url, local_path, "--no-sign-request"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info("Downloaded HRSL %s", tif_name)
            return local_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: HDX direct HTTPS
    hdx_url = (
        f"https://data.humdata.org/dataset/highresolutionpopulationdensitymaps-{iso3}"
        f"/resource/{tif_name}"
    )
    try:
        result = subprocess.run(
            ["curl", "-L", "-o", local_path, hdx_url],
            capture_output=True, timeout=600,
        )
        if result.returncode == 0 and os.path.getsize(local_path) > 10000:
            logger.info("Downloaded HRSL %s via HDX", tif_name)
            return local_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    logger.warning(
        "Could not download HRSL for %s (%s). "
        "Try manually from https://data.humdata.org/dataset/"
        "highresolutionpopulationdensitymaps",
        country, iso3,
    )
    if os.path.exists(local_path):
        os.remove(local_path)
    return None


def extract_hrsl_patch(
    tif_path: str,
    lon: float,
    lat: float,
    patch_size_m: float = 2560,
    output_size: int = 256,
    clip_max: float = 1000.0,
) -> np.ndarray:
    """
    Extract a population-density patch (1, H, W) centred on (lon, lat).
    Pixel values are clipped to clip_max then normalised to [0, 1].

    Returns zeros if the file doesn't cover the point.
    """
    half = patch_size_m / 2
    deg  = 1 / 111320
    minx, maxx = lon - half * deg, lon + half * deg
    miny, maxy = lat - half * deg, lat + half * deg

    try:
        with rasterio.open(tif_path) as src:
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            arr = src.read(1, window=win, out_shape=(output_size, output_size),
                           resampling=Resampling.bilinear).astype(np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            arr = np.clip(arr / clip_max, 0, 1)
            return arr[np.newaxis]  # (1, H, W)
    except Exception as e:
        logger.debug("HRSL extract failed for (%.4f, %.4f): %s", lon, lat, e)
        return np.zeros((1, output_size, output_size), dtype=np.float32)


def main():
    cfg = load_config()
    out_dir = cfg.get("hrsl", {}).get("tiles_dir", "data/raw/hrsl")
    countries = cfg.get("hrsl", {}).get("countries", list(_COUNTRY_ISO3.keys()))

    for country in countries:
        path = download_hrsl_country(country, out_dir)
        if path:
            logger.info("HRSL ready: %s → %s", country, path)


if __name__ == "__main__":
    main()
