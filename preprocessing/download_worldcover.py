"""
Phase 2 — ESA WorldCover 10m land-use / land-cover downloader.

WorldCover classifies every pixel as one of 11 classes:
  10=Trees  20=Shrubland  30=Grassland  40=Cropland  50=Built-up
  60=Bare   70=Snow       80=Water      90=Wetland   95=Mangrove  100=Moss

Built-up density (class 50) is a strong SDG-11 proxy.
Free, CC-BY 4.0. No registration required.

Download via AWS S3 (no credentials needed):
  2020 v100: s3://esa-worldcover/v100/2020/map/
  2021 v200: s3://esa-worldcover/v200/2021/map/

Run:
    python -m preprocessing.download_worldcover
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

# WorldCover S3 paths (no-sign-request — free)
_S3_BASE = {
    "2021": "s3://esa-worldcover/v200/2021/map/",
    "2020": "s3://esa-worldcover/v100/2020/map/",
}

# Tile naming: ESA_WorldCover_10m_<year>_v<ver>_<lat><lon>_Map.tif
# Tiles are 3×3 degree in EPSG:4326. We derive tile names from bbox.
def _tile_names(bbox: list, year: str = "2021") -> list[str]:
    """Return S3 tile URLs covering the given bbox [lon_min, lat_min, lon_max, lat_max]."""
    lon_min, lat_min, lon_max, lat_max = bbox

    # Tile origin = floor to nearest 3 degrees
    lons = range(int(lon_min // 3) * 3, int(lon_max // 3 + 1) * 3, 3)
    lats = range(int(lat_min // 3) * 3, int(lat_max // 3 + 1) * 3, 3)

    tiles = []
    ver = "v200" if year == "2021" else "v100"
    for lat in lats:
        for lon in lons:
            lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            name = f"ESA_WorldCover_10m_{year}_{ver}_{lat_str}{lon_str}_Map.tif"
            base = _S3_BASE[year]
            tiles.append(f"{base}{name}")
    return tiles


def download_worldcover_tiles(bbox: list, output_dir: str, year: str = "2021") -> list[str]:
    """
    Download WorldCover GeoTIFF tiles covering bbox using AWS CLI.
    Falls back to GDAL /vsicurl virtual filesystem if aws CLI is unavailable.

    Returns list of local tile paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    local_paths = []

    for s3_url in _tile_names(bbox, year):
        filename = os.path.basename(s3_url)
        local_path = os.path.join(output_dir, filename)

        if os.path.exists(local_path):
            logger.info("WorldCover tile already exists: %s", filename)
            local_paths.append(local_path)
            continue

        # Try aws s3 cp first (fastest)
        try:
            result = subprocess.run(
                ["aws", "s3", "cp", s3_url, local_path, "--no-sign-request"],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info("Downloaded WorldCover tile: %s", filename)
                local_paths.append(local_path)
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Fallback: GDAL virtual filesystem (slower but no AWS CLI needed)
        https_url = s3_url.replace("s3://esa-worldcover", "https://esa-worldcover.s3.eu-central-1.amazonaws.com")
        vsi_path  = f"/vsicurl/{https_url}"
        try:
            result = subprocess.run(
                ["gdal_translate", "-co", "COMPRESS=DEFLATE", vsi_path, local_path],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info("Downloaded WorldCover tile via gdal_translate: %s", filename)
                local_paths.append(local_path)
                continue
        except FileNotFoundError:
            pass

        logger.warning(
            "Could not download %s. Install AWS CLI: brew install awscli", filename
        )

    return local_paths


def extract_worldcover_patch(
    tile_paths: list[str],
    lon: float,
    lat: float,
    patch_size_m: float = 2560,   # metres → ~256 px at 10m
    output_size: int = 256,
) -> np.ndarray:
    """
    Extract a WorldCover patch (H, W) centred on (lon, lat).
    Values are normalised to [0, 1] by dividing by 100 (max class = 100).

    Returns a (1, output_size, output_size) float32 array, or zeros if no tile covers the point.
    """
    half = patch_size_m / 2
    deg_per_m = 1 / 111320
    minx = lon - half * deg_per_m
    maxx = lon + half * deg_per_m
    miny = lat - half * deg_per_m
    maxy = lat + half * deg_per_m

    for tile_path in tile_paths:
        try:
            with rasterio.open(tile_path) as src:
                win = from_bounds(minx, miny, maxx, maxy, src.transform)
                # Check tile covers the point
                if win.col_off < 0 or win.row_off < 0:
                    continue
                arr = src.read(1, window=win, out_shape=(output_size, output_size),
                               resampling=Resampling.nearest).astype(np.float32)
                return (arr / 100.0)[np.newaxis]  # (1, H, W)
        except Exception:
            continue

    return np.zeros((1, output_size, output_size), dtype=np.float32)


def main():
    cfg = load_config()
    bbox    = cfg["sentinel2"]["bbox"]
    out_dir = cfg.get("worldcover", {}).get("tiles_dir", "data/raw/worldcover")
    year    = str(cfg.get("worldcover", {}).get("year", "2021"))

    tiles = download_worldcover_tiles(bbox, out_dir, year=year)
    logger.info("WorldCover: %d tiles available for bbox %s", len(tiles), bbox)


if __name__ == "__main__":
    main()
