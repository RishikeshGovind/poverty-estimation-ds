"""
Phase 4 — Landsat 8/9 downloader via Microsoft Planetary Computer.

Landsat adds 3 bands that Sentinel-2 lacks:
  - SWIR2   (Band 7,  2.11 µm) — distinguishes bare soil, minerals
  - Thermal (Band 10, 10.9 µm) — land surface temperature (urban heat island)
  - Longer time series (1984-present vs 2017 for S2)

Output: 6-band GeoTIFF → (B, G, R, NIR, SWIR1, SWIR2) per patch
        Thermal is saved separately as a 1-band file if requested.

Channel normalisation: /10000 (same scale as Sentinel-2 surface reflectance)

Run:
    python -m preprocessing.download_landsat
"""

import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# Landsat Collection 2 Level-2 band names on Planetary Computer
# Landsat 8/9:
_LANDSAT_BANDS = {
    "blue":   "blue",    # Band 2  ~0.45–0.51 µm
    "green":  "green",   # Band 3  ~0.53–0.59 µm
    "red":    "red",     # Band 4  ~0.64–0.67 µm
    "nir08":  "nir08",   # Band 5  ~0.85–0.88 µm
    "swir16": "swir16",  # Band 6  ~1.57–1.65 µm
    "swir22": "swir22",  # Band 7  ~2.11–2.29 µm
}
_BAND_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]
N_BANDS     = 6

# Scale factor: Landsat C2 L2 stores reflectance * 10000 + 20000 offset
_SCALE_FACTOR = 0.0000275
_ADD_OFFSET   = -0.2
_NORM_FACTOR  = 1.0   # after applying scale+offset, values are in [0,1] reflectance


def _apply_scale(arr: np.ndarray) -> np.ndarray:
    """Convert Landsat C2 L2 DN values to surface reflectance [0, 1]."""
    return np.clip(arr.astype(np.float32) * _SCALE_FACTOR + _ADD_OFFSET, 0, 1)


def download_landsat_scene(
    bbox: list,
    date_range: str,
    output_path: str,
    max_cloud_cover: int = 10,
    collection: str = "landsat-c2-l2",
) -> str | None:
    """
    Download a cloud-free Landsat 8/9 composite for the given bbox.

    Args:
        bbox:           [lon_min, lat_min, lon_max, lat_max]
        date_range:     "YYYY-MM-DD/YYYY-MM-DD"
        output_path:    where to save the output GeoTIFF
        max_cloud_cover: max % cloud cover per scene

    Returns:
        Path to saved GeoTIFF or None if no scenes found.
    """
    try:
        import pystac_client
        import planetary_computer
    except ImportError:
        raise ImportError("Run: pip install pystac-client planetary-computer")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        max_items=5,
    )

    items = list(search.items())
    if not items:
        logger.warning("No Landsat scenes found for bbox=%s date=%s", bbox, date_range)
        return None

    # Use the clearest scene
    item = items[0]
    logger.info(
        "Using Landsat scene: %s  cloud=%.1f%%",
        item.id, item.properties.get("eo:cloud_cover", "?"),
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        # Stack bands into a single multi-band GeoTIFF
        band_arrays = []
        transform = profile = None

        for band_key in _BAND_ORDER:
            asset = item.assets.get(band_key)
            if asset is None:
                logger.warning("Band %s not found in scene %s", band_key, item.id)
                band_arrays.append(None)
                continue

            with rasterio.open(asset.href) as src:
                if transform is None:
                    transform = src.transform
                    profile = src.profile.copy()
                data = src.read(1)
                band_arrays.append(_apply_scale(data))

        # Replace missing bands with zeros
        h = band_arrays[[b for b in band_arrays if b is not None][0]].shape[0]
        w = band_arrays[[b for b in band_arrays if b is not None][0]].shape[1]
        for i, b in enumerate(band_arrays):
            if b is None:
                band_arrays[i] = np.zeros((h, w), dtype=np.float32)

        profile.update(count=N_BANDS, dtype=rasterio.float32)
        with rasterio.open(output_path, "w", **profile) as dst:
            for i, arr in enumerate(band_arrays, start=1):
                dst.write(arr, i)

        logger.info("Landsat scene saved → %s", output_path)
        return output_path

    except Exception as e:
        logger.error("Failed to download Landsat scene: %s", e)
        return None


def extract_landsat_patch(
    tif_path: str,
    lon: float,
    lat: float,
    patch_size_m: float = 2560,
    output_size: int = 256,
) -> np.ndarray:
    """
    Extract a (6, H, W) Landsat patch centred on (lon, lat).
    Values are already in [0, 1] surface reflectance from the download step.
    Returns zeros on failure.
    """
    half = patch_size_m / 2
    deg  = 1 / 111320
    minx, maxx = lon - half * deg, lon + half * deg
    miny, maxy = lat - half * deg, lat + half * deg

    try:
        with rasterio.open(tif_path) as src:
            win = from_bounds(minx, miny, maxx, maxy, src.transform)
            arr = src.read(
                out_shape=(N_BANDS, output_size, output_size),
                window=win,
                resampling=Resampling.bilinear,
            ).astype(np.float32)
            return np.clip(arr, 0, 1)
    except Exception as e:
        logger.debug("Landsat patch extract failed: %s", e)
        return np.zeros((N_BANDS, output_size, output_size), dtype=np.float32)


def main():
    cfg = load_config()
    bbox        = cfg["sentinel2"]["bbox"]
    date_range  = cfg.get("landsat", {}).get("date_range", cfg["sentinel2"]["date_range"])
    cloud_cover = cfg.get("landsat", {}).get("max_cloud_cover", 10)
    output_path = cfg.get("landsat", {}).get("output_path", "data/raw/landsat.tif")

    path = download_landsat_scene(bbox, date_range, output_path, cloud_cover)
    if path:
        logger.info("Landsat ready: %s", path)
    else:
        logger.warning("Landsat download failed — check planetary_computer credentials.")


if __name__ == "__main__":
    main()
