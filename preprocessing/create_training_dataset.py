"""
Extract image patches centered at DHS survey coordinates from every available
sensor (Sentinel-2, Sentinel-1, VIIRS) and write a training CSV.

For each survey point the script:
  1. Extracts a patch_size × patch_size window from the S2 GeoTIFF (10 m/px).
  2. Derives the geographic bounds of that window.
  3. Re-projects those bounds into S1's CRS and extracts the corresponding window.
  4. Does the same for VIIRS (500 m/px — the patch will be ~5×5 px; the dataset
     class up-samples it to patch_size × patch_size at load time).

Sensors 1 and 2 are optional: if the file doesn't exist the column is left empty
and the MultiSensorDataset will fill the missing channels with zeros.
"""

import os
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from pathlib import Path

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _extract_window(src, geo_bounds_src_crs, fill_value=0):
    """Read a window defined by geographic bounds (in the source's own CRS)."""
    win = window_from_bounds(*geo_bounds_src_crs, transform=src.transform)
    return src.read(window=win, boundless=True, fill_value=fill_value).astype(np.float32)


def main():
    cfg = load_config()
    survey_path = cfg["data"]["survey_path"]
    s2_path = cfg["data"]["image_path"]
    s1_path = cfg["data"].get("sentinel1_path", "")
    viirs_path = cfg["data"].get("viirs_path", "")
    output_dir = Path(cfg["data"]["patches_dir"])
    patch_size = cfg["training"]["patch_size"]
    out_csv = cfg["data"]["training_csv"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, path in [("survey", survey_path), ("S2 image", s2_path)]:
        if not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            return

    df = pd.read_csv(survey_path)
    half = patch_size // 2
    records = []

    s1_available = bool(s1_path) and os.path.exists(s1_path)
    viirs_available = bool(viirs_path) and os.path.exists(viirs_path)
    logger.info(
        "Extracting patches | S2=yes  S1=%s  VIIRS=%s  n_points=%d",
        "yes" if s1_available else "no",
        "yes" if viirs_available else "no",
        len(df),
    )

    s1_src = rasterio.open(s1_path) if s1_available else None
    viirs_src = rasterio.open(viirs_path) if viirs_available else None

    try:
        with rasterio.open(s2_path) as s2_src:
            if not s2_src.crs.is_geographic:
                logger.error("S2 image CRS %s is not geographic. Re-project to EPSG:4326.", s2_src.crs)
                return

            for idx, row in df.iterrows():
                lon, lat = row["longitude"], row["latitude"]
                try:
                    py, px = s2_src.index(lon, lat)
                    s2_win = Window(px - half, py - half, patch_size, patch_size)
                    s2_patch = s2_src.read(window=s2_win, boundless=True, fill_value=0).astype(np.float32)

                    if np.count_nonzero(s2_patch) <= patch_size * patch_size * 0.5:
                        logger.debug("Skipping point %d — S2 patch mostly empty.", idx)
                        continue

                    # Geographic bounds of this S2 window (in S2's CRS = EPSG:4326)
                    geo_bounds_4326 = array_bounds(
                        patch_size, patch_size, s2_src.window_transform(s2_win)
                    )

                    # --- Save S2 patch ---
                    s2_file = str(output_dir / f"s2_{idx:05d}.npy")
                    np.save(s2_file, s2_patch)

                    # --- S1 patch (optional) ---
                    s1_file = ""
                    if s1_src is not None:
                        try:
                            s1_bounds = transform_bounds(s2_src.crs, s1_src.crs, *geo_bounds_4326)
                            s1_patch = _extract_window(s1_src, s1_bounds)
                            s1_file = str(output_dir / f"s1_{idx:05d}.npy")
                            np.save(s1_file, s1_patch)
                        except Exception as e:
                            logger.warning("S1 extraction failed for point %d: %s", idx, e)

                    # --- VIIRS patch (optional) ---
                    viirs_file = ""
                    if viirs_src is not None:
                        try:
                            viirs_bounds = transform_bounds(s2_src.crs, viirs_src.crs, *geo_bounds_4326)
                            viirs_patch = _extract_window(viirs_src, viirs_bounds)
                            viirs_file = str(output_dir / f"viirs_{idx:05d}.npy")
                            np.save(viirs_file, viirs_patch)
                        except Exception as e:
                            logger.warning("VIIRS extraction failed for point %d: %s", idx, e)

                    records.append({
                        "patch_file": s2_file,       # backward-compat alias
                        "s2_patch_file": s2_file,
                        "s1_patch_file": s1_file,
                        "viirs_patch_file": viirs_file,
                        "latitude": lat,
                        "longitude": lon,
                        "label": row["wealth_index"],
                    })

                except Exception as e:
                    logger.warning("Skipped point %d (lon=%.4f lat=%.4f): %s", idx, lon, lat, e)

    finally:
        if s1_src:
            s1_src.close()
        if viirs_src:
            viirs_src.close()

    pd.DataFrame(records).to_csv(out_csv, index=False)
    logger.info("Training dataset: %d samples → %s", len(records), out_csv)


if __name__ == "__main__":
    main()
