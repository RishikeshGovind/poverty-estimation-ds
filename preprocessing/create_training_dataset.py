import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np
import os

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    cfg = load_config()
    survey_path = cfg["data"]["survey_path"]
    image_path = cfg["data"]["image_path"]
    output_dir = Path(cfg["data"]["patches_dir"])
    patch_size = cfg["training"]["patch_size"]
    out_csv = cfg["data"]["training_csv"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(survey_path):
        logger.error("Survey file not found: %s. Run load_dhs_data.py first.", survey_path)
        return
    if not os.path.exists(image_path):
        logger.error("Image file not found: %s. Run download_sentinel.py first.", image_path)
        return

    df = pd.read_csv(survey_path)
    half_size = patch_size // 2
    records = []

    with rasterio.open(image_path) as src:
        # Verify CRS is geographic (lat/lon) before calling src.index with lon/lat
        if not src.crs.is_geographic:
            logger.error(
                "Image CRS %s is not geographic. Re-project to EPSG:4326 before extraction.",
                src.crs,
            )
            return

        logger.info("Extracting patches from %s for %d survey points...", image_path, len(df))
        for idx, row in df.iterrows():
            lon, lat = row["longitude"], row["latitude"]
            try:
                py, px = src.index(lon, lat)
                window = Window(px - half_size, py - half_size, patch_size, patch_size)
                patch = src.read(window=window, boundless=True, fill_value=0)

                if np.count_nonzero(patch) > (patch_size * patch_size * 0.5):
                    patch_path = output_dir / f"survey_patch_{idx:05d}.npy"
                    np.save(patch_path, patch)
                    records.append({
                        "patch_file": str(patch_path),
                        "latitude": lat,
                        "longitude": lon,
                        "label": row["wealth_index"],
                    })
            except Exception as e:
                logger.warning("Skipped point %d (lon=%.4f, lat=%.4f): %s", idx, lon, lat, e)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    logger.info("Created training dataset: %d samples -> %s", len(records), out_csv)

if __name__ == "__main__":
    main()
