"""
Download VIIRS Black Marble monthly nighttime lights (VNP46A2) via NASA Earthdata.

Prerequisites (free):
  1. Register at https://urs.earthdata.nasa.gov
  2. Run: earthaccess.login(strategy="interactive") once to cache credentials,
     or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars.

Output: single-band GeoTIFF of mean annual radiance (nW/cm²/sr) clipped to bbox.
Normalized to [0, 1] at training time by clipping to viirs.normalization_clip.
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.transform import from_bounds as transform_from_bounds
import earthaccess

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# HDF5 subdataset name inside each VNP46A2 granule for the gap-filled DNB radiance
_DNB_LAYER = "//HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/Gap_Filled_DNB_BRDF-Corrected_NTL"
_FILL_VALUE = 65535  # VNP46A2 fill/no-data value


def _granule_to_array(filepath: str):
    """Read the DNB radiance layer from a VNP46A2 HDF5 granule."""
    hdf_path = f"HDF5:{filepath}:{_DNB_LAYER}"
    with rasterio.open(hdf_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs or CRS.from_epsg(4326)
    # Apply scale factor (VNP46A2 stores radiance * 0.1)
    data = np.where(data == _FILL_VALUE, np.nan, data * 0.1)
    return data, transform, crs


def main():
    cfg = load_config()
    vcfg = cfg["viirs"]
    bbox = vcfg["bbox"]
    year = vcfg["year"]
    short_name = vcfg["short_name"]
    output_path = cfg["data"]["viirs_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Authenticating with NASA Earthdata...")
    earthaccess.login(strategy="environment")   # reads EARTHDATA_USERNAME / PASSWORD env vars

    logger.info("Searching for %s granules over bbox=%s year=%s", short_name, bbox, year)
    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=tuple(bbox),
        temporal=(f"{year}-01-01", f"{year}-12-31"),
    )

    if not results:
        logger.error("No %s granules found for bbox=%s year=%s", short_name, bbox, year)
        return
    logger.info("Found %d granules. Downloading...", len(results))

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        files = earthaccess.download(results, local_path=tmpdir)
        logger.info("Downloaded %d files", len(files))

        arrays, transforms, crs = [], [], None
        for f in files:
            try:
                arr, tfm, c = _granule_to_array(f)
                arrays.append(arr)
                transforms.append(tfm)
                if crs is None:
                    crs = c
            except Exception as e:
                logger.warning("Could not read granule %s: %s", f, e)

        if not arrays:
            logger.error("No valid granules could be read.")
            return

        # Average all monthly composites into one annual mean
        stack = np.stack(arrays, axis=0)
        annual_mean = np.nanmean(stack, axis=0)

        # Clip to bbox using the first granule's transform as reference
        # Write to temp GeoTIFF then re-read with windowed read
        h, w = annual_mean.shape
        full_tif = os.path.join(tmpdir, "viirs_annual.tif")
        with rasterio.open(
            full_tif, "w",
            driver="GTiff", height=h, width=w,
            count=1, dtype="float32", crs=crs,
            transform=transforms[0],
            nodata=np.nan,
        ) as dst:
            dst.write(annual_mean[np.newaxis])

        with rasterio.open(full_tif) as src:
            window = from_bounds(*bbox, transform=src.transform)
            clipped = src.read(1, window=window)
            out_transform = src.window_transform(window)
            out_profile = src.profile.copy()
            out_profile.update({
                "height": clipped.shape[0],
                "width": clipped.shape[1],
                "transform": out_transform,
            })

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(clipped[np.newaxis])

    valid = clipped[~np.isnan(clipped)]
    logger.info(
        "VIIRS annual mean saved to %s | shape=%s | radiance min=%.2f max=%.2f mean=%.2f nW/cm²/sr",
        output_path, clipped.shape, valid.min(), valid.max(), valid.mean(),
    )


if __name__ == "__main__":
    main()
