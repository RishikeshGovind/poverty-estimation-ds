"""
Download Sentinel-1 RTC (Radiometrically Terrain Corrected) imagery from
Microsoft Planetary Computer.

Outputs a 2-band GeoTIFF (VV, VH) at the configured bbox.
VV captures surface roughness / built structures; VH captures vegetation volume.
Both are in linear power scale (not dB) — normalized to [0, 1] at training time
by clipping to sentinel1.normalization_clip and dividing.
"""

import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import numpy as np
import os
import tempfile

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def main():
    cfg = load_config()
    s1cfg = cfg["sentinel1"]
    bbox = s1cfg["bbox"]
    date_range = s1cfg["date_range"]
    polarizations = s1cfg["polarizations"]
    output_path = cfg["data"]["sentinel1_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"Invalid bbox: {bbox}. Expected [min_lon, min_lat, max_lon, max_lat].")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=date_range,
    )

    items = list(search.get_items())
    if not items:
        logger.error("No S1 RTC items found for bbox=%s, dates=%s", bbox, date_range)
        return

    # Use the most recent item — S1 revisits every ~12 days so any single pass works
    items.sort(key=lambda i: i.datetime, reverse=True)
    item = items[0]
    logger.info("Selected S1 item: %s  date=%s", item.id, item.datetime)

    missing = [p for p in polarizations if p not in item.assets]
    if missing:
        logger.error("Item %s missing polarization assets: %s", item.id, missing)
        return

    arrays = []
    profile = None
    for pol in polarizations:
        href = item.assets[pol].href
        with rasterio.open(href) as src:
            window = from_bounds(*bbox, transform=src.transform)
            arr = src.read(1, window=window).astype(np.float32)
            arrays.append(arr)
            if profile is None:
                profile = src.profile.copy()
                profile.update({
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "transform": src.window_transform(window),
                })
        logger.info("Downloaded %s band: shape=%s  min=%.4f  max=%.4f", pol, arr.shape, arr.min(), arr.max())

    stack = np.stack(arrays, axis=0)   # (2, H, W)
    profile.update(count=len(polarizations), driver="GTiff", dtype="float32")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)

    logger.info("S1 RTC image saved to %s  bands=%s", output_path, polarizations)


if __name__ == "__main__":
    main()
