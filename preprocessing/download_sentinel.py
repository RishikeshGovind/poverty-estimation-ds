import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
import numpy as np
import os

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    cfg = load_config()
    s2cfg = cfg["sentinel2"]
    bbox = s2cfg["bbox"]
    date_range = s2cfg["date_range"]
    max_cloud = s2cfg["max_cloud_cover"]
    output_path = cfg["data"]["image_path"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not (len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f"Invalid bbox: {bbox}. Expected [min_lon, min_lat, max_lon, max_lat].")

    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(stac_url)

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )

    items = list(search.get_items())
    if not items:
        logger.error("No images found for bbox=%s, dates=%s, max_cloud=%s", bbox, date_range, max_cloud)
        return

    items.sort(key=lambda item: item.properties.get("eo:cloud_cover", 100))
    item = items[0]
    logger.info(
        "Selected image: date=%s, cloud_cover=%.1f%%",
        item.datetime, item.properties.get("eo:cloud_cover", 0),
    )

    required_bands = {"red": "B04", "green": "B03", "blue": "B02"}
    missing = [b for b in required_bands.values() if b not in item.assets]
    if missing:
        logger.error("Selected image is missing required bands: %s", missing)
        return

    arrays = []
    profile = None
    for band in required_bands.values():
        signed_href = planetary_computer.sign(item.assets[band].href)
        with rasterio.open(signed_href) as src:
            window = from_bounds(*bbox, transform=src.transform)
            arr = src.read(1, window=window)
            arrays.append(arr)
            if profile is None:
                profile = src.profile.copy()
                profile.update({
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "transform": src.window_transform(window),
                })

    rgb = np.stack(arrays, axis=0)
    profile.update(count=3, driver="GTiff")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb)

    logger.info("RGB image saved to %s", output_path)

if __name__ == "__main__":
    main()
