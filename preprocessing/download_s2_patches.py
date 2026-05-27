"""
Download Sentinel-2 RGB patches centred on DHS cluster locations.

For each cluster in data/dhs_survey.csv (filtered by --country), we:
  1. Query Planetary Computer STAC for the least-cloudy S2 L2A scene.
  2. Do a windowed COG read — only the cluster's tile is transferred.
  3. Normalise to [0, 1] and save as float32 .npy (3 × patch_px × patch_px).
  4. Append a row to data/patches/s2_<country>/index.csv.

Usage:
  python -m preprocessing.download_s2_patches --country Kenya
  python -m preprocessing.download_s2_patches --country Kenya --max 100 --workers 4

The script is resumable: clusters already present in index.csv are skipped.
"""

import argparse
import os
import csv
import time
import numpy as np
import pandas as pd
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

STAC_URL    = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATE_RANGE  = "2022-01-01/2023-12-31"
MAX_CLOUD   = 10            # % cloud cover threshold
PATCH_DEG   = 0.025         # half-width in degrees (~2.5 km at equator)
PATCH_PX    = 64            # output patch size in pixels
NORM_FACTOR = 10_000.0      # S2 L2A reflectance scale
BANDS       = ["B04", "B03", "B02"]   # R, G, B at 10 m


def _fetch_patch(row, out_dir: str) -> dict | None:
    """Download one S2 patch. Returns index row dict or None on failure."""
    cluster_id = int(row["cluster_id"])
    lon, lat   = float(row["longitude"]), float(row["latitude"])
    out_path   = os.path.join(out_dir, f"s2_{cluster_id:05d}.npy")

    bbox = [lon - PATCH_DEG, lat - PATCH_DEG, lon + PATCH_DEG, lat + PATCH_DEG]

    try:
        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=DATE_RANGE,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
            max_items=5,
        )
        items = list(search.items())
        if not items:
            return None

        items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
        item = items[0]

        arrays = []
        for band in BANDS:
            if band not in item.assets:
                return None
            href = item.assets[band].href
            with rasterio.open(href) as src:
                # S2 tiles are in UTM — reproject the WGS84 bbox to scene CRS
                bbox_native = transform_bounds("EPSG:4326", src.crs, *bbox)
                window = from_bounds(*bbox_native, transform=src.transform)
                arr = src.read(1, window=window, out_shape=(PATCH_PX, PATCH_PX),
                               resampling=rasterio.enums.Resampling.bilinear)
                arrays.append(arr.astype(np.float32))

        patch = np.stack(arrays, axis=0) / NORM_FACTOR   # (3, H, W) in [0, 1]

        # Discard near-black patches (cloud/nodata)
        if patch.mean() < 0.01:
            return None

        np.save(out_path, patch.astype(np.float32))
        return {
            "cluster_id":   cluster_id,
            "country":      row["country"],
            "latitude":     lat,
            "longitude":    lon,
            "wealth_index": float(row["wealth_index"]),
            "patch_path":   out_path,
            "scene_date":   item.datetime.strftime("%Y-%m-%d") if item.datetime else "",
            "cloud_cover":  item.properties.get("eo:cloud_cover", -1),
        }

    except Exception as e:
        return None   # silently skip; tqdm will show overall progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country",  default="Kenya")
    parser.add_argument("--max",      type=int, default=None,
                        help="Max clusters to process (for testing)")
    parser.add_argument("--workers",  type=int, default=4,
                        help="Parallel download threads")
    args = parser.parse_args()

    survey = pd.read_csv("data/dhs_survey.csv")
    df = survey[survey["country"] == args.country].copy().reset_index(drop=True)
    df["cluster_id"] = df.index

    if args.max:
        df = df.head(args.max)

    out_dir   = f"data/patches/s2_{args.country.lower()}"
    index_csv = os.path.join(out_dir, "index.csv")
    os.makedirs(out_dir, exist_ok=True)

    # Resume: skip already-downloaded clusters
    done_ids = set()
    if os.path.exists(index_csv):
        existing = pd.read_csv(index_csv)
        done_ids = set(existing["cluster_id"].tolist())
        print(f"Resuming — {len(done_ids)} already downloaded.")

    df = df[~df["cluster_id"].isin(done_ids)]
    print(f"Downloading {len(df)} patches for {args.country} ({args.workers} threads) ...")

    index_mode = "a" if done_ids else "w"
    fieldnames = ["cluster_id","country","latitude","longitude",
                  "wealth_index","patch_path","scene_date","cloud_cover"]

    success = 0
    with open(index_csv, index_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if index_mode == "w":
            writer.writeheader()

        rows = df.to_dict("records")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fetch_patch, r, out_dir): r for r in rows}
            with tqdm(total=len(rows), unit="patch") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        writer.writerow(result)
                        f.flush()
                        success += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=success, skip=pbar.n - success)

    print(f"\nDone: {success}/{len(df)} patches saved to {out_dir}/")
    print(f"Index: {index_csv}")


if __name__ == "__main__":
    main()
