"""
Download VIIRS nighttime lights patches centred on DHS cluster locations.

Uses NASA GIBS WMS (no auth) to request a 64×64 PNG for each cluster's bbox.
Patches saved as float32 .npy (1 × 64 × 64), values in [0, 1].

Usage:
  python -m preprocessing.download_viirs_patches --country Kenya
  python -m preprocessing.download_viirs_patches --country Nigeria --workers 8
"""

import argparse
import os
import csv
import io
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# GIBS WMS — accepts any bbox, no tile math needed
WMS_URL = (
    "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
    "&LAYERS=VIIRS_SNPP_DayNightBand_ENCC"
    "&SRS=EPSG:4326&FORMAT=image/png"
    "&TIME=2023-01-01"
    "&WIDTH=64&HEIGHT=64"
    "&BBOX={minx},{miny},{maxx},{maxy}"
)

PATCH_DEG = 0.25     # half-width (~25 km) — VIIRS is 500 m/px so 50 km ≈ 100 px; we resize to 64
PATCH_PX  = 64


def _fetch_viirs_patch(row, out_dir: str) -> dict | None:
    cluster_id = int(row["cluster_id"])
    lon, lat   = float(row["longitude"]), float(row["latitude"])
    out_path   = os.path.join(out_dir, f"viirs_{cluster_id:05d}.npy")

    minx, miny = lon - PATCH_DEG, lat - PATCH_DEG
    maxx, maxy = lon + PATCH_DEG, lat + PATCH_DEG
    url = WMS_URL.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None

        img = Image.open(io.BytesIO(resp.content)).convert("L")  # grayscale
        arr = np.array(img.resize((PATCH_PX, PATCH_PX), Image.BILINEAR), dtype=np.float32)

        # PNG pixel values 0–255 map to radiance; normalise to [0, 1]
        arr = np.clip(arr / 255.0, 0, 1)
        patch = arr[np.newaxis, :, :]   # (1, 64, 64)
        np.save(out_path, patch)

        return {
            "cluster_id":   cluster_id,
            "country":      row["country"],
            "latitude":     lat,
            "longitude":    lon,
            "wealth_index": float(row["wealth_index"]),
            "patch_path":   out_path,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country",  default="Kenya")
    parser.add_argument("--max",      type=int, default=None)
    parser.add_argument("--workers",  type=int, default=8)
    args = parser.parse_args()

    survey = pd.read_csv("data/dhs_survey.csv")
    df = survey[survey["country"] == args.country].copy().reset_index(drop=True)
    df["cluster_id"] = df.index

    if args.max:
        df = df.head(args.max)

    out_dir   = f"data/patches/viirs_{args.country.lower()}"
    index_csv = os.path.join(out_dir, "index.csv")
    os.makedirs(out_dir, exist_ok=True)

    # Clear previous failed attempt
    if os.path.exists(index_csv):
        existing = pd.read_csv(index_csv)
        if len(existing) == 0:
            os.remove(index_csv)

    done_ids = set()
    if os.path.exists(index_csv):
        done_ids = set(pd.read_csv(index_csv)["cluster_id"].tolist())
        print(f"Resuming — {len(done_ids)} already downloaded.")

    df = df[~df["cluster_id"].isin(done_ids)]
    print(f"Downloading {len(df)} VIIRS patches for {args.country} ({args.workers} threads) ...")

    fieldnames = ["cluster_id","country","latitude","longitude","wealth_index","patch_path"]
    index_mode = "a" if done_ids else "w"
    success = 0

    with open(index_csv, index_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if index_mode == "w":
            writer.writeheader()

        rows = df.to_dict("records")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_fetch_viirs_patch, r, out_dir): r for r in rows}
            with tqdm(total=len(rows), unit="patch") as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        writer.writerow(result)
                        f.flush()
                        success += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=success, skip=pbar.n - success)

    print(f"\nDone: {success}/{len(df)} VIIRS patches → {out_dir}/")


if __name__ == "__main__":
    main()
