"""
Phase 3 — Microsoft / Google Open Buildings density extractor.

Building count and mean footprint area per cluster are among the strongest
single predictors of wealth — better than nighttime lights in dense urban areas.

Sources:
  - Microsoft + Google + OSM combined: https://source.coop/vida/google-microsoft-osm-open-buildings
  - Africa-specific (Microsoft): https://github.com/microsoft/Uganda-Tanzania-Building-Footprints
  - Google Open Buildings v3 (global): https://sites.research.google/open-buildings/

This script:
  1. Downloads building footprint GeoParquet tiles that cover a given bbox
     from the VIDA combined dataset on Source Cooperative (free, no login).
  2. For each survey cluster, computes:
       - building_count     — # buildings within 1km radius
       - mean_area_m2       — mean footprint area within 1km radius
       - building_density   — buildings per km²
  3. Rasterises building density to a (1, 256, 256) patch saved as .npy
     so it can be used as a new sensor channel in MultiSensorDataset.
  4. Adds 'buildings_patch_file' column to the training CSV.

Run:
    python -m preprocessing.extract_building_features
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import requests
import warnings

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# VIDA combined buildings GeoParquet (quadkey-indexed tiles, no auth needed)
_VIDA_BASE_URL = "https://data.source.coop/vida/google-microsoft-osm-open-buildings/geoparquet/"

# Google Open Buildings v3 direct GCS URL (public)
_GOOGLE_BUILDINGS_BASE = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/"


def _download_tile_parquet(tile_url: str, cache_dir: str) -> str | None:
    """Download a GeoParquet tile, return local path or None on failure."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(tile_url.split("?")[0])
    local = os.path.join(cache_dir, filename)
    if os.path.exists(local):
        return local
    try:
        resp = requests.get(tile_url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(local, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded buildings tile: %s", filename)
        return local
    except Exception as e:
        logger.warning("Could not download %s: %s", tile_url, e)
        return None


def load_buildings_bbox(
    bbox: list,
    cache_dir: str = "data/raw/buildings",
    source: str = "google",
) -> gpd.GeoDataFrame | None:
    """
    Load building footprints for a bounding box.

    Args:
        bbox:      [lon_min, lat_min, lon_max, lat_max]
        cache_dir: local cache directory
        source:    "google" uses Google Open Buildings v3 (broader coverage)

    Returns:
        GeoDataFrame with building polygons, or None if unavailable.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    if source == "google":
        # Google Open Buildings v3 uses S2 level-4 cells (~roughly 5°×5°)
        # File naming: <lat_floor>_<lon_floor>_buildings.csv.gz
        lat_floor = int(math.floor(lat_min / 5) * 5)
        lon_floor = int(math.floor(lon_min / 5) * 5)
        lat_str   = f"{'N' if lat_floor >= 0 else 'S'}{abs(lat_floor):02d}"
        lon_str   = f"{'E' if lon_floor >= 0 else 'W'}{abs(lon_floor):03d}"
        fname     = f"{lat_str}_{lon_str}_buildings.csv.gz"
        url       = _GOOGLE_BUILDINGS_BASE + fname

        local = _download_tile_parquet(url, cache_dir)
        if not local:
            return None

        try:
            df = pd.read_csv(local, compression="gzip",
                             usecols=["latitude", "longitude", "area_in_meters",
                                      "confidence", "geometry"])
            # Filter to bbox
            df = df[
                (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
                (df["latitude"]  >= lat_min) & (df["latitude"]  <= lat_max) &
                (df["confidence"] >= 0.7)
            ]
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                crs="EPSG:4326"
            )
            logger.info("Loaded %d buildings for bbox %s", len(gdf), bbox)
            return gdf
        except Exception as e:
            logger.warning("Failed to parse buildings file: %s", e)
            return None

    return None


def compute_cluster_building_features(
    survey_df: pd.DataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    radius_m: float = 1000.0,
) -> pd.DataFrame:
    """
    For each survey cluster, compute building statistics within radius_m.

    Adds columns: building_count, mean_area_m2, building_density_per_km2
    """
    deg_per_m = 1 / 111320
    radius_deg = radius_m * deg_per_m

    counts, areas, densities = [], [], []

    for _, row in survey_df.iterrows():
        lon, lat = row["longitude"], row["latitude"]
        # Approximate circle as bbox for speed, then filter by distance
        clip = buildings_gdf.cx[
            lon - radius_deg: lon + radius_deg,
            lat - radius_deg: lat + radius_deg,
        ]
        if len(clip) > 0:
            dists = np.sqrt(
                (clip.geometry.x - lon) ** 2 + (clip.geometry.y - lat) ** 2
            )
            within = clip[dists <= radius_deg]
        else:
            within = clip

        n = len(within)
        counts.append(n)
        area_col = "area_in_meters" if "area_in_meters" in within.columns else None
        areas.append(within[area_col].mean() if area_col and n > 0 else 0.0)
        area_km2 = math.pi * (radius_m / 1000) ** 2
        densities.append(n / area_km2)

    survey_df = survey_df.copy()
    survey_df["building_count"]          = counts
    survey_df["mean_area_m2"]            = areas
    survey_df["building_density_per_km2"] = densities
    return survey_df


def rasterize_building_density(
    buildings_gdf: gpd.GeoDataFrame,
    lon: float,
    lat: float,
    patch_size_m: float = 2560,
    output_size: int = 256,
    clip_max: float = 500.0,
) -> np.ndarray:
    """
    Rasterise building point density into a (1, H, W) patch centred on (lon, lat).
    Each pixel = building count in that cell, normalised to [0, 1].
    """
    half = patch_size_m / 2
    deg  = 1 / 111320
    lon_min = lon - half * deg;  lon_max = lon + half * deg
    lat_min = lat - half * deg;  lat_max = lat + half * deg

    clip = buildings_gdf.cx[lon_min:lon_max, lat_min:lat_max]

    grid = np.zeros((output_size, output_size), dtype=np.float32)
    if len(clip) > 0:
        lon_vals = np.array(clip.geometry.x)
        lat_vals = np.array(clip.geometry.y)
        # Map to pixel coordinates
        col_idx = ((lon_vals - lon_min) / (lon_max - lon_min) * output_size).astype(int)
        row_idx = ((lat_max - lat_vals) / (lat_max - lat_min) * output_size).astype(int)
        valid   = (col_idx >= 0) & (col_idx < output_size) & \
                  (row_idx >= 0) & (row_idx < output_size)
        for r, c in zip(row_idx[valid], col_idx[valid]):
            grid[r, c] += 1.0

    grid = np.clip(grid / clip_max, 0, 1)
    return grid[np.newaxis]  # (1, H, W)


def extract_and_save_patches(
    survey_csv: str,
    buildings_gdf: gpd.GeoDataFrame,
    patches_dir: str,
    output_csv: str = None,
    patch_size: int = 256,
):
    """
    Generate building-density .npy patches for all clusters and update the CSV.
    """
    df = pd.read_csv(survey_csv)
    os.makedirs(patches_dir, exist_ok=True)
    patch_paths = []

    for i, row in df.iterrows():
        fname = os.path.join(patches_dir, f"buildings_{i:05d}.npy")
        patch = rasterize_building_density(
            buildings_gdf, row["longitude"], row["latitude"],
            output_size=patch_size,
        )
        np.save(fname, patch)
        patch_paths.append(fname)

    df["buildings_patch_file"] = patch_paths
    output_csv = output_csv or survey_csv
    df.to_csv(output_csv, index=False)
    logger.info("Building patches saved → %s", patches_dir)
    return output_csv


def main():
    cfg = load_config()
    bbox        = cfg["sentinel2"]["bbox"]
    survey_csv  = cfg["data"]["training_csv"]
    patches_dir = os.path.join(cfg["data"]["patches_dir"], "buildings")
    cache_dir   = cfg.get("buildings", {}).get("cache_dir", "data/raw/buildings")

    logger.info("Loading building footprints for bbox %s…", bbox)
    buildings = load_buildings_bbox(bbox, cache_dir=cache_dir, source="google")

    if buildings is None or len(buildings) == 0:
        logger.warning(
            "No building data available. "
            "Check internet connection or download manually from "
            "https://sites.research.google/open-buildings/"
        )
        return

    if os.path.exists(survey_csv):
        extract_and_save_patches(survey_csv, buildings, patches_dir)
    else:
        logger.warning("Training CSV not found at %s. Run create_training_dataset.py first.", survey_csv)


if __name__ == "__main__":
    main()
