"""
Phase 5 — OpenStreetMap accessibility feature extractor.

Road network density and distance to key services (hospitals, schools, markets)
are strong predictors of economic integration and poverty access.
All data is free via the Overpass API — no registration.

Features computed per cluster:
  - road_density_km_per_km2   : total road length within 5km radius / area
  - dist_to_hospital_km       : nearest hospital / clinic
  - dist_to_school_km         : nearest school
  - dist_to_market_km         : nearest marketplace / supermarket
  - dist_to_bank_km           : nearest bank / ATM
  - n_amenities_1km           : total OSM amenity count within 1km

Run:
    python -m preprocessing.extract_osm_features
"""

import os
import math
import time
import numpy as np
import pandas as pd

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# Overpass API query templates
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Amenity categories mapped to feature columns
_AMENITY_QUERIES = {
    "hospital": '["amenity"~"hospital|clinic|health_post|pharmacy"]',
    "school":   '["amenity"~"school|university|college|kindergarten"]',
    "market":   '["amenity"~"marketplace|supermarket|mall"]',
    "bank":     '["amenity"~"bank|atm"]',
}


def _haversine_km(lon1, lat1, lon2_arr, lat2_arr):
    """Vectorised haversine distance in km from one point to many."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = phi2 - phi1
    dlam = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def _overpass_amenities(lon, lat, radius_m, amenity_filter, timeout=30):
    """
    Query Overpass API for nodes matching amenity_filter within radius_m of (lon, lat).
    Returns list of (lon, lat) tuples.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    query = f"""
    [out:json][timeout:{timeout}];
    (
      node{amenity_filter}(around:{radius_m},{lat},{lon});
      way{amenity_filter}(around:{radius_m},{lat},{lon});
    );
    out center;
    """
    try:
        resp = requests.post(_OVERPASS_URL, data={"data": query}, timeout=timeout + 5)
        resp.raise_for_status()
        data = resp.json()
        pts = []
        for el in data.get("elements", []):
            if el["type"] == "node":
                pts.append((el["lon"], el["lat"]))
            elif "center" in el:
                pts.append((el["center"]["lon"], el["center"]["lat"]))
        return pts
    except Exception as e:
        logger.debug("Overpass query failed: %s", e)
        return []


def _overpass_roads(lon, lat, radius_m, timeout=30):
    """Query road network length within radius_m. Returns total km."""
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    query = f"""
    [out:json][timeout:{timeout}];
    way["highway"~"primary|secondary|tertiary|unclassified|residential|trunk"]
      (around:{radius_m},{lat},{lon});
    out geom;
    """
    try:
        resp = requests.post(_OVERPASS_URL, data={"data": query}, timeout=timeout + 5)
        resp.raise_for_status()
        data = resp.json()
        total_m = 0.0
        for way in data.get("elements", []):
            pts = way.get("geometry", [])
            for i in range(len(pts) - 1):
                d = _haversine_km(
                    pts[i]["lon"], pts[i]["lat"],
                    np.array([pts[i+1]["lon"]]), np.array([pts[i+1]["lat"]]),
                )
                total_m += float(d[0])
        return total_m  # in km
    except Exception as e:
        logger.debug("Overpass road query failed: %s", e)
        return 0.0


def extract_osm_features_for_cluster(
    lon: float,
    lat: float,
    radius_amenity_m: int = 5000,
    radius_road_m: int = 5000,
    sleep_s: float = 1.0,
) -> dict:
    """
    Extract all OSM features for a single cluster point.
    Sleeps sleep_s seconds between queries to respect Overpass rate limits.
    """
    features = {
        "road_density_km_per_km2": 0.0,
        "dist_to_hospital_km":     999.0,
        "dist_to_school_km":       999.0,
        "dist_to_market_km":       999.0,
        "dist_to_bank_km":         999.0,
        "n_amenities_1km":         0,
    }

    # Road density
    road_km = _overpass_roads(lon, lat, radius_road_m)
    area_km2 = math.pi * (radius_road_m / 1000) ** 2
    features["road_density_km_per_km2"] = road_km / area_km2
    time.sleep(sleep_s)

    # Amenity distances
    all_pts_1km = []
    for amenity_key, amenity_filter in _AMENITY_QUERIES.items():
        pts = _overpass_amenities(lon, lat, radius_amenity_m, amenity_filter)
        if pts:
            lons = np.array([p[0] for p in pts])
            lats = np.array([p[1] for p in pts])
            dists = _haversine_km(lon, lat, lons, lats)
            features[f"dist_to_{amenity_key}_km"] = float(dists.min())
        time.sleep(sleep_s)

        # Count all amenities within 1km
        pts_1km = _overpass_amenities(lon, lat, 1000, amenity_filter)
        all_pts_1km.extend(pts_1km)

    features["n_amenities_1km"] = len(all_pts_1km)
    return features


def extract_all_clusters(
    survey_csv: str,
    output_csv: str = None,
    max_clusters: int = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Extract OSM features for all clusters in the survey CSV.

    Args:
        survey_csv:   input CSV with 'longitude', 'latitude' columns
        output_csv:   where to save the enriched CSV (default: overwrites input)
        max_clusters: limit for testing
        resume:       skip clusters already in the output CSV

    Returns:
        Updated DataFrame with OSM feature columns.
    """
    df = pd.read_csv(survey_csv)
    output_csv = output_csv or survey_csv

    osm_cols = [
        "road_density_km_per_km2", "dist_to_hospital_km",
        "dist_to_school_km", "dist_to_market_km",
        "dist_to_bank_km", "n_amenities_1km",
    ]

    # Resume: only compute for rows without features yet
    if resume and all(c in df.columns for c in osm_cols):
        missing_mask = df["dist_to_hospital_km"].isna() | (df["dist_to_hospital_km"] == 999.0)
        indices = df[missing_mask].index.tolist()
    else:
        for col in osm_cols:
            if col not in df.columns:
                df[col] = np.nan
        indices = df.index.tolist()

    if max_clusters:
        indices = indices[:max_clusters]

    logger.info("Extracting OSM features for %d clusters…", len(indices))

    for i, idx in enumerate(indices):
        row = df.loc[idx]
        logger.info(
            "[%d/%d] (%.4f, %.4f)", i + 1, len(indices),
            row["longitude"], row["latitude"],
        )
        feats = extract_osm_features_for_cluster(row["longitude"], row["latitude"])
        for col, val in feats.items():
            df.at[idx, col] = val

        # Save incrementally so we don't lose progress
        if (i + 1) % 10 == 0:
            df.to_csv(output_csv, index=False)
            logger.info("  Checkpoint saved → %s", output_csv)

    df.to_csv(output_csv, index=False)
    logger.info("OSM features saved → %s", output_csv)
    return df


def main():
    cfg = load_config()
    survey_csv = cfg["data"]["training_csv"]
    max_c      = cfg.get("osm", {}).get("max_clusters", None)

    if not os.path.exists(survey_csv):
        logger.warning("Training CSV not found. Run create_training_dataset.py first.")
        return

    extract_all_clusters(survey_csv, max_clusters=max_c)


if __name__ == "__main__":
    main()
