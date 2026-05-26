"""
DHS data loader.

Real data:
  Register at https://dhsprogram.com/data/dataset_admin/login_main.cfm (free).
  Request the GPS dataset (.zip containing a shapefile) and the Household Recode (.DTA).
  Set dhs.gps_path and dhs.hr_path in config.yaml.

  GPS file columns used : DHSCLUST, LATNUM, LONGNUM
  Household recode cols : hv001 (cluster ID), hv271 (wealth factor score × 100000)

Synthetic fallback:
  If dhs.gps_path is empty the script generates 50 synthetic points over the
  sentinel2.bbox region so the rest of the pipeline can run immediately.
"""

import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
import numpy as np
import os

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _load_real_dhs(gps_path: str, hr_path: str, wealth_scale: float) -> pd.DataFrame:
    """Merge DHS GPS clusters with household recode wealth index."""
    # --- GPS clusters ---
    if gps_path.endswith((".shp", ".dbf", ".gpkg")):
        gps_df = gpd.read_file(gps_path)[["DHSCLUST", "LATNUM", "LONGNUM"]]
    else:
        gps_df = pd.read_csv(gps_path)[["DHSCLUST", "LATNUM", "LONGNUM"]]

    missing = {"DHSCLUST", "LATNUM", "LONGNUM"} - set(gps_df.columns)
    if missing:
        raise ValueError(f"GPS file missing columns: {missing}")

    # Filter out dummy/missing coordinates (DHS uses 0,0 for urban displacement)
    gps_df = gps_df[(gps_df["LATNUM"] != 0) | (gps_df["LONGNUM"] != 0)]

    # --- Household recode ---
    if hr_path.endswith(".dta"):
        hr_df = pd.read_stata(hr_path, columns=["hv001", "hv271"])
    else:
        hr_df = pd.read_csv(hr_path)[["hv001", "hv271"]]

    missing_hr = {"hv001", "hv271"} - set(hr_df.columns)
    if missing_hr:
        raise ValueError(f"Household recode missing columns: {missing_hr}")

    # hv271 is stored as score * 100000 — rescale to roughly [-2, +2]
    hr_df["wealth_index"] = hr_df["hv271"] / wealth_scale

    cluster_wealth = (
        hr_df.groupby("hv001")["wealth_index"]
        .mean()
        .reset_index()
        .rename(columns={"hv001": "DHSCLUST"})
    )

    merged = gps_df.merge(cluster_wealth, on="DHSCLUST", how="inner")
    merged = merged.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"})
    logger.info(
        "Merged %d GPS clusters with household recode (%.0f%% match rate)",
        len(merged),
        100 * len(merged) / max(len(gps_df), 1),
    )
    return merged[["longitude", "latitude", "wealth_index"]]


def _generate_synthetic(bbox: list, n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "longitude": np.random.uniform(bbox[0], bbox[2], n),
        "latitude": np.random.uniform(bbox[1], bbox[3], n),
        "wealth_index": np.random.uniform(-2, 2, n),
    })


def main():
    cfg = load_config()
    data_path = cfg["data"]["survey_path"]
    bbox = cfg["sentinel2"]["bbox"]
    dhs_cfg = cfg["dhs"]
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    gps_path = dhs_cfg.get("gps_path", "")
    hr_path = dhs_cfg.get("hr_path", "")
    wealth_scale = dhs_cfg.get("wealth_scale_factor", 100000)

    if gps_path and hr_path:
        if not os.path.exists(gps_path):
            raise FileNotFoundError(f"DHS GPS file not found: {gps_path}")
        if not os.path.exists(hr_path):
            raise FileNotFoundError(f"DHS household recode not found: {hr_path}")
        logger.info("Loading real DHS data from %s + %s", gps_path, hr_path)
        df = _load_real_dhs(gps_path, hr_path, wealth_scale)
    else:
        logger.warning(
            "dhs.gps_path / dhs.hr_path not set in config.yaml. "
            "Generating synthetic data. Register at dhsprogram.com for real data."
        )
        df = _generate_synthetic(bbox)

    df.to_csv(data_path, index=False)
    logger.info("Survey data written to %s (%d points)", data_path, len(df))

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    logger.info(
        "Wealth index | min=%.3f  max=%.3f  mean=%.3f",
        df["wealth_index"].min(), df["wealth_index"].max(), df["wealth_index"].mean(),
    )

    center = [gdf["latitude"].mean(), gdf["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=12)
    for _, row in gdf.iterrows():
        color = "green" if row["wealth_index"] > 0 else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5, color=color, fill=True, fill_opacity=0.7,
            popup=f"Wealth Index: {row['wealth_index']:.2f}",
        ).add_to(m)

    output_map = "outputs/maps/dhs_survey_map.html"
    m.save(output_map)
    logger.info("Interactive map saved to %s", output_map)


if __name__ == "__main__":
    main()
