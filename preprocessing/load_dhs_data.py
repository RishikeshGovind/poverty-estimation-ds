"""
DHS data loader — multi-country.

Reads GPS shapefiles (DHSCLUST, LATNUM, LONGNUM) and Household Recode .DTA files
(hv001 cluster ID, hv271 wealth factor score × 100000) for each country defined
in config.yaml under dhs.countries, merges them, and writes a combined CSV to
data.survey_path.

Falls back to 50 synthetic points if no countries are configured.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _load_one_country(name: str, gps_path: str, hr_path: str, wealth_scale: float) -> pd.DataFrame:
    extra_cols = ["ADM1NAME", "DHSREGNA", "URBAN_RURA"]
    if gps_path.endswith((".shp", ".dbf", ".gpkg")):
        all_cols = ["DHSCLUST", "LATNUM", "LONGNUM"] + extra_cols
        avail = [c for c in all_cols if c in gpd.read_file(gps_path, rows=1).columns]
        gps_df = gpd.read_file(gps_path)[avail]
    else:
        all_cols = ["DHSCLUST", "LATNUM", "LONGNUM"] + extra_cols
        raw = pd.read_csv(gps_path)
        gps_df = raw[[c for c in all_cols if c in raw.columns]]

    # Drop dummy coordinates (DHS displaces urban clusters — 0,0 = missing)
    gps_df = gps_df[(gps_df["LATNUM"] != 0) | (gps_df["LONGNUM"] != 0)]

    if hr_path.endswith(".dta") or hr_path.endswith(".DTA"):
        hr_df = pd.read_stata(hr_path, columns=["hv001", "hv271"], convert_categoricals=False)
    else:
        hr_df = pd.read_csv(hr_path)[["hv001", "hv271"]]

    hr_df["wealth_index"] = hr_df["hv271"] / wealth_scale

    cluster_wealth = (
        hr_df.groupby("hv001")["wealth_index"]
        .mean()
        .reset_index()
        .rename(columns={"hv001": "DHSCLUST"})
    )

    merged = gps_df.merge(cluster_wealth, on="DHSCLUST", how="inner")
    merged = merged.rename(columns={"LONGNUM": "longitude", "LATNUM": "latitude"})
    merged["country"] = name

    match_rate = 100 * len(merged) / max(len(gps_df), 1)
    logger.info("%s: %d clusters (%.0f%% GPS match) | wealth [%.3f, %.3f]",
                name, len(merged), match_rate,
                merged["wealth_index"].min(), merged["wealth_index"].max())

    out_cols = ["country", "longitude", "latitude", "wealth_index"]
    for col in ["ADM1NAME", "DHSREGNA", "URBAN_RURA"]:
        if col in merged.columns:
            out_cols.append(col)
    return merged[out_cols]


def _generate_synthetic(bbox: list, n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "country": "synthetic",
        "longitude": np.random.uniform(bbox[0], bbox[2], n),
        "latitude":  np.random.uniform(bbox[1], bbox[3], n),
        "wealth_index": np.random.uniform(-2, 2, n),
    })


def main():
    cfg = load_config()
    out_path = cfg["data"]["survey_path"]
    bbox = cfg["sentinel2"]["bbox"]
    dhs_cfg = cfg.get("dhs", {})
    wealth_scale = dhs_cfg.get("wealth_scale_factor", 100000)
    countries = dhs_cfg.get("countries", {})

    os.makedirs("data", exist_ok=True)

    if not countries:
        logger.warning("No dhs.countries configured — using synthetic data.")
        df = _generate_synthetic(bbox)
    else:
        frames = []
        for name, paths in countries.items():
            gps_path = paths.get("gps_path", "")
            hr_path  = paths.get("hr_path", "")
            if not gps_path or not hr_path:
                logger.warning("Skipping %s — paths not set.", name)
                continue
            if not os.path.exists(gps_path):
                raise FileNotFoundError(f"GPS file not found: {gps_path}")
            if not os.path.exists(hr_path):
                raise FileNotFoundError(f"HR file not found: {hr_path}")
            frames.append(_load_one_country(name, gps_path, hr_path, wealth_scale))

        if not frames:
            logger.warning("All countries skipped — falling back to synthetic data.")
            df = _generate_synthetic(bbox)
        else:
            df = pd.concat(frames, ignore_index=True)

    df.to_csv(out_path, index=False)
    logger.info("Wrote %d clusters to %s", len(df), out_path)
    logger.info("Countries: %s", df["country"].value_counts().to_dict())


if __name__ == "__main__":
    main()
