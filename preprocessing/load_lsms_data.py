"""
Phase 1 — World Bank LSMS-ISA data loader.

LSMS-ISA (Living Standards Measurement Study — Integrated Surveys on Agriculture)
covers Ethiopia, Nigeria, Tanzania, Uganda, Malawi, Niger, Burkina Faso.
Unlike DHS, it is OPEN ACCESS — no approval required.

Download from: https://www.worldbank.org/en/programs/lsms/initiatives/lsms-ISA
  → Select a country → Download the household panel (Stata .dta files)

Key files needed per country:
  - Household consumption/expenditure file (contains cluster ID + welfare aggregate)
  - GPS coordinates file (contains cluster ID + lat/lon)

Run:
    python -m preprocessing.load_lsms_data
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# Column name variants across LSMS waves and countries
# The loader tries each alias in order and takes the first match.
_ID_ALIASES      = ["ea_id", "cluster_id", "cluster", "ea", "hhid", "hh_id"]
_LAT_ALIASES     = ["lat_dd_mod", "lat", "latitude",  "gps_lat",  "y_gps_village"]
_LON_ALIASES     = ["lon_dd_mod", "lon", "longitude", "gps_lon",  "x_gps_village"]
_WELFARE_ALIASES = [
    "total_cons_ann",  # Ethiopia ESS
    "totcons",         # Nigeria GHS
    "expmR",           # Tanzania NPS
    "cons_W3",         # Malawi IHS
    "pcexp",           # Uganda UNPS
    "welfare",         # generic
    "hh_consumption",
]


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
        # Case-insensitive fallback
        for c in df.columns:
            if c.lower() == a.lower():
                return c
    return None


def load_lsms_country(
    household_path: str,
    gps_path: str,
    country: str = "unknown",
    log_transform: bool = True,
) -> pd.DataFrame:
    """
    Load one LSMS country wave and return a DataFrame with columns:
        latitude, longitude, label, country

    Args:
        household_path: Stata .dta (or CSV) with cluster ID + consumption/welfare
        gps_path:       Stata .dta / CSV / shapefile with cluster ID + coordinates
        country:        country name string for the 'country' column
        log_transform:  apply log1p to consumption expenditure (recommended)

    Returns:
        DataFrame with ['latitude', 'longitude', 'label', 'country']
    """
    # ── Load GPS ──────────────────────────────────────────────────────────────
    if gps_path.endswith((".shp", ".gpkg")):
        gps_df = gpd.read_file(gps_path)
        gps_df = pd.DataFrame(gps_df.drop(columns="geometry"))
    elif gps_path.endswith(".dta"):
        gps_df = pd.read_stata(gps_path)
    else:
        gps_df = pd.read_csv(gps_path)

    lat_col = _find_col(gps_df, _LAT_ALIASES)
    lon_col = _find_col(gps_df, _LON_ALIASES)
    id_col_gps = _find_col(gps_df, _ID_ALIASES)

    if not all([lat_col, lon_col, id_col_gps]):
        raise ValueError(
            f"GPS file {gps_path} missing lat/lon/id columns.\n"
            f"Found: {list(gps_df.columns)}"
        )

    gps_df = gps_df.rename(columns={
        lat_col: "latitude", lon_col: "longitude", id_col_gps: "_cluster_id"
    })[["_cluster_id", "latitude", "longitude"]]
    # Drop junk coordinates
    gps_df = gps_df[(gps_df["latitude"].abs() > 0.001) | (gps_df["longitude"].abs() > 0.001)]

    # ── Load household consumption ────────────────────────────────────────────
    if household_path.endswith(".dta"):
        hh_df = pd.read_stata(household_path)
    else:
        hh_df = pd.read_csv(household_path)

    welfare_col = _find_col(hh_df, _WELFARE_ALIASES)
    id_col_hh   = _find_col(hh_df, _ID_ALIASES)

    if not welfare_col or not id_col_hh:
        raise ValueError(
            f"Household file {household_path} missing welfare/id columns.\n"
            f"Found: {list(hh_df.columns)}"
        )

    hh_df = hh_df.rename(columns={welfare_col: "_welfare", id_col_hh: "_cluster_id"})
    # Drop invalid values
    hh_df = hh_df[hh_df["_welfare"] > 0].dropna(subset=["_welfare"])

    # Aggregate to cluster level (mean consumption per cluster)
    cluster_welfare = (
        hh_df.groupby("_cluster_id")["_welfare"].mean().reset_index()
    )

    if log_transform:
        cluster_welfare["_welfare"] = np.log1p(cluster_welfare["_welfare"])

    # Standardise to roughly [-2, +2] like DHS wealth index
    mu = cluster_welfare["_welfare"].mean()
    sd = cluster_welfare["_welfare"].std()
    cluster_welfare["label"] = (cluster_welfare["_welfare"] - mu) / (sd + 1e-8) * 0.85

    # ── Merge ─────────────────────────────────────────────────────────────────
    merged = gps_df.merge(cluster_welfare[["_cluster_id", "label"]], on="_cluster_id", how="inner")
    merged["country"] = country
    result = merged[["latitude", "longitude", "label", "country"]]

    logger.info(
        "LSMS %s: %d clusters | label min=%.2f max=%.2f mean=%.2f",
        country, len(result),
        result["label"].min(), result["label"].max(), result["label"].mean(),
    )
    return result


def load_all_lsms(lsms_cfg: dict) -> pd.DataFrame:
    """
    Load multiple LSMS country configs and concatenate.

    lsms_cfg is the 'lsms' section from config.yaml, e.g.:
        countries:
          Ethiopia:
            household_path: data/lsms/eth_household.dta
            gps_path:       data/lsms/eth_gps.dta
          Nigeria:
            household_path: ...
    """
    frames = []
    for country, paths in lsms_cfg.get("countries", {}).items():
        hh_path  = paths.get("household_path", "")
        gps_path = paths.get("gps_path", "")
        if not hh_path or not gps_path:
            logger.warning("LSMS %s: paths not set in config, skipping.", country)
            continue
        if not os.path.exists(hh_path) or not os.path.exists(gps_path):
            logger.warning("LSMS %s: files not found, skipping.", country)
            continue
        frames.append(load_lsms_country(hh_path, gps_path, country=country))

    if not frames:
        logger.warning("No LSMS data loaded. Check lsms.countries paths in config.yaml.")
        return pd.DataFrame(columns=["latitude", "longitude", "label", "country"])

    df = pd.concat(frames, ignore_index=True)
    logger.info("LSMS total: %d clusters across %d countries", len(df), len(frames))
    return df


def main():
    cfg = load_config()
    lsms_cfg = cfg.get("lsms", {})
    output_path = lsms_cfg.get("output_csv", "data/lsms_survey.csv")

    df = load_all_lsms(lsms_cfg)
    if df.empty:
        logger.warning(
            "No data loaded. Download LSMS files from "
            "https://www.worldbank.org/en/programs/lsms/initiatives/lsms-ISA "
            "and set paths in config.yaml under the 'lsms' section."
        )
        return

    os.makedirs(os.path.dirname(output_path) or "data", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("LSMS survey data saved → %s", output_path)


if __name__ == "__main__":
    main()
