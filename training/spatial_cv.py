"""
Spatial cross-validation utilities.

Assigns a country label to each survey point using Natural Earth boundaries,
then provides leave-one-country-out split generators.

If country assignment fails (e.g. geopandas naturalearth data unavailable),
falls back to KMeans spatial clustering — producing k pseudo-country folds
that still enforce spatial separation between train and test sets.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def assign_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'country' column by spatially joining lat/lon with Natural Earth polygons.
    Falls back to KMeans cluster IDs if geopandas data is unavailable.
    """
    if "country" in df.columns:
        logger.info("'country' column already present — skipping assignment.")
        return df

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        # naturalearth_lowres ships with geopandas (deprecated in v1 but still present)
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except AttributeError:
            # geopandas >= 1.0 removed datasets; use geodatasets if available
            from geodatasets import get_path
            world = gpd.read_file(get_path("naturalearth.land"))

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])],
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(gdf, world[["name", "geometry"]], how="left", predicate="within")
        df = df.copy()
        df["country"] = joined["name"].fillna("Unknown").values
        counts = df["country"].value_counts()
        logger.info("Country assignment:\n%s", counts.to_string())

    except Exception as e:
        logger.warning("Country assignment via geopandas failed (%s). Using KMeans spatial clusters.", e)
        df = _kmeans_spatial_split(df)

    return df


def _kmeans_spatial_split(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    coords = df[["latitude", "longitude"]].values
    labels = KMeans(n_clusters=min(k, len(df)), random_state=42, n_init=10).fit_predict(coords)
    df = df.copy()
    df["country"] = [f"cluster_{i}" for i in labels]
    logger.info("KMeans spatial clusters: %s", pd.Series(df["country"]).value_counts().to_dict())
    return df


def leave_one_country_out(df: pd.DataFrame):
    """
    Yield (held_out_country, train_indices, test_indices) for each country.
    Skips countries with fewer than min_test_samples points.
    """
    cfg = load_config()
    col = cfg["spatial_cv"]["country_col"]
    min_test = cfg["spatial_cv"]["min_test_samples"]

    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Run assign_countries() first.")

    countries = df[col].dropna().unique()
    logger.info("Spatial CV: %d countries/folds", len(countries))

    for country in sorted(countries):
        test_mask  = df[col] == country
        train_mask = ~test_mask

        n_test  = test_mask.sum()
        n_train = train_mask.sum()

        if n_test < min_test:
            logger.debug("Skipping %s — only %d test samples (min=%d).", country, n_test, min_test)
            continue
        if n_train == 0:
            logger.warning("Skipping %s — no training samples left.", country)
            continue

        logger.info("Fold: hold-out=%s  train=%d  test=%d", country, n_train, n_test)
        yield country, df[train_mask].index.tolist(), df[test_mask].index.tolist()
