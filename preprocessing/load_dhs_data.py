import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
import numpy as np
import os

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    cfg = load_config()
    data_path = cfg["data"]["survey_path"]
    bbox = cfg["sentinel2"]["bbox"]
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    if not os.path.exists(data_path):
        logger.warning("%s not found. Generating synthetic DHS data for prototype use.", data_path)
        np.random.seed(42)
        num_samples = 50
        df_synth = pd.DataFrame({
            "longitude": np.random.uniform(bbox[0], bbox[2], num_samples),
            "latitude": np.random.uniform(bbox[1], bbox[3], num_samples),
            "wealth_index": np.random.uniform(-2, 2, num_samples),
        })
        df_synth.to_csv(data_path, index=False)
        logger.info("Synthetic data written to %s", data_path)

    df = pd.read_csv(data_path)
    required = {"longitude", "latitude", "wealth_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Survey CSV missing columns: {missing}")

    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    logger.info(
        "Loaded %d survey points | lat [%.3f, %.3f] | lon [%.3f, %.3f]",
        len(gdf), gdf["latitude"].min(), gdf["latitude"].max(),
        gdf["longitude"].min(), gdf["longitude"].max(),
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
