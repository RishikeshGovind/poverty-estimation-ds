"""
Generate predictions.geojson for the GitHub Pages frontend.

Source priority:
  1. data/predictions.csv  (from evaluate_model.py — has 'prediction' column)
  2. data/training_dataset.csv  (raw labels used as ground truth)
  3. Synthetic demo data generated from config bbox

The output is written to scoring.output_geojson (docs/data/predictions.geojson).

Usage:
    python -m scoring.generate_geojson
"""

import os
import json
import numpy as np
import pandas as pd

from scoring.sdg_scorer import SDGScorer
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _synthetic_demo(bbox, n=60) -> pd.DataFrame:
    """Generate plausible demo data when no real predictions exist."""
    np.random.seed(42)
    lons = np.random.uniform(bbox[0], bbox[2], n)
    lats = np.random.uniform(bbox[1], bbox[3], n)
    wealth = np.random.uniform(-2, 2, n)
    ntl = np.clip(np.random.exponential(0.08, n), 0, 1)
    buildup = np.clip(np.random.exponential(0.1, n), 0, 1)
    return pd.DataFrame({
        "longitude":    lons,
        "latitude":     lats,
        "prediction":   wealth,
        "label":        wealth,
        "sdg7_ntl":     ntl,
        "sdg11_buildup": buildup,
        "country":      "Kenya (demo)",
        "uncertainty":  np.random.uniform(0.05, 0.3, n),
    })


def _load_predictions(cfg) -> pd.DataFrame:
    pred_path  = cfg["data"]["predictions_csv"]
    train_path = cfg["data"]["training_csv"]

    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        logger.info("Loaded predictions from %s (%d rows)", pred_path, len(df))
        return df

    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        df["prediction"] = df["label"]   # use ground truth as prediction
        logger.info("No predictions file; using training labels from %s (%d rows)", train_path, len(df))
        return df

    logger.warning("No predictions or training CSV found — generating synthetic demo data.")
    return _synthetic_demo(cfg["sentinel2"]["bbox"])


def main():
    cfg = load_config()
    out_path = cfg["scoring"]["output_geojson"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = _load_predictions(cfg)
    scorer = SDGScorer()

    features = []
    for _, row in df.iterrows():
        scores = scorer.score_row(row.to_dict())

        props = {
            "wealth_index":    round(float(row.get("prediction", row.get("label", 0))), 3),
            "country":         str(row.get("country", "Unknown")),
            "uncertainty":     round(float(row["uncertainty"]), 3) if "uncertainty" in row and not np.isnan(float(row.get("uncertainty", float("nan")))) else None,
            **scores,
        }
        # Include raw task values when available
        for col in ["sdg7_ntl", "sdg11_buildup"]:
            if col in row and not _isnan(row[col]):
                props[col] = round(float(row[col]), 4)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": props,
        })

    geojson = {"type": "FeatureCollection", "features": features}

    with open(out_path, "w") as f:
        json.dump(geojson, f, allow_nan=False)

    logger.info("GeoJSON written: %s  (%d features)", out_path, len(features))

    # Summary stats
    scores = [f["properties"]["composite_score"] for f in features]
    logger.info(
        "Composite score | mean=%.1f  min=%.1f  max=%.1f",
        np.mean(scores), np.min(scores), np.max(scores),
    )


def _isnan(v) -> bool:
    try:
        return np.isnan(float(v))
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    main()
