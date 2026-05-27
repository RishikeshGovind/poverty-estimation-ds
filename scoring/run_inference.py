"""
Run inference with the best trained model on all clusters and write
data/predictions.csv + docs/data/predictions.geojson.

Usage:
    python -m scoring.run_inference
    python -m scoring.run_inference --sensors s2 viirs --csv data/training_dataset.csv
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from training.multi_sensor_dataset import MultiSensorDataset, sensor_channels
from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", nargs="+", default=["s2", "viirs"])
    parser.add_argument("--csv",     default="data/training_dataset.csv")
    parser.add_argument("--ckpt",    default=None)
    args = parser.parse_args()

    cfg = load_config()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load all data (split=1.0 → no val holdout)
    ds = MultiSensorDataset(sensors=args.sensors, csv_path=args.csv,
                            train=True, split=1.0)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    # Load model
    n_ch = sensor_channels(args.sensors)
    dropout_p = cfg.get("model", {}).get("dropout_p", 0.0)
    model = ResNetRegression(in_channels=n_ch, dropout_p=dropout_p).to(device)

    ckpt = args.ckpt or cfg["evaluation"]["checkpoint"]
    if not os.path.exists(ckpt):
        logger.error("Checkpoint not found: %s", ckpt)
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info("Loaded checkpoint: %s", ckpt)

    # Inference
    all_preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            preds = model(imgs.to(device)).cpu().numpy().flatten()
            all_preds.extend(preds.tolist())

    # Build output CSV
    df = ds.data.copy().reset_index(drop=True)
    df["prediction"] = all_preds

    pred_path = cfg["data"]["predictions_csv"]
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    df.to_csv(pred_path, index=False)
    logger.info("Predictions saved: %s (%d rows)", pred_path, len(df))

    from sklearn.metrics import r2_score, mean_squared_error
    r2   = r2_score(df["label"], df["prediction"])
    rmse = np.sqrt(mean_squared_error(df["label"], df["prediction"]))
    logger.info("Full-dataset R²=%.4f  RMSE=%.4f", r2, rmse)

    # Generate GeoJSON
    from scoring.generate_geojson import main as gen_geojson
    gen_geojson()


if __name__ == "__main__":
    main()
