"""
Training loop for MultiStreamResNet — one encoder per sensor, features fused
in an MLP head.  API mirrors trainer.run_training so experiments/compare_satellites.py
can call either interchangeably.

Run:
    python -m training.multistream_trainer
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.multi_stream_model import MultiStreamResNet
from training.multi_sensor_dataset import MultiSensorDataset, sensor_channels
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def run_multistream_training(
    sensors,
    run_name,
    epochs=None,
    csv_path=None,
    checkpoint_dir=None,
):
    """
    Train a MultiStreamResNet for the given sensor combination.

    Returns the same dict schema as trainer.run_training so comparison
    scripts can aggregate results uniformly.
    """
    cfg = load_config()
    tcfg = cfg["training"]
    epochs = epochs or tcfg["epochs"]
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    csv_path = csv_path or cfg["data"]["training_csv"]
    dropout_p = cfg.get("model", {}).get("dropout_p", 0.3)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])

    train_ds = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=True,  transform=aug)
    val_ds   = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiStreamResNet(sensors=sensors, dropout_p=dropout_p).to(device)

    # Load NTL-pretrained S2 encoder if available
    pretrained_ckpt = cfg.get("pretrain_ntl", {}).get("checkpoint")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt) and "s2" in sensors:
        model.encoders["s2"].load_state_dict(
            torch.load(pretrained_ckpt, map_location="cpu"), strict=False
        )
        logger.info("Loaded NTL-pretrained S2 encoder from %s", pretrained_ckpt)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_ckpt = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_ckpt = os.path.join(checkpoint_dir, f"multistream_{run_name.replace(' ', '_')}.pth")

    logger.info(
        "▶  MultiStream %s | sensors=%s  device=%s", run_name, sensors, device
    )

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{run_name} epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_running += criterion(model(imgs), labels).item() * imgs.size(0)
        val_loss = val_running / len(val_ds)

        scheduler.step(val_loss)
        logger.info(
            "MultiStream %s  epoch %d/%d  train=%.4f  val=%.4f",
            run_name, epoch + 1, epochs,
            running / len(train_ds), val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_ckpt:
                torch.save(model.state_dict(), best_ckpt)

    # Final evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            all_preds.extend(model(imgs.to(device)).cpu().numpy())
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    r2   = r2_score(labels, preds)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    mae  = mean_absolute_error(labels, preds)

    logger.info("✔  MultiStream %s | R²=%.4f  RMSE=%.4f  MAE=%.4f", run_name, r2, rmse, mae)
    return {
        "run_name":      f"multistream_{run_name}",
        "sensors":       "+".join(sensors),
        "n_channels":    sensor_channels(sensors),
        "model_type":    "multi_stream",
        "r2":            round(r2,   4),
        "rmse":          round(rmse, 4),
        "mae":           round(mae,  4),
        "best_val_loss": round(best_val_loss, 6),
    }


if __name__ == "__main__":
    cfg = load_config()
    run_multistream_training(
        sensors=["s2", "s1", "viirs"],
        run_name="s2+s1+viirs",
        checkpoint_dir=cfg["training"]["model_dir"],
    )
