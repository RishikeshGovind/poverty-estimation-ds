"""
Reusable training loop.  Used by both train.py (single run) and
experiments/compare_satellites.py (multi-run comparison).
"""

import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from training.multi_sensor_dataset import MultiSensorDataset, sensor_channels
from models.resnet_model import ResNetRegression
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def run_training(sensors, run_name, epochs=None, csv_path=None, checkpoint_dir=None):
    """
    Train a ResNet regression model for the given sensor combination.

    Args:
        sensors:        list of sensor keys, e.g. ["s2", "s1", "viirs"]
        run_name:       string label used in logs and checkpoint filename
        epochs:         override training.epochs from config
        csv_path:       override data.training_csv from config
        checkpoint_dir: directory to save best model; None = don't save

    Returns:
        dict with keys: run_name, sensors, n_channels, r2, rmse, mae, best_val_loss
    """
    cfg = load_config()
    tcfg = cfg["training"]
    epochs = epochs or tcfg["epochs"]
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    csv_path = csv_path or cfg["data"]["training_csv"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    train_ds = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=True, transform=aug)
    val_ds   = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    n_channels = sensor_channels(sensors)
    dropout_p = cfg.get("model", {}).get("dropout_p", 0.0)
    model = ResNetRegression(in_channels=n_channels, dropout_p=dropout_p).to(device)

    # --- Stage 2: load pretrained backbone if available ---
    pretrained_ckpt = cfg.get("pretrain_ntl", {}).get("checkpoint")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt) and "s2" in sensors:
        pretrained_sd = torch.load(pretrained_ckpt, map_location="cpu")
        model_sd = model.backbone.state_dict()
        # Skip layers where shapes differ (e.g. conv1 when n_channels > 3)
        compatible = {k: v for k, v in pretrained_sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        model.backbone.load_state_dict(compatible, strict=False)
        logger.info("Loaded NTL-pretrained backbone from %s (%d/%d layers)",
                    pretrained_ckpt, len(compatible), len(pretrained_sd))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_ckpt = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_ckpt = os.path.join(checkpoint_dir, f"{run_name.replace(' ', '_')}.pth")

    logger.info("▶  %s | sensors=%s  channels=%d  device=%s", run_name, sensors, n_channels, device)

    for epoch in range(epochs):
        # --- train ---
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{run_name} epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_ds)

        # --- validate ---
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_running += criterion(model(imgs), labels).item() * imgs.size(0)
        val_loss = val_running / len(val_ds)

        scheduler.step(val_loss)
        logger.info(
            "%s  epoch %d/%d  train=%.4f  val=%.4f  rmse=%.4f",
            run_name, epoch + 1, epochs, train_loss, val_loss, math.sqrt(val_loss),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_ckpt:
                torch.save(model.state_dict(), best_ckpt)

    # --- final evaluation on val set ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            all_preds.extend(model(imgs).cpu().numpy())
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    r2   = r2_score(labels, preds)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    mae  = mean_absolute_error(labels, preds)

    logger.info("✔  %s | R²=%.4f  RMSE=%.4f  MAE=%.4f", run_name, r2, rmse, mae)
    return {
        "run_name":      run_name,
        "sensors":       "+".join(sensors),
        "n_channels":    n_channels,
        "r2":            round(r2,   4),
        "rmse":          round(rmse, 4),
        "mae":           round(mae,  4),
        "best_val_loss": round(best_val_loss, 6),
    }
