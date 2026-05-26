"""
Phase 5 — Training loop for TabularFusionModel (CNN + OSM/building features).

Run:
    python -m training.tabular_fusion_trainer
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.tabular_fusion_model import TabularFusionModel, OSM_FEATURE_COLS
from training.multi_sensor_dataset import _load_patch
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


class TabularPovertyDataset(Dataset):
    """
    Yields (s2_patch_tensor, tabular_tensor, label) per cluster.
    Missing tabular values are imputed with column medians.
    """

    def __init__(self, csv_path, sensors=None, train=True, transform=None):
        cfg = load_config()
        split       = cfg["training"]["split"]
        self.patch_size = cfg["training"]["patch_size"]
        self.s2_norm    = cfg["sentinel2"]["normalization_factor"]
        self.transform  = transform

        df = pd.read_csv(csv_path)
        if "s2_patch_file" not in df.columns and "patch_file" in df.columns:
            df["s2_patch_file"] = df["patch_file"]

        # Impute missing tabular features with medians
        for col in OSM_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median = df[col].median()
            df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)
            # Distance features capped at 50 km for sanity
            if col.startswith("dist_to"):
                df[col] = df[col].clip(upper=50.0)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * split)
        self.data = df.iloc[:split_idx] if train else df.iloc[split_idx:]

        logger.info(
            "TabularPovertyDataset | split=%s  n=%d  tabular_cols=%d",
            "train" if train else "val", len(self.data), len(OSM_FEATURE_COLS),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img = _load_patch(
            row.get("s2_patch_file", ""), 3, self.s2_norm, self.patch_size
        )
        if self.transform:
            img = self.transform(img)

        tabular = torch.tensor(
            row[OSM_FEATURE_COLS].values.astype(np.float32),
            dtype=torch.float32,
        )
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return img, tabular, label


def run_tabular_fusion_training(
    csv_path=None,
    run_name="tabular_fusion",
    epochs=None,
    checkpoint_dir=None,
):
    cfg = load_config()
    tcfg = cfg["training"]
    epochs    = epochs or tcfg["epochs"]
    csv_path  = csv_path or cfg["data"]["training_csv"]
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

    train_ds = TabularPovertyDataset(csv_path, train=True,  transform=aug)
    val_ds   = TabularPovertyDataset(csv_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)

    model = TabularFusionModel(in_channels=3, dropout_p=dropout_p).to(device)

    # Load NTL-pretrained backbone if available
    pretrained_ckpt = cfg.get("pretrain_ntl", {}).get("checkpoint")
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        model.load_pretrained_backbone(pretrained_ckpt)
        logger.info("Loaded NTL-pretrained backbone → tabular fusion model")

    optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_ckpt = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_ckpt = os.path.join(checkpoint_dir, f"{run_name}.pth")

    logger.info("▶  TabularFusion | device=%s", device)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, tabular, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            imgs, tabular, labels = imgs.to(device), tabular.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, tabular), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, tabular, labels in val_loader:
                imgs, tabular, labels = imgs.to(device), tabular.to(device), labels.to(device)
                val_running += criterion(model(imgs, tabular), labels).item() * imgs.size(0)
        val_loss = val_running / len(val_ds)

        scheduler.step(val_loss)
        logger.info("Epoch %d/%d  train=%.4f  val=%.4f", epoch+1, epochs,
                    running / len(train_ds), val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_ckpt:
                torch.save(model.state_dict(), best_ckpt)

    # Final eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, tabular, labels in val_loader:
            all_preds.extend(model(imgs.to(device), tabular.to(device)).cpu().numpy())
            all_labels.extend(labels.numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    r2   = r2_score(labels, preds)
    rmse = math.sqrt(mean_squared_error(labels, preds))
    mae  = mean_absolute_error(labels, preds)

    logger.info("✔  TabularFusion | R²=%.4f  RMSE=%.4f  MAE=%.4f", r2, rmse, mae)
    return {
        "run_name":      run_name,
        "model_type":    "tabular_fusion",
        "r2":            round(r2, 4),
        "rmse":          round(rmse, 4),
        "mae":           round(mae, 4),
        "best_val_loss": round(best_val_loss, 6),
    }


if __name__ == "__main__":
    cfg = load_config()
    run_tabular_fusion_training(checkpoint_dir=cfg["training"]["model_dir"])
