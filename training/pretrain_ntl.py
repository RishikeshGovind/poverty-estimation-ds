"""
Phase 1 of the sustainlab-style two-stage training:
  1. Pretrain ResNet18 to predict VIIRS nighttime-light values from S2 imagery.
     (NTL is a free, data-rich proxy for economic activity.)
  2. Save pretrained backbone weights to a checkpoint.
  3. Fine-tuning (Stage 2) happens in trainer.py via pretrained_backbone arg.

Run:
    python -m training.pretrain_ntl
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.resnet_model import ResNetNTLPretrain
from training.multi_sensor_dataset import _load_patch
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


class NTLDataset(Dataset):
    """
    Returns (s2_patch, ntl_value) pairs.
    ntl_value is the mean of the VIIRS patch (a scalar proxy for NTL).
    Rows with missing S2 or VIIRS patches are dropped.
    """

    def __init__(self, csv_path, train=True, transform=None):
        cfg = load_config()
        split = cfg["training"]["split"]
        self.patch_size = cfg["training"]["patch_size"]
        self.s2_norm = cfg["sentinel2"]["normalization_factor"]
        self.viirs_norm = cfg["viirs"]["normalization_clip"]
        self.transform = transform

        df = pd.read_csv(csv_path)
        if "s2_patch_file" not in df.columns and "patch_file" in df.columns:
            df["s2_patch_file"] = df["patch_file"]

        # Keep only rows that have both S2 and VIIRS patches
        has_s2 = df["s2_patch_file"].notna() & (df["s2_patch_file"] != "")
        has_ntl = (
            df["viirs_patch_file"].notna() & (df["viirs_patch_file"] != "")
            if "viirs_patch_file" in df.columns
            else pd.Series(False, index=df.index)
        )
        # Fall back to sdg7_ntl label column if no VIIRS patches
        has_ntl_label = (
            df["sdg7_ntl"].notna() if "sdg7_ntl" in df.columns
            else pd.Series(False, index=df.index)
        )
        df = df[has_s2 & (has_ntl | has_ntl_label)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                "NTLDataset: no rows with both S2 and VIIRS data found. "
                "Run preprocessing/compute_sdg_labels.py first."
            )

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * split)
        self.data = df.iloc[:split_idx] if train else df.iloc[split_idx:]
        logger.info(
            "NTLDataset | split=%s  n=%d", "train" if train else "val", len(self.data)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        s2 = _load_patch(
            row.get("s2_patch_file", ""), 3, self.s2_norm, self.patch_size
        )
        if self.transform:
            s2 = self.transform(s2)

        # NTL target: prefer raw VIIRS patch mean, fall back to sdg7_ntl label
        if "viirs_patch_file" in row and str(row["viirs_patch_file"]).strip():
            viirs = _load_patch(
                row["viirs_patch_file"], 1, self.viirs_norm, self.patch_size
            )
            ntl = float(viirs.mean())
        else:
            ntl = float(row["sdg7_ntl"])

        return s2, torch.tensor(ntl, dtype=torch.float32)


def pretrain(
    csv_path=None,
    epochs=None,
    lr=None,
    checkpoint_path=None,
):
    cfg = load_config()
    ptcfg = cfg.get("pretrain_ntl", {})
    csv_path = csv_path or cfg["data"]["training_csv"]
    epochs = epochs or ptcfg.get("epochs", 10)
    lr = lr or ptcfg.get("lr", 1e-4)
    checkpoint_path = checkpoint_path or ptcfg.get(
        "checkpoint", "outputs/models/pretrain_ntl.pth"
    )

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

    train_ds = NTLDataset(csv_path, train=True,  transform=aug)
    val_ds   = NTLDataset(csv_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)

    model = ResNetNTLPretrain(in_channels=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, ntl in tqdm(train_loader, desc=f"NTL pretrain {epoch+1}/{epochs}", leave=False):
            imgs, ntl = imgs.to(device), ntl.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), ntl)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)

        model.eval()
        val_running, preds_all, labels_all = 0.0, [], []
        with torch.no_grad():
            for imgs, ntl in val_loader:
                imgs, ntl = imgs.to(device), ntl.to(device)
                out = model(imgs)
                val_running += criterion(out, ntl).item() * imgs.size(0)
                preds_all.extend(out.cpu().numpy())
                labels_all.extend(ntl.cpu().numpy())

        val_loss = val_running / len(val_ds)
        scheduler.step(val_loss)
        r2 = r2_score(labels_all, preds_all)
        logger.info(
            "NTL pretrain epoch %d/%d  val_loss=%.4f  R²=%.4f",
            epoch + 1, epochs, val_loss, r2,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.backbone.state_dict(), checkpoint_path)
            logger.info("  Saved backbone → %s", checkpoint_path)

    logger.info("NTL pretraining complete. Backbone saved to %s", checkpoint_path)
    return checkpoint_path


if __name__ == "__main__":
    pretrain()
