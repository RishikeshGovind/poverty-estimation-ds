"""
Multi-task training loop for MultiTaskResNet.

Each task has its own MSE loss, weighted by tasks.weights in config.yaml.
NaN labels (e.g. sdg7_ntl when VIIRS patches are absent) are masked out of
the loss so they don't corrupt gradients.
"""

import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from models.multitask_model import MultiTaskResNet
from training.multi_sensor_dataset import MultiSensorDataset, sensor_channels
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE ignoring NaN targets."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return ((pred[mask] - target[mask]) ** 2).mean()


class _MultiTaskDataset(MultiSensorDataset):
    """Extends MultiSensorDataset to return a tasks tensor instead of a scalar."""

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self._tasks = tasks

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        row = self.data.iloc[idx]
        label_vec = torch.tensor(
            [float(row.get(t, float("nan"))) for t in self._tasks],
            dtype=torch.float32,
        )
        return img, label_vec


def run_multitask_training(
    sensors=None,
    tasks=None,
    run_name="multitask",
    epochs=None,
    csv_path=None,
    checkpoint_path=None,
    train_indices=None,
    val_indices=None,
):
    """
    Train MultiTaskResNet.

    Args:
        sensors:          sensor list (default from MultiSensorDataset)
        tasks:            task name list (default from config tasks.names)
        train_indices:    explicit row indices for training (spatial CV use)
        val_indices:      explicit row indices for validation (spatial CV use)

    Returns:
        dict of per-task {r2, rmse, mae} plus best_val_loss
    """
    cfg = load_config()
    tcfg = cfg["training"]
    task_weights = cfg["tasks"]["weights"]
    epochs = epochs or tcfg["epochs"]
    batch_size = tcfg["batch_size"]
    lr = tcfg["lr"]
    csv_path = csv_path or cfg["data"]["training_csv"]

    if tasks is None:
        tasks = cfg["tasks"]["names"]
    if sensors is None:
        sensors = ["s2"]

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])

    full_train_ds = _MultiTaskDataset(tasks=tasks, sensors=sensors, csv_path=csv_path,
                                      train=True,  transform=aug)
    full_val_ds   = _MultiTaskDataset(tasks=tasks, sensors=sensors, csv_path=csv_path,
                                      train=False)

    # Spatial CV override: replace internal random split with explicit indices
    if train_indices is not None and val_indices is not None:
        # Build a flat dataset (no split) and subset it
        full_ds = _MultiTaskDataset(tasks=tasks, sensors=sensors, csv_path=csv_path,
                                    train=True, transform=None)
        full_ds.data = full_ds.data.iloc[:len(full_ds.data)]  # keep all
        train_ds = Subset(full_ds, train_indices)
        # Re-build val without aug
        val_full = _MultiTaskDataset(tasks=tasks, sensors=sensors, csv_path=csv_path, train=True)
        val_ds = Subset(val_full, val_indices)
    else:
        train_ds, val_ds = full_train_ds, full_val_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    n_ch = sensor_channels(sensors)
    model = MultiTaskResNet(in_channels=n_ch, tasks=tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

    best_val_loss = float("inf")
    logger.info("▶ %s | tasks=%s  sensors=%s  channels=%d  device=%s",
                run_name, tasks, sensors, n_ch, device)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{run_name} {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = sum(
                task_weights.get(t, 1.0) * _masked_mse(preds[t], labels[:, i])
                for i, t in enumerate(tasks)
            )
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_ds)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                val_running += sum(
                    task_weights.get(t, 1.0) * _masked_mse(preds[t], labels[:, i]).item() * imgs.size(0)
                    for i, t in enumerate(tasks)
                )
        val_loss = val_running / len(val_ds)
        scheduler.step(val_loss)
        logger.info("%s  epoch %d/%d  train=%.4f  val=%.4f", run_name, epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

    # Per-task evaluation
    model.eval()
    task_preds  = {t: [] for t in tasks}
    task_labels = {t: [] for t in tasks}
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            for i, t in enumerate(tasks):
                lbl = labels[:, i].numpy()
                pred = preds[t].cpu().numpy()
                valid = ~np.isnan(lbl)
                task_preds[t].extend(pred[valid])
                task_labels[t].extend(lbl[valid])

    results = {"run_name": run_name, "best_val_loss": round(best_val_loss, 6)}
    for t in tasks:
        p, l = np.array(task_preds[t]), np.array(task_labels[t])
        if len(p) < 2:
            continue
        r2   = r2_score(l, p)
        rmse = math.sqrt(mean_squared_error(l, p))
        mae  = mean_absolute_error(l, p)
        results[f"{t}_r2"]   = round(r2,   4)
        results[f"{t}_rmse"] = round(rmse, 4)
        results[f"{t}_mae"]  = round(mae,  4)
        logger.info("  %-20s  R²=%.4f  RMSE=%.4f  MAE=%.4f", t, r2, rmse, mae)

    return results, model
