"""
Ridge regression on CNN features (sustainlab-group technique).

Why this works: ResNet features are richer than raw pixels but DHS labels are
scarce (~hundreds of clusters per country).  A linear Ridge on 512-d features
generalises better than end-to-end fine-tuning with so few labels, especially
at test time on unseen countries.

Pipeline:
  1. Load a trained ResNet (or MultiStreamResNet) checkpoint.
  2. Strip the regression head — keep only the backbone feature extractor.
  3. Run the full dataset through the backbone to get (N, feature_dim) embeddings.
  4. Fit sklearn RidgeCV (searches alpha automatically) on train embeddings.
  5. Save the fitted Ridge model with joblib.

Usage:
    # After training a backbone:
    python -m training.ridge_trainer

Inference:
    from training.ridge_trainer import RidgePredictor
    rp = RidgePredictor.load("outputs/models/ridge_model.joblib",
                              "outputs/models/best_model.pth")
    pred = rp.predict(tensor)   # tensor: (1, C, H, W)
"""

import os

import joblib
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet_model import ResNetRegression
from training.multi_sensor_dataset import MultiSensorDataset, sensor_channels
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _extract_features(model: ResNetRegression, loader: DataLoader, device) -> tuple:
    """Return (features_np, labels_np) for all samples in loader."""
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting features", leave=False):
            feats = model.extract_features(imgs.to(device))
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_feats), np.concatenate(all_labels)


def run_ridge_training(
    sensors=None,
    backbone_checkpoint=None,
    ridge_output_path=None,
    csv_path=None,
    alphas=None,
):
    """
    Extract CNN features from a trained backbone and fit RidgeCV.

    Returns a dict with R², RMSE, MAE on the validation split.
    """
    cfg = load_config()
    sensors = sensors or ["s2"]
    csv_path = csv_path or cfg["data"]["training_csv"]
    backbone_checkpoint = backbone_checkpoint or cfg["training"]["checkpoint"]
    ridge_output_path = ridge_output_path or os.path.join(
        cfg["training"]["model_dir"], "ridge_model.joblib"
    )
    alphas = alphas or [0.1, 1.0, 10.0, 100.0, 1000.0]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    n_channels = sensor_channels(sensors)
    model = ResNetRegression(in_channels=n_channels, dropout_p=0.0).to(device)

    if os.path.exists(backbone_checkpoint):
        state = torch.load(backbone_checkpoint, map_location="cpu")
        # Handle both full model state dicts and bare backbone dicts
        if any(k.startswith("backbone.") for k in state):
            model.load_state_dict(state)
        else:
            model.backbone.load_state_dict(state, strict=False)
        logger.info("Loaded backbone from %s", backbone_checkpoint)
    else:
        logger.warning("Checkpoint not found at %s — using random init.", backbone_checkpoint)

    train_ds = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=True)
    val_ds   = MultiSensorDataset(sensors=sensors, csv_path=csv_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    logger.info("Extracting train features…")
    X_train, y_train = _extract_features(model, train_loader, device)
    logger.info("Extracting val features…")
    X_val, y_val = _extract_features(model, val_loader, device)

    # Normalise features (important for Ridge)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    logger.info("Fitting RidgeCV on %d samples, %d features…", len(X_train), X_train.shape[1])
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_train_s, y_train)
    logger.info("Best alpha: %.1f", ridge.alpha_)

    preds = ridge.predict(X_val_s)
    import math
    r2   = r2_score(y_val, preds)
    rmse = math.sqrt(mean_squared_error(y_val, preds))
    mae  = mean_absolute_error(y_val, preds)
    logger.info("Ridge | R²=%.4f  RMSE=%.4f  MAE=%.4f", r2, rmse, mae)

    os.makedirs(os.path.dirname(ridge_output_path), exist_ok=True)
    joblib.dump({"ridge": ridge, "scaler": scaler, "sensors": sensors}, ridge_output_path)
    logger.info("Ridge model saved → %s", ridge_output_path)

    return {
        "run_name":   "ridge_" + "+".join(sensors),
        "sensors":    "+".join(sensors),
        "model_type": "ridge",
        "alpha":      ridge.alpha_,
        "r2":         round(r2,   4),
        "rmse":       round(rmse, 4),
        "mae":        round(mae,  4),
    }


class RidgePredictor:
    """Thin wrapper for inference with a saved Ridge model."""

    def __init__(self, ridge, scaler, model: ResNetRegression, device):
        self.ridge = ridge
        self.scaler = scaler
        self.model = model.eval()
        self.device = device

    @classmethod
    def load(cls, ridge_path: str, backbone_checkpoint: str, sensors=None):
        cfg = load_config()
        sensors = sensors or ["s2"]
        device = torch.device("cpu")

        bundle = joblib.load(ridge_path)
        sensors = bundle.get("sensors", sensors)
        n_channels = sensor_channels(sensors)
        model = ResNetRegression(in_channels=n_channels, dropout_p=0.0)
        if os.path.exists(backbone_checkpoint):
            model.load_state_dict(
                torch.load(backbone_checkpoint, map_location="cpu"), strict=False
            )
        return cls(bundle["ridge"], bundle["scaler"], model, device)

    def predict(self, tensor: torch.Tensor) -> float:
        with torch.no_grad():
            feat = self.model.extract_features(tensor.to(self.device)).cpu().numpy()
        feat_s = self.scaler.transform(feat)
        return float(self.ridge.predict(feat_s)[0])


if __name__ == "__main__":
    run_ridge_training(sensors=["s2"])
