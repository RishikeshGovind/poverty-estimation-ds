"""
Monte Carlo Dropout uncertainty estimation.

Usage
-----
    from utils.uncertainty import mc_predict

    mean, std = mc_predict(model, img_tensor, device=device)
    # mean: (B,) predicted wealth index
    # std:  (B,) epistemic uncertainty (1-sigma)
"""

import torch
import torch.nn as nn
import numpy as np

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _enable_dropout(model: nn.Module):
    """Switch all Dropout layers to train mode while keeping BN in eval mode."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = None,
    primary_task: str = None,
    device: torch.device = None,
):
    """
    Run MC Dropout inference.

    Args:
        model:        trained model (ResNetRegression or MultiTaskResNet)
        x:            input tensor (B, C, H, W) — on CPU or device
        n_samples:    number of stochastic forward passes (default from config)
        primary_task: for MultiTaskResNet, which task key to return (default: first)
        device:       target device; if None uses x.device

    Returns:
        mean (B,), std (B,)  — numpy arrays
    """
    cfg = load_config()
    n_samples = n_samples or cfg["model"]["mc_samples"]
    if device is None:
        device = x.device if hasattr(x, "device") else torch.device("cpu")

    x = x.to(device)
    model.to(device)
    model.eval()
    _enable_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            if isinstance(out, dict):
                # MultiTaskResNet — use specified or first task
                key = primary_task or next(iter(out))
                out = out[key]
            preds.append(out.cpu().numpy())

    preds = np.stack(preds, axis=0)   # (n_samples, B)
    return preds.mean(axis=0), preds.std(axis=0)


def calibration_summary(means: np.ndarray, stds: np.ndarray, targets: np.ndarray) -> dict:
    """
    Quick calibration check: what fraction of true values fall within ±1σ and ±2σ.
    A well-calibrated model should hit ~68% and ~95% respectively.
    """
    within_1s = np.mean(np.abs(targets - means) <= stds)
    within_2s = np.mean(np.abs(targets - means) <= 2 * stds)
    logger.info("Calibration | ±1σ coverage=%.1f%%  ±2σ coverage=%.1f%%", within_1s * 100, within_2s * 100)
    return {"coverage_1sigma": round(within_1s, 4), "coverage_2sigma": round(within_2s, 4)}
