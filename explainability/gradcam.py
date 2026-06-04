"""
GradCAM and GradCAM++ explainability for poverty estimation models.

Supports:
  - ResNetRegression  (single-task scalar output)
  - MultiTaskResNet   (dict of per-SDG-task scalar outputs)

Target layer: backbone.layer4[-1]  — the final convolutional block of ResNet18,
whose feature maps carry the richest spatial signal before global pooling.

Usage
-----
    from explainability.gradcam import compute_cams, save_explanation_grid

    gc_maps, gcpp_maps = compute_cams(model, images, task="sdg1_wealth")
    save_explanation_grid(images, gc_maps, gcpp_maps, preds, labels, "out.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _RegressionTarget:
    """Callable target for pytorch-grad-cam — returns its input as-is (scalar regression)."""

    def __call__(self, output: torch.Tensor) -> torch.Tensor:
        return output


class _SingleTaskWrapper(nn.Module):
    """
    Makes MultiTaskResNet look like a single-output model.

    pytorch-grad-cam iterates over the model output to zip with targets.
    When the output is a dict that iteration yields string keys, breaking the
    gradient computation.  This wrapper selects one task's tensor so the
    library sees a plain (B,) float tensor.
    """

    def __init__(self, model: nn.Module, task: str) -> None:
        super().__init__()
        self._wrapped = model
        self._task = task
        # expose backbone so _get_target_layer can find layer4
        self.backbone = getattr(model, "backbone", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._wrapped(x)
        return out[self._task] if isinstance(out, dict) else out


def _get_target_layer(model: nn.Module) -> nn.Module:
    """Return layer4[-1] from the ResNet18 backbone (standard GradCAM hook point).

    Handles both ResNetRegression (backbone = ResNet18) and MultiTaskResNet
    (backbone = Sequential[ResNet18, Dropout]) by unwrapping the Sequential.
    """
    backbone = getattr(model, "backbone", model)
    # MultiTaskResNet wraps ResNet18 + Dropout in nn.Sequential
    if isinstance(backbone, nn.Sequential):
        backbone = backbone[0]
    return backbone.layer4[-1]


def _patch_to_rgb(patch: np.ndarray) -> np.ndarray:
    """
    Convert a C×H×W float patch (any number of channels) to H×W×3 float32
    in [0, 1] suitable for overlay rendering.

    First 3 channels are used as RGB; single-channel inputs are triplicated.
    """
    if patch.shape[0] >= 3:
        rgb = patch[:3]
    else:
        rgb = np.repeat(patch[:1], 3, axis=0)

    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.float32)
    lo, hi = rgb.min(), rgb.max()
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_cams(
    model: nn.Module,
    images: torch.Tensor,
    task: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GradCAM and GradCAM++ on a batch of images.

    Parameters
    ----------
    model  : ResNetRegression or MultiTaskResNet (trained, any device)
    images : (B, C, H, W) tensor, values in [0, 1]
    task   : task name for MultiTaskResNet (e.g. 'sdg1_wealth'); None = first task
    device : inference device; defaults to CUDA if available, else CPU

    Returns
    -------
    gc_maps   : np.ndarray (B, H, W)  GradCAM activations in [0, 1]
    gcpp_maps : np.ndarray (B, H, W)  GradCAM++ activations in [0, 1]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    images = images.to(device)

    # If the model returns a dict (MultiTaskResNet), wrap it so pytorch-grad-cam
    # sees a plain (B,) tensor — iterating a dict yields string keys, not tensors.
    with torch.no_grad():
        probe = model(images[:1])
    if isinstance(probe, dict):
        resolved_task = task or next(iter(probe))
        effective_model: nn.Module = _SingleTaskWrapper(model, resolved_task)
    else:
        effective_model = model

    target_layers = [_get_target_layer(effective_model)]
    B = images.shape[0]
    targets = [_RegressionTarget()] * B

    results: dict[str, np.ndarray] = {}
    for name, CamClass in [("gc", GradCAM), ("gcpp", GradCAMPlusPlus)]:
        with CamClass(model=effective_model, target_layers=target_layers) as cam:
            maps = cam(input_tensor=images, targets=targets)  # (B, H, W)
        results[name] = maps

    return results["gc"], results["gcpp"]


def activation_stats(cam_map: np.ndarray) -> dict:
    """
    Quantify how focused the activation map is.

    Returns
    -------
    top10_mean : mean activation in the top-10% brightest pixels
                 (higher = model concentrates on a small, high-confidence region)
    entropy    : Shannon entropy of the normalised activation distribution
                 (lower = sharper / more localised attention)
    """
    flat = cam_map.ravel()
    threshold = np.percentile(flat, 90)
    top10_mean = float(flat[flat >= threshold].mean())

    p = flat / (flat.sum() + 1e-9)
    entropy = float(-np.sum(p * np.log(p + 1e-9)))

    return {"top10_mean": round(top10_mean, 4), "entropy": round(entropy, 4)}


def save_explanation_grid(
    images: torch.Tensor,
    gc_maps: np.ndarray,
    gcpp_maps: np.ndarray,
    predictions: np.ndarray,
    labels: Optional[np.ndarray],
    out_path: str | Path,
    task: str = "sdg1_wealth",
) -> None:
    """
    Save a B-row × 3-column figure:
      col 0 — original satellite patch (RGB)
      col 1 — GradCAM overlay
      col 2 — GradCAM++ overlay

    Each row is annotated with the model prediction and (if provided) the label.
    """
    B = images.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(9, B * 2.8))
    if B == 1:
        axes = axes[np.newaxis, :]

    for i in range(B):
        rgb = _patch_to_rgb(images[i].cpu().numpy())
        gc_overlay   = show_cam_on_image(rgb, gc_maps[i],   use_rgb=True)
        gcpp_overlay = show_cam_on_image(rgb, gcpp_maps[i], use_rgb=True)

        ann = f"pred = {predictions[i]:.3f}"
        if labels is not None:
            ann += f"  |  label = {labels[i]:.3f}"

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Satellite\n{ann}", fontsize=7)
        axes[i, 1].imshow(gc_overlay)
        axes[i, 1].set_title("GradCAM", fontsize=7)
        axes[i, 2].imshow(gcpp_overlay)
        axes[i, 2].set_title("GradCAM++", fontsize=7)
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(
        f"Poverty model explainability  —  task: {task}",
        fontsize=9, y=1.005,
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
