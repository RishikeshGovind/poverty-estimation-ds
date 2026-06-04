"""
Run GradCAM / GradCAM++ explainability on a sample of patches.

Loads a trained checkpoint, picks random patches from the training CSV, and
produces two artefacts in outputs/explainability/:

  gradcam_<task>.png           — visual grid: satellite | GradCAM | GradCAM++
  gradcam_summary_<task>.csv   — per-patch predictions, labels, focus metrics

Usage
-----
    python -m explainability.run_explainability
    python -m explainability.run_explainability --n-samples 16 --task sdg7_ntl
    python -m explainability.run_explainability \\
        --checkpoint outputs/models/multitask_best.pth \\
        --model-type multitask --task sdg11_buildup
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch

from utils.config import load_config
from utils.logging import get_logger
from explainability.gradcam import (
    activation_stats,
    compute_cams,
    save_explanation_grid,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(
    checkpoint_path: str,
    model_type: str,
    in_channels: int,
) -> torch.nn.Module:
    from models.resnet_model import ResNetRegression
    from models.multitask_model import MultiTaskResNet

    cfg = load_config()
    if model_type == "multitask":
        model = MultiTaskResNet(in_channels=in_channels)
    else:
        model = ResNetRegression(
            in_channels=in_channels,
            dropout_p=cfg["model"]["dropout_p"],
        )

    if not os.path.exists(checkpoint_path):
        logger.warning(
            "Checkpoint not found at '%s' — using random weights for demo.",
            checkpoint_path,
        )
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

    return model.eval()


def _load_patches(
    csv_path: str,
    n_samples: int,
    norm_factor: float,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """
    Load n_samples random patches from training_csv.

    Returns
    -------
    images : (B, C, H, W) float32 tensor in [0, 1]
    labels : (B,) float array
    paths  : list of patch file paths (length B)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training CSV not found: {csv_path}\n"
            "Run the data pipeline first (pipeline/run_phase1.py)."
        )

    df = pd.read_csv(csv_path)
    if "patch_file" not in df.columns:
        raise ValueError("CSV must have a 'patch_file' column.")

    # Keep only rows where the patch file actually exists on disk
    df = df[df["patch_file"].notna() & df["patch_file"].ne("")]
    df = df[df["patch_file"].apply(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No accessible patch files found via '{csv_path}'.\n"
            "Check that the patches_dir paths in config.yaml are correct."
        )

    sample = df.sample(min(n_samples, len(df)), random_state=0).reset_index(drop=True)
    images, labels, paths = [], [], []
    for _, row in sample.iterrows():
        img = np.load(row["patch_file"]).astype(np.float32)
        img = np.clip(img / norm_factor, 0.0, 1.0)
        images.append(img)
        labels.append(float(row.get("label", 0.0)))
        paths.append(str(row["patch_file"]))

    return torch.from_numpy(np.stack(images)), np.array(labels), paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GradCAM / GradCAM++ explainability for poverty models"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to .pth checkpoint (default: training.checkpoint from config.yaml)",
    )
    parser.add_argument(
        "--model-type", default="resnet", choices=["resnet", "multitask"],
        help="Model architecture to load (default: resnet)",
    )
    parser.add_argument(
        "--task", default=None,
        help="SDG task head for MultiTaskResNet, e.g. sdg1_wealth (default: first task)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=8,
        help="Number of patches to explain (default: 8)",
    )
    parser.add_argument(
        "--output-dir", default="outputs/explainability",
        help="Directory for output figures and CSV",
    )
    args = parser.parse_args()

    cfg         = load_config()
    checkpoint  = args.checkpoint or cfg["training"]["checkpoint"]
    csv_path    = cfg["data"]["training_csv"]
    norm_factor = cfg["sentinel2"]["normalization_factor"]
    task        = args.task or cfg["tasks"]["names"][0]
    out_dir     = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load patches ────────────────────────────────────────────────────────
    logger.info("Loading up to %d patches from %s", args.n_samples, csv_path)
    images, labels, paths = _load_patches(csv_path, args.n_samples, norm_factor)
    in_channels = images.shape[1]
    B = len(images)
    logger.info("Loaded %d patches  shape=%s  channels=%d", B, tuple(images.shape), in_channels)

    # ── Load model ──────────────────────────────────────────────────────────
    model = _load_model(checkpoint, args.model_type, in_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    logger.info("Model: %s  device: %s", args.model_type, device)

    # ── Predictions ─────────────────────────────────────────────────────────
    with torch.no_grad():
        raw_out = model(images.to(device))
    if isinstance(raw_out, dict):
        preds = raw_out[task].cpu().numpy()
    else:
        preds = raw_out.cpu().numpy()

    # ── GradCAM / GradCAM++ ─────────────────────────────────────────────────
    logger.info("Running GradCAM and GradCAM++ ...")
    task_arg = task if args.model_type == "multitask" else None
    gc_maps, gcpp_maps = compute_cams(model, images, task=task_arg, device=device)

    # ── Save figure ──────────────────────────────────────────────────────────
    fig_path = os.path.join(out_dir, f"gradcam_{task}.png")
    save_explanation_grid(images, gc_maps, gcpp_maps, preds, labels, fig_path, task=task)
    logger.info("Explanation grid  →  %s", fig_path)

    # ── Save summary CSV ─────────────────────────────────────────────────────
    rows = []
    for i, path in enumerate(paths):
        gc_stat   = activation_stats(gc_maps[i])
        gcpp_stat = activation_stats(gcpp_maps[i])
        rows.append({
            "patch_file":      path,
            "task":            task,
            "prediction":      round(float(preds[i]),  4),
            "label":           round(float(labels[i]), 4),
            # Focus metrics — higher top10_mean / lower entropy = more localised attention
            "gc_top10_mean":   gc_stat["top10_mean"],
            "gc_entropy":      gc_stat["entropy"],
            "gcpp_top10_mean": gcpp_stat["top10_mean"],
            "gcpp_entropy":    gcpp_stat["entropy"],
        })

    summary_df = pd.DataFrame(rows)
    csv_out = os.path.join(out_dir, f"gradcam_summary_{task}.csv")
    summary_df.to_csv(csv_out, index=False)
    logger.info("Summary CSV       →  %s", csv_out)
    logger.info("\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
