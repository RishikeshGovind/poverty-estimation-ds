"""
Creates realistic synthetic satellite patches for pipeline testing.

Patches are spatially structured and correlated with the wealth index so the
model can actually learn a signal.  Replace with real data when available.

S2  (3ch): richer clusters → brighter (more built-up reflectance)
S1  (2ch): VV/VH backscatter with roughness noise
VIIRS (1ch): nighttime lights proportional to wealth
"""

import os
import numpy as np
import pandas as pd
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)
rng = np.random.default_rng(42)


def _perlin_like(h, w, scale=32):
    """Simple spatially-smooth noise via bilinear upsampling."""
    small = rng.random((h // scale + 2, w // scale + 2)).astype(np.float32)
    from PIL import Image
    big = np.array(
        Image.fromarray(small).resize((w, h), Image.BILINEAR)
    )
    return (big - big.min()) / (big.max() - big.min() + 1e-8)


def make_s2_patch(wealth: float, size: int = 256) -> np.ndarray:
    """
    3-band S2 patch (R, G, B) normalised to [0, 1].
    Wealth ∈ [-2, 2] → base reflectance ∈ [0.05, 0.45].
    Rich = brighter (built-up); poor = darker (bare soil/vegetation).
    """
    base = 0.05 + 0.20 * (wealth + 2) / 4          # 0.05 → 0.25
    urban_frac = max(0, (wealth + 1) / 3)            # 0 → 1 for richest

    spatial = _perlin_like(size, size, scale=32)
    urban_mask = (spatial > (1 - urban_frac)).astype(np.float32)

    bands = []
    for gain in [1.0, 0.95, 0.85]:                  # R, G, B
        noise = rng.normal(0, 0.02, (size, size)).astype(np.float32)
        b = base * gain + 0.15 * urban_mask + noise
        bands.append(np.clip(b, 0, 1))

    return np.stack(bands, axis=0)                   # (3, H, W)


def make_s1_patch(wealth: float, size: int = 256) -> np.ndarray:
    """
    2-band S1 patch (VV, VH) normalised to [0, 1].
    Urban structures → higher VV backscatter.
    """
    spatial = _perlin_like(size, size, scale=48)
    base_vv = 0.10 + 0.25 * (wealth + 2) / 4
    base_vh = base_vv * 0.6

    noise_vv = rng.normal(0, 0.03, (size, size)).astype(np.float32)
    noise_vh = rng.normal(0, 0.02, (size, size)).astype(np.float32)

    vv = np.clip(base_vv + 0.10 * spatial + noise_vv, 0, 1)
    vh = np.clip(base_vh + 0.06 * spatial + noise_vh, 0, 1)
    return np.stack([vv, vh], axis=0)                # (2, H, W)


def make_viirs_patch(wealth: float, size: int = 256) -> np.ndarray:
    """
    1-band VIIRS nighttime lights patch normalised to [0, 1].
    Very poor = near-zero, richest = 0.5 with urban bright spots.
    """
    ntl_base = max(0, (wealth + 1.5) / 4) ** 1.5    # nonlinear
    spatial  = _perlin_like(size, size, scale=64)
    noise    = rng.normal(0, 0.01, (size, size)).astype(np.float32)

    lights = np.clip(ntl_base * (0.5 + 0.5 * spatial) + noise, 0, 1)
    return lights[np.newaxis]                         # (1, H, W)


def create_patches_for_survey(
    survey_csv: str,
    patches_dir: str,
    patch_size: int = 256,
    output_csv: str = None,
):
    df = pd.read_csv(survey_csv)
    os.makedirs(os.path.join(patches_dir, "s2"),    exist_ok=True)
    os.makedirs(os.path.join(patches_dir, "s1"),    exist_ok=True)
    os.makedirs(os.path.join(patches_dir, "viirs"), exist_ok=True)

    s2_paths, s1_paths, viirs_paths = [], [], []

    for i, row in df.iterrows():
        wealth = float(row.get("wealth_index", row.get("label", 0.0)))

        s2_patch    = make_s2_patch(wealth,   patch_size) * 10000   # undo /10000 norm
        s1_patch    = make_s1_patch(wealth,   patch_size) * 0.5     # undo /0.5 norm
        viirs_patch = make_viirs_patch(wealth, patch_size) * 200    # undo /200 norm

        s2_path    = os.path.join(patches_dir, "s2",    f"patch_{i:05d}.npy")
        s1_path    = os.path.join(patches_dir, "s1",    f"patch_{i:05d}.npy")
        viirs_path = os.path.join(patches_dir, "viirs", f"patch_{i:05d}.npy")

        np.save(s2_path,    s2_patch.astype(np.float32))
        np.save(s1_path,    s1_patch.astype(np.float32))
        np.save(viirs_path, viirs_patch.astype(np.float32))

        s2_paths.append(s2_path)
        s1_paths.append(s1_path)
        viirs_paths.append(viirs_path)

    df["label"]            = df.get("wealth_index", df.get("label", 0.0))
    df["s2_patch_file"]    = s2_paths
    df["s1_patch_file"]    = s1_paths
    df["viirs_patch_file"] = viirs_paths
    df["patch_file"]       = s2_paths   # backward compat

    out = output_csv or survey_csv.replace(".csv", "_with_patches.csv")
    df.to_csv(out, index=False)
    logger.info("Synthetic patches created → %s  (%d rows)", out, len(df))
    return out


if __name__ == "__main__":
    cfg = load_config()
    out_csv = create_patches_for_survey(
        survey_csv  = cfg["data"]["survey_path"],
        patches_dir = cfg["data"]["patches_dir"],
        patch_size  = cfg["training"]["patch_size"],
        output_csv  = cfg["data"]["training_csv"],
    )
    print(f"Training CSV ready: {out_csv}")
