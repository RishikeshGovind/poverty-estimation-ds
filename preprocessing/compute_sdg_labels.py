"""
Augment the training CSV with SDG proxy label columns derived from existing
satellite patches. Must be run after create_training_dataset.py.

Labels added
------------
sdg1_wealth   : DHS wealth index (alias of 'label', primary supervised signal)
sdg7_ntl      : mean VIIRS radiance normalised to [0,1] — proxy for energy access (SDG 7)
sdg11_buildup : mean S2 brightness normalised to [0,1] — proxy for built-up density (SDG 11)

Values are NaN when the corresponding patch file is absent.
"""

import os
import numpy as np
import pandas as pd

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)


def _patch_mean(path: str, norm: float) -> float:
    if not path or not isinstance(path, str) or not path.strip():
        return np.nan
    if not os.path.exists(path):
        return np.nan
    try:
        arr = np.load(path).astype(np.float32)
        return float(np.nanmean(arr) / norm)
    except Exception as e:
        logger.warning("Could not read patch %s: %s", path, e)
        return np.nan


def main():
    cfg = load_config()
    csv_path = cfg["data"]["training_csv"]
    s2_norm   = cfg["sentinel2"]["normalization_factor"]
    viirs_norm = cfg["viirs"]["normalization_clip"]

    if not os.path.exists(csv_path):
        logger.error("Training CSV not found: %s — run create_training_dataset.py first.", csv_path)
        return

    df = pd.read_csv(csv_path)

    # SDG 1: wealth index is already the label
    df["sdg1_wealth"] = df["label"]

    # SDG 7: mean VIIRS radiance per cluster
    viirs_col = "viirs_patch_file"
    if viirs_col in df.columns:
        df["sdg7_ntl"] = df[viirs_col].apply(lambda p: _patch_mean(p, viirs_norm))
        n_valid = df["sdg7_ntl"].notna().sum()
        logger.info("SDG 7 (NTL): %d / %d samples have VIIRS data", n_valid, len(df))
    else:
        df["sdg7_ntl"] = np.nan
        logger.warning("No viirs_patch_file column — sdg7_ntl will be NaN for all samples.")

    # SDG 11: mean S2 brightness across all bands per cluster
    s2_col = "s2_patch_file" if "s2_patch_file" in df.columns else "patch_file"
    df["sdg11_buildup"] = df[s2_col].apply(lambda p: _patch_mean(p, s2_norm))
    n_valid = df["sdg11_buildup"].notna().sum()
    logger.info("SDG 11 (buildup): %d / %d samples have S2 data", n_valid, len(df))

    df.to_csv(csv_path, index=False)
    logger.info(
        "SDG labels written to %s | columns added: sdg1_wealth, sdg7_ntl, sdg11_buildup",
        csv_path,
    )
    logger.info(
        "Label stats:\n%s",
        df[["sdg1_wealth", "sdg7_ntl", "sdg11_buildup"]].describe().to_string(),
    )


if __name__ == "__main__":
    main()
