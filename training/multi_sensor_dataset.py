"""
MultiSensorDataset — loads patches from one or more of: S2, S1, VIIRS.

Sensors are specified as a list, e.g. ["s2", "s1"] or ["s2", "s1", "viirs"].
Each sensor's patch is normalised independently then all channels are
concatenated along axis-0 before being returned as a single tensor.

Channel layout (when all sensors present):
  [0:3]  S2  — B04, B03, B02  (3 ch, normalised /10000)
  [3:5]  S1  — VV, VH          (2 ch, clipped /0.5)
  [5:6]  VIIRS — DNB            (1 ch, clipped /200)

If a patch file is missing for a sample that sensor's channels are filled
with zeros so the batch shape is always consistent.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

# Sensor metadata: (csv_column, n_bands, norm_config_key, norm_attr)
_SENSOR_META = {
    "s2":    ("s2_patch_file",    3, "sentinel2",  "normalization_factor"),
    "s1":    ("s1_patch_file",    2, "sentinel1",  "normalization_clip"),
    "viirs": ("viirs_patch_file", 1, "viirs",      "normalization_clip"),
}

ALL_SENSORS = ["s2", "s1", "viirs"]


def sensor_channels(sensors):
    """Return total channel count for a given list of sensors."""
    return sum(_SENSOR_META[s][1] for s in sensors)


def _load_patch(path, n_bands, norm, patch_size):
    """Load a .npy patch, normalise, and resize to patch_size × patch_size."""
    if not path or not isinstance(path, str) or path.strip() == "":
        return torch.zeros(n_bands, patch_size, patch_size)
    try:
        arr = np.load(path).astype(np.float32)           # (C, H, W)
        arr = np.clip(arr / norm, 0, 1)
        t = torch.from_numpy(arr)
        # Handle patches with fewer bands than expected (edge case)
        if t.shape[0] < n_bands:
            pad = torch.zeros(n_bands - t.shape[0], *t.shape[1:])
            t = torch.cat([t, pad], dim=0)
        elif t.shape[0] > n_bands:
            t = t[:n_bands]
        # Resize to target spatial size if needed (e.g. VIIRS ~5×5 → 256×256)
        if t.shape[-2:] != (patch_size, patch_size):
            t = F.interpolate(
                t.unsqueeze(0), size=(patch_size, patch_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
        return t
    except Exception as e:
        logger.warning("Could not load patch %s: %s — returning zeros.", path, e)
        return torch.zeros(n_bands, patch_size, patch_size)


class MultiSensorDataset(Dataset):
    """
    Args:
        sensors: subset of ["s2", "s1", "viirs"]. Defaults to all available.
        csv_path: path to training CSV (defaults to config data.training_csv).
        train: if True load train split, else val split.
    """

    def __init__(self, sensors=None, csv_path=None, train=True, split=None, transform=None):
        cfg = load_config()
        csv_path = csv_path or cfg["data"]["training_csv"]
        split = split or cfg["training"]["split"]
        patch_size = cfg["training"]["patch_size"]

        self.sensors = sensors if sensors is not None else ALL_SENSORS
        self.patch_size = patch_size
        self.transform = transform

        # Pre-compute per-sensor normalisation constants from config
        self._norms = {}
        for s in self.sensors:
            _, _, cfg_section, cfg_key = _SENSOR_META[s]
            self._norms[s] = cfg[cfg_section][cfg_key]

        df = pd.read_csv(csv_path)

        # Backward-compat: if s2_patch_file column absent, use patch_file
        if "s2_patch_file" not in df.columns and "patch_file" in df.columns:
            df["s2_patch_file"] = df["patch_file"]

        if len(df) == 0:
            raise ValueError(f"Dataset CSV is empty: {csv_path}")

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * split)
        self.data = df.iloc[:split_idx] if train else df.iloc[split_idx:]
        self.n_channels = sensor_channels(self.sensors)

        logger.info(
            "MultiSensorDataset | sensors=%s  channels=%d  split=%s  n=%d",
            self.sensors, self.n_channels, "train" if train else "val", len(self.data),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patches = []
        for s in self.sensors:
            col, n_bands, _, _ = _SENSOR_META[s]
            path = row.get(col, "")
            patches.append(_load_patch(path, n_bands, self._norms[s], self.patch_size))

        img = torch.cat(patches, dim=0)   # (C_total, H, W)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return img, label
