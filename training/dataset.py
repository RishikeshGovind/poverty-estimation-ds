import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils.config import load_config
from utils.logging import get_logger

logger = get_logger(__name__)

class PovertyDataset(Dataset):
    def __init__(self, csv_path=None, train=True, split=None, transform=None):
        cfg = load_config()
        csv_path = csv_path or cfg["data"]["training_csv"]
        split = split or cfg["training"]["split"]
        self.norm_factor = cfg["sentinel2"]["normalization_factor"]

        df = pd.read_csv(csv_path)
        if not {"patch_file", "label"}.issubset(df.columns):
            raise ValueError(f"CSV must contain 'patch_file' and 'label' columns, got: {df.columns.tolist()}")
        if len(df) == 0:
            raise ValueError(f"Dataset CSV is empty: {csv_path}")

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(df) * split)
        self.data = df.iloc[:split_idx] if train else df.iloc[split_idx:]
        self.transform = transform
        logger.info("Loaded %s split: %d samples from %s", "train" if train else "val", len(self.data), csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = np.load(row["patch_file"]).astype(np.float32)
        # Sentinel-2 L2A surface reflectance values are 0–10000; scale to [0, 1]
        img = np.clip(img / self.norm_factor, 0, 1)
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(row["label"], dtype=torch.float32)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor, label_tensor
