"""
Multi-stream ResNet model (inspired by simonlazarus/Predicting-Poverty-with-Satellite-Images).

One independent ResNet18 encoder per sensor:
  - S2  stream : 3-channel input  → 512-d features
  - S1  stream : 2-channel input  → 512-d features
  - VIIRS stream: 1-channel input → 512-d features

Features are concatenated then passed through a small MLP fusion head.
This lets each sensor learn its own representation before fusion,
which outperforms early-fusion (stacking channels) when sensor modalities
have very different statistics.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from models.resnet_model import _patch_conv1

# Channels per sensor — must match multi_sensor_dataset._SENSOR_META
_SENSOR_CHANNELS = {"s2": 3, "s1": 2, "viirs": 1}


def _make_encoder(in_channels: int) -> nn.Module:
    """ResNet18 backbone that outputs (batch, 512) feature vectors."""
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    _patch_conv1(base, in_channels)
    base.fc = nn.Identity()
    return base


class MultiStreamResNet(nn.Module):
    """
    Args:
        sensors: ordered list of sensor keys from ["s2", "s1", "viirs"].
                 Input tensor channels must match the concatenated order.
        dropout_p: dropout probability in the fusion MLP.
    """

    FEATURE_DIM = 512

    def __init__(self, sensors: list[str], dropout_p: float = 0.3):
        super().__init__()
        if not sensors:
            raise ValueError("sensors list must not be empty")

        self.sensors = sensors
        self.channel_splits = [_SENSOR_CHANNELS[s] for s in sensors]
        fused_dim = self.FEATURE_DIM * len(sensors)

        self.encoders = nn.ModuleDict(
            {s: _make_encoder(_SENSOR_CHANNELS[s]) for s in sensors}
        )

        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (batch, fused_dim) before the regression head."""
        streams = torch.split(x, self.channel_splits, dim=1)
        feats = [self.encoders[s](patch) for s, patch in zip(self.sensors, streams)]
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.extract_features(x)).squeeze(1)  # (batch,)
