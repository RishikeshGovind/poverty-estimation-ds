"""
MultiTaskResNet — shared ResNet18 backbone with one regression head per SDG task.

Architecture
------------
  Input (B, C, H, W)
      │
  ResNet18 backbone (conv1 → layer4 → avgpool)   [shared weights]
      │  (B, 512)
  Dropout(dropout_p)
      │
  ┌───┴──────────────┐
  head_sdg1_wealth  head_sdg7_ntl  head_sdg11_buildup  …
  Linear(512, 1)    Linear(512, 1)    Linear(512, 1)
      │                   │                  │
  scalar pred         scalar pred        scalar pred

forward() returns a dict {task_name: (B,) tensor}.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from utils.config import load_config


def _build_backbone(in_channels: int, dropout_p: float):
    """Return ResNet18 backbone (up to avgpool) and feature dim."""
    base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    old_conv = base.conv1
    base.conv1 = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels <= 3:
            base.conv1.weight = nn.Parameter(old_conv.weight[:, :in_channels].clone())
        else:
            base.conv1.weight[:, :3] = old_conv.weight
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                base.conv1.weight[:, i] = mean_w.squeeze(1)

    feat_dim = base.fc.in_features
    # Remove the original FC — we'll use task heads instead
    base.fc = nn.Identity()

    layers = nn.Sequential(
        base,
        nn.Dropout(p=dropout_p),
    )
    return layers, feat_dim


class MultiTaskResNet(nn.Module):
    def __init__(self, in_channels: int = 3, tasks: list = None, dropout_p: float = 0.3):
        super().__init__()
        cfg = load_config()
        if tasks is None:
            tasks = cfg["tasks"]["names"]
        if dropout_p is None:
            dropout_p = cfg["model"]["dropout_p"]

        self.tasks = tasks
        self.backbone, feat_dim = _build_backbone(in_channels, dropout_p)
        self.heads = nn.ModuleDict({
            task: nn.Linear(feat_dim, 1) for task in tasks
        })

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)                         # (B, feat_dim)
        return {task: self.heads[task](features).squeeze(1) # (B,)
                for task in self.tasks}

    def predict_primary(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method — returns only the first task's predictions."""
        return self.forward(x)[self.tasks[0]]
