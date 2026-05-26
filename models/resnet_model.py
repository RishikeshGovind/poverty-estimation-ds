# models/resnet_model.py
# Modified ResNet18 for multispectral regression
# Uses torch, torchvision

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetRegression(nn.Module):
    def __init__(self, in_channels=3, dropout_p=0.0):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if in_channels != 3:
            with torch.no_grad():
                if in_channels < 3:
                    self.model.conv1.weight = nn.Parameter(
                        old_conv.weight[:, :in_channels].clone()
                    )
                else:
                    self.model.conv1.weight[:, :3] = old_conv.weight
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    for i in range(3, in_channels):
                        self.model.conv1.weight[:, i] = mean_w.squeeze(1)

        num_features = self.model.fc.in_features
        self.model.fc = (
            nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(num_features, 1))
            if dropout_p > 0
            else nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)  # (batch,)

# Example usage:
# model = ResNetRegression(in_channels=3)
