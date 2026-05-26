# models/resnet_model.py
# Modified ResNet18 for multispectral regression
# Uses torch, torchvision

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetRegression(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 1. Load pretrained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 2. Modify first conv layer for multispectral input
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        # If in_channels != 3, average weights or initialize
        if in_channels != 3:
            with torch.no_grad():
                if in_channels > 3:
                    self.model.conv1.weight[:, :3] = old_conv.weight
                    for i in range(3, in_channels):
                        self.model.conv1.weight[:, i] = old_conv.weight[:, 0]
                else:
                    self.model.conv1.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
        # 3. Replace classification head with regression head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)  # Output shape: (batch,)

# Example usage:
# model = ResNetRegression(in_channels=3)
