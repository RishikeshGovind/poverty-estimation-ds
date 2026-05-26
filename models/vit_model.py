# models/vit_model.py
# Vision Transformer (ViT) for regression using timm and torch

import torch
import torch.nn as nn
import timm

class ViTRegression(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # 1. Load pretrained ViT backbone
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # 2. Replace classification head with regression head
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)  # Output shape: (batch,)

# Example usage:
# model = ViTRegression()
