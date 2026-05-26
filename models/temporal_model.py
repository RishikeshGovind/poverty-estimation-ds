# models/temporal_model.py
# Simple temporal model for image sequences using CNN + RNN
# Uses torch, torchvision

import torch
import torch.nn as nn
import torchvision.models as models

class TemporalPovertyModel(nn.Module):
    def __init__(self, in_channels=3, hidden_size=128, num_layers=1):
        super().__init__()
        # 1. CNN backbone (ResNet18, remove final layer)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Output: (batch, 512, 1, 1)
        self.feature_dim = 512
        # 2. RNN for temporal modeling
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # 3. Regression head
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        batch, seq_len, C, H, W = x.shape
        x = x.view(batch * seq_len, C, H, W)
        feats = self.cnn(x).view(batch, seq_len, -1)  # (batch, seq_len, feature_dim)
        rnn_out, _ = self.rnn(feats)  # (batch, seq_len, hidden_size)
        out = self.fc(rnn_out[:, -1, :])  # Use last time step
        return out.squeeze(1)  # (batch,)

# Example usage:
# model = TemporalPovertyModel(in_channels=3)
