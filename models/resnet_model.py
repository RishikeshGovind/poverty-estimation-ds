import torch
import torch.nn as nn
import torchvision.models as models


def _patch_conv1(resnet, in_channels):
    """Replace conv1 to accept arbitrary channel count, initialising extra channels
    as the mean of the pretrained RGB weights (preserves scale)."""
    old = resnet.conv1
    resnet.conv1 = nn.Conv2d(
        in_channels, old.out_channels,
        kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=old.bias is not None,
    )
    if in_channels != 3:
        with torch.no_grad():
            if in_channels < 3:
                resnet.conv1.weight.copy_(old.weight[:, :in_channels])
            else:
                resnet.conv1.weight[:, :3] = old.weight
                mean_w = old.weight.mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    resnet.conv1.weight[:, i] = mean_w.squeeze(1)
    else:
        resnet.conv1.weight = old.weight
        if old.bias is not None:
            resnet.conv1.bias = old.bias


class ResNetRegression(nn.Module):
    """Single-task ResNet18 regressor. Exposes feature_dim for downstream use."""

    FEATURE_DIM = 512

    def __init__(self, in_channels=3, dropout_p=0.0):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _patch_conv1(base, in_channels)
        self.feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.head = (
            nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(self.feature_dim, 1))
            if dropout_p > 0
            else nn.Linear(self.feature_dim, 1)
        )

    def extract_features(self, x):
        """Return (batch, feature_dim) without going through the regression head."""
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)  # (batch,)

    # Keep old attribute access for any code that used model.model.layer4
    @property
    def model(self):
        return self.backbone


class ResNetNTLPretrain(nn.Module):
    """
    Pretraining variant: shared backbone predicts both NTL (VIIRS proxy) and
    optionally a second head for the final wealth task.

    Used by training/pretrain_ntl.py.  After pretraining, transfer
    self.backbone weights to a ResNetRegression for fine-tuning.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _patch_conv1(base, in_channels)
        feat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.ntl_head = nn.Linear(feat, 1)   # predicts normalised VIIRS value

    def forward(self, x):
        f = self.backbone(x)
        return self.ntl_head(f).squeeze(1)   # (batch,)

    def transfer_to(self, target: ResNetRegression):
        """Copy backbone weights into a ResNetRegression for fine-tuning."""
        target.backbone.load_state_dict(self.backbone.state_dict())
        return target
