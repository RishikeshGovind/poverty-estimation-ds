"""
Phase 5 — Tabular + CNN fusion model.

Combines ResNet image features (512-d) with OSM/building tabular features
through a learned MLP fusion head.  This matters because accessibility
(road density, distance to hospital) adds signal that's invisible in satellite
images at ~10m resolution.

Architecture:
  image input   → ResNet18 backbone → 512-d features
  tabular input → BatchNorm → Linear → 64-d features
  concat [512 + 64] → Linear(576, 128) → ReLU → Dropout → Linear(128, 1)
"""

import torch
import torch.nn as nn

from models.resnet_model import ResNetRegression

# OSM + building features (must match extract_osm_features.py column order)
OSM_FEATURE_COLS = [
    "road_density_km_per_km2",
    "dist_to_hospital_km",
    "dist_to_school_km",
    "dist_to_market_km",
    "dist_to_bank_km",
    "n_amenities_1km",
    "building_count",
    "mean_area_m2",
    "building_density_per_km2",
]
N_TABULAR = len(OSM_FEATURE_COLS)


class TabularFusionModel(nn.Module):
    """
    Args:
        in_channels:  image channels (default 3 for S2-only)
        n_tabular:    number of tabular features
        dropout_p:    dropout probability in fusion head
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_tabular: int = N_TABULAR,
        dropout_p: float = 0.3,
    ):
        super().__init__()

        # Image branch: ResNet18 backbone (feature extractor only)
        cnn_base = ResNetRegression(in_channels=in_channels, dropout_p=0.0)
        self.cnn_backbone = cnn_base.backbone
        img_feat_dim = cnn_base.feature_dim   # 512

        # Tabular branch: normalise + project to 64-d
        self.tab_branch = nn.Sequential(
            nn.BatchNorm1d(n_tabular),
            nn.Linear(n_tabular, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = img_feat_dim + 64

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 1),
        )

    def extract_features(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        """Return (batch, fused_dim) before regression head."""
        img_feat = self.cnn_backbone(images)          # (B, 512)
        tab_feat = self.tab_branch(tabular)            # (B, 64)
        return torch.cat([img_feat, tab_feat], dim=1)  # (B, 576)

    def forward(self, images: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        return self.fusion_head(self.extract_features(images, tabular)).squeeze(1)  # (B,)

    def load_pretrained_backbone(self, checkpoint_path: str):
        """Load NTL-pretrained backbone weights into the CNN branch."""
        import torch
        state = torch.load(checkpoint_path, map_location="cpu")
        self.cnn_backbone.load_state_dict(state, strict=False)
