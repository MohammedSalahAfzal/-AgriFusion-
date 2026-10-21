"""Agrifusion Net: EfficientNetV2-S spatial features + Transformer encoder."""
import torch
from torch import nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class AgrifusionNet(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 256, num_heads: int = 8,
                 num_transformer_layers: int = 2, dropout: float = 0.25,
                 pretrained: bool = False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.projection = nn.Conv2d(1280, embed_dim, kernel_size=1)
        self.position_embedding = nn.Parameter(torch.zeros(1, 49, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        feature_map = self.features(x)              # [B, 1280, 7, 7] for 224px inputs
        tokens = self.projection(feature_map).flatten(2).transpose(1, 2)  # [B, 49, D]
        if tokens.size(1) != self.position_embedding.size(1):
            # Supports other resolutions by interpolating positional embeddings.
            side = int(self.position_embedding.size(1) ** 0.5)
            pos = self.position_embedding.transpose(1, 2).reshape(1, -1, side, side)
            pos = nn.functional.interpolate(pos, size=(int(tokens.size(1) ** 0.5),) * 2,
                                            mode="bilinear", align_corners=False)
            pos = pos.flatten(2).transpose(1, 2)
        else:
            pos = self.position_embedding
        tokens = self.transformer(tokens + pos)
        pooled = self.norm(tokens.mean(dim=1))
        return self.classifier(pooled)
