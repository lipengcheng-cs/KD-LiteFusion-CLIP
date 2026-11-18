import torch
from torch import nn
from torch.nn import functional as F


class LiteFusionStudent(nn.Module):
    def __init__(self, dim, rank, hidden_dim, num_classes, dropout, shortcut_weight):
        super().__init__()
        self.shortcut_weight = shortcut_weight
        self.vision_down = nn.Linear(dim, rank, bias=False)
        self.text_down = nn.Linear(dim, rank, bias=False)
        self.interaction_up = nn.Linear(rank, dim)
        self.residual = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, vision_feature, text_feature):
        interaction = self.interaction_up(
            self.vision_down(vision_feature) * self.text_down(text_feature)
        )
        residual = self.residual(torch.cat((vision_feature, text_feature), dim=-1))
        fused = self.norm(interaction + residual)
        shortcut = (vision_feature + text_feature) / 2
        feature = F.normalize(fused + self.shortcut_weight * shortcut, dim=-1)
        return self.classifier(feature)
