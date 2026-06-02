import torch
import torch.nn as nn


class SemanticAwareRefinementBlock(nn.Module):
    """
    Semantic-aware Feature Refinement (SAFR).
    F_out = F_in + delta(F_in, S_vl)
    """
    def __init__(self, channels: int, num_classes: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(channels // reduction, 32)

        self.semantic_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, semantic_prior=None) -> torch.Tensor:
        if semantic_prior is None or semantic_prior.dim() != 2:
            return feat

        B, C, _, _ = feat.shape
        if semantic_prior.shape[0] != B:
            return feat

        gate = self.semantic_mlp(semantic_prior.float()).view(B, C, 1, 1)
        delta = self.refine(feat) * gate
        return feat + self.gamma * delta
