import torch
import torch.nn as nn


class VLSPModule(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        module_type: str = "clip_text_guided_modulation",
        reduction: int = 4,
        gamma_init: float = 0.1,
    ):
        super().__init__()
        self.module_type = module_type
        hidden_dim = max(channels // reduction, 32)
        self.semantic_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, semantic_prior=None) -> torch.Tensor:
        if semantic_prior is None or semantic_prior.dim() != 2:
            return feat
        if semantic_prior.shape[0] != feat.shape[0]:
            return feat
        if self.module_type != "clip_text_guided_modulation":
            return feat

        gate = self.semantic_mlp(semantic_prior.float()).view(
            feat.shape[0], feat.shape[1], 1, 1
        )
        return feat + self.gamma * feat * gate
