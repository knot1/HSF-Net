import torch
import torch.nn as nn
import torch.nn.functional as F


class CMAL(nn.Module):
    def __init__(self, loss_type: str = "cosine", temperature: float = 0.07):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.temperature = temperature

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        if feat_a is None or feat_b is None:
            raise ValueError("CCA loss requires both feature tensors.")

        vec_a = self._pool(feat_a)
        vec_b = self._pool(feat_b)
        vec_a = F.normalize(vec_a, dim=1)
        vec_b = F.normalize(vec_b, dim=1)

        if self.loss_type == "cosine":
            return 1.0 - (vec_a * vec_b).sum(dim=1).mean()

        if self.loss_type == "contrastive":
            logits = vec_a @ vec_b.t() / self.temperature
            labels = torch.arange(vec_a.size(0), device=vec_a.device)
            loss_a = F.cross_entropy(logits, labels)
            loss_b = F.cross_entropy(logits.t(), labels)
            return 0.5 * (loss_a + loss_b)

        raise ValueError(f"Unsupported CCA loss type: {self.loss_type}")
