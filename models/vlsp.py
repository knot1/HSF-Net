import torch
import torch.nn as nn
import torch.nn.functional as F


class VLSPModule(nn.Module):
    """
    Vision-Language Semantic Prior (VLSP) modulation.

    Inputs:
        feat:           [B, C, H, W]
        semantic_prior: [B, K]
        conflict_map:   [B, 1, H, W]
    """
    def __init__(self, num_classes: int, channels: int, reduction: int = 4):
        super().__init__()
        hidden_dim = max(channels // reduction, 32)

        self.semantic_mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
            nn.Sigmoid()
        )

        self.conflict_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, feat: torch.Tensor, semantic_prior=None, conflict_map=None):
        if semantic_prior is None or semantic_prior.dim() != 2:
            return feat

        B, C, H, W = feat.shape
        if semantic_prior.shape[0] != B:
            return feat

        channel_gate = self.semantic_mlp(semantic_prior.float()).view(B, C, 1, 1)

        if conflict_map is not None:
            if conflict_map.shape[-2:] != (H, W):
                conflict_map = F.interpolate(
                    conflict_map,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            spatial_gate = self.conflict_gate(conflict_map.float())
            gate = channel_gate * spatial_gate
        else:
            gate = channel_gate

        return feat + self.gamma * feat * gate
# class VLSPModule(nn.Module):

#     def __init__(
#         self,
#         num_classes: int,
#         channels: int,
#         reduction: int = 4
#     ):
#         super().__init__()

#         hidden_dim = max(channels // reduction, 32)

#         self.semantic_mlp = nn.Sequential(
#             nn.Linear(num_classes, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, channels),
#             nn.Sigmoid()
#         )

#         self.conflict_gate = nn.Sequential(
#             nn.Conv2d(
#                 1,
#                 1,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False
#             ),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(
#                 1,
#                 channels,
#                 kernel_size=1,
#                 bias=False
#             ),

#             nn.Sigmoid()
#         )

#         self.gamma = nn.Parameter(
#             torch.tensor(0.01, dtype=torch.float32)
#         )

#     def forward(
#         self,
#         feat: torch.Tensor,
#         semantic_prior=None,
#         conflict_map=None
#     ):

#         if semantic_prior is None:
#             return feat

#         if semantic_prior.dim() != 2:
#             return feat

#         B, C, H, W = feat.shape

#         if semantic_prior.shape[0] != B:
#             return feat

#         channel_gate = self.semantic_mlp(
#             semantic_prior.float()
#         ).view(B, C, 1, 1)

#         if conflict_map is not None:

#             if conflict_map.shape[-2:] != (H, W):

#                 conflict_map = F.interpolate(
#                     conflict_map,
#                     size=(H, W),
#                     mode='bilinear',
#                     align_corners=False
#                 )

#             spatial_gate = self.conflict_gate(
#                 conflict_map.float()
#             )

#             gate = channel_gate * spatial_gate

#         else:
#             gate = channel_gate

#         return feat + self.gamma * feat * gate