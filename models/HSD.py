import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):

        x = x.flatten(2).transpose(1, 2)

        x = self.proj(x)

        return x


class InteractionRefinement(nn.Module):
    """
    Lightweight Interaction Refinement

    Purpose:
        Enhance collaborative interaction among
        multi-level fused representations without
        destroying the original SegFormer fusion stability.
    """

    def __init__(self, channels, norm_layer=nn.BatchNorm2d):

        super().__init__()

        # local interaction
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        # channel interaction
        self.pwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False
        )

        self.norm = norm_layer(channels)

        self.act = nn.ReLU(inplace=True)

        # residual scaling
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        identity = x

        x = self.dwconv(x)

        x = self.pwconv(x)

        x = self.norm(x)

        x = self.act(x)

        return identity + self.gamma * x


class DecoderHead(nn.Module):

    def __init__(
        self,
        in_channels=[64, 128, 320, 512],
        num_classes=40,
        dropout_ratio=0.1,
        norm_layer=nn.BatchNorm2d,
        embed_dim=768,
        align_corners=False
    ):

        super(DecoderHead, self).__init__()

        self.num_classes = num_classes

        self.dropout_ratio = dropout_ratio

        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = nn.Identity()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim

        # ---------------------------------------------------
        # Linear Embedding
        # ---------------------------------------------------

        self.linear_c4 = MLP(
            input_dim=c4_in_channels,
            embed_dim=embedding_dim
        )

        self.linear_c3 = MLP(
            input_dim=c3_in_channels,
            embed_dim=embedding_dim
        )

        self.linear_c2 = MLP(
            input_dim=c2_in_channels,
            embed_dim=embedding_dim
        )

        self.linear_c1 = MLP(
            input_dim=c1_in_channels,
            embed_dim=embedding_dim
        )

        # ---------------------------------------------------
        # Original SegFormer Fusion
        # ---------------------------------------------------

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(
                in_channels=embedding_dim * 4,
                out_channels=embedding_dim,
                kernel_size=1,
                bias=False
            ),

            norm_layer(embedding_dim),

            nn.ReLU(inplace=True)
        )

        # ---------------------------------------------------
        # Lightweight Interaction Refinement
        # ---------------------------------------------------

        self.interaction_refine = InteractionRefinement(
            embedding_dim,
            norm_layer=norm_layer
        )

        # ---------------------------------------------------
        # Prediction Head
        # ---------------------------------------------------

        self.linear_pred = nn.Conv2d(
            embedding_dim,
            self.num_classes,
            kernel_size=1
        )

    def forward(self, inputs):

        c1, c2, c3, c4 = inputs

        n = c4.shape[0]

        # ---------------------------------------------------
        # Linear Projection
        # ---------------------------------------------------

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(
            n,
            -1,
            c4.shape[2],
            c4.shape[3]
        )

        _c4 = F.interpolate(
            _c4,
            size=c1.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(
            n,
            -1,
            c3.shape[2],
            c3.shape[3]
        )

        _c3 = F.interpolate(
            _c3,
            size=c1.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(
            n,
            -1,
            c2.shape[2],
            c2.shape[3]
        )

        _c2 = F.interpolate(
            _c2,
            size=c1.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners
        )

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(
            n,
            -1,
            c1.shape[2],
            c1.shape[3]
        )

        # ---------------------------------------------------
        # Original SegFormer Multi-scale Fusion
        # ---------------------------------------------------

        _c = self.linear_fuse(
            torch.cat([_c4, _c3, _c2, _c1], dim=1)
        )

        # ---------------------------------------------------
        # Cross-level Interaction Refinement
        # ---------------------------------------------------

        _c = self.interaction_refine(_c)

        # ---------------------------------------------------
        # Prediction
        # ---------------------------------------------------

        x = self.dropout(_c)

        x = self.linear_pred(x)

        return x