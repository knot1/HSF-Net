import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as DCT
import matplotlib.pyplot as plt
import numpy as np

from .cmal import CMAL
from .sffm import FDRModule
from .sgfm import VLSPModule

def denormalize_dsm(x):
    mean = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    std = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    return x * std + mean

def denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
    return x * std + mean




def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


class Baseline(nn.Module):
    def __init__(self, cfg=None, num_classes=None, norm_layer=nn.BatchNorm2d, in_chans=None, class_labels=None):
        super(Baseline, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            self.in_chans = [3, 1]

        if cfg.backbone == 'mit_b5':
            from .encoder import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)
        elif cfg.backbone == 'mit_b4':
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)
        elif cfg.backbone == 'mit_b2':
            from .encoder import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)
        elif cfg.backbone == 'mit_b1':
            from .encoder import mit_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)
        elif cfg.backbone == 'mit_b0':
            from .encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)
            self.channels = [32, 64, 160, 256]
        else:
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes)

        self.backbone.apply_semantic_modulation = False

        # from .Seg_head import DecoderHead
        from .HSD import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)

        self.prompt_semantic = None
        prompt_cfg = getattr(cfg, "prompt_semantic", None)
        if prompt_cfg is not None and getattr(prompt_cfg, "enabled", False):
            if class_labels is None:
                class_labels = [str(idx) for idx in range(num_classes)]
            from .prompt_semantic import PromptSemanticPrior
            self.prompt_semantic = PromptSemanticPrior(
                class_labels=class_labels,
                model_name=prompt_cfg.model_name,
                pretrained=prompt_cfg.pretrained,
                image_size=prompt_cfg.image_size,
            )

        self.vlsp_modules = nn.ModuleDict()
        self.fdr_modules = nn.ModuleDict()
        self.cca_loss = None
        self.cca_weight = 0.0
        self.cca_position = None

        vlsp_cfg = getattr(cfg, "vlsp", None)
        if vlsp_cfg is not None and getattr(vlsp_cfg, "enable", False):
            vlsp_stages = self._normalize_stages(getattr(vlsp_cfg, "stages", []))
            vlsp_type = getattr(vlsp_cfg, "type", "clip_text_guided_modulation")
            for stage in vlsp_stages:
                self.vlsp_modules[str(stage)] = VLSPModule(
                    num_classes=num_classes,
                    channels=self.channels[stage - 1],
                    module_type=vlsp_type,
                )

        fdr_cfg = getattr(cfg, "fdr", None)
        if fdr_cfg is not None and getattr(fdr_cfg, "enable", False):
            fdr_stages = self._normalize_stages(getattr(fdr_cfg, "stages", []))
            fdr_mode = getattr(fdr_cfg, "mode", "fft")
            for stage in fdr_stages:
                self.fdr_modules[str(stage)] = FDRModule(
                    dim=self.channels[stage - 1],
                    mode=fdr_mode,
                )

        cca_cfg = getattr(cfg, "cca", None)
        if cca_cfg is not None and getattr(cca_cfg, "enable", False):
            self.cca_loss = CMAL(loss_type=getattr(cca_cfg, "loss_type", "cosine"))
            self.cca_weight = float(getattr(cca_cfg, "loss_weight", 1.0))
            self.cca_position = getattr(cca_cfg, "position", "post_fusion")

        self.init_weights(cfg, pretrained=cfg.pretrained_backbone)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def _normalize_stages(self, stages):
        stage_set = set()
        if stages is None:
            return []
        for stage in stages:
            try:
                idx = int(stage)
            except (TypeError, ValueError):
                raise ValueError(f"Stage value must be an integer, got {stage!r}")
            if not 1 <= idx <= len(self.channels):
                raise ValueError(f"Stage {idx} out of valid range 1..{len(self.channels)}")
            stage_set.add(idx)
        return sorted(stage_set)

    def _apply_stage_modules(self, features, semantic_prior=None):
        if not self.vlsp_modules and not self.fdr_modules:
            return features
        outputs = []
        for idx, feat in enumerate(features, start=1):
            stage_key = str(idx)
            if stage_key in self.fdr_modules:
                feat = self.fdr_modules[stage_key](feat)
            if stage_key in self.vlsp_modules:
                feat = self.vlsp_modules[stage_key](feat, semantic_prior=semantic_prior)
            outputs.append(feat)
        return outputs

    def _zero_loss(self, modal_features=None, fallback=None):
        if fallback is not None:
            return fallback.new_zeros(1)
        if isinstance(modal_features, dict) and modal_features:
            sample = next(iter(modal_features.values()))
            return sample.new_zeros(1)
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)

    def _compute_cca_loss(self, fused_feat, modal_features):
        if self.cca_loss is None:
            return self._zero_loss(modal_features=modal_features, fallback=fused_feat)
        if modal_features is None:
            return self._zero_loss(fallback=fused_feat)

        rgb_feat = modal_features.get("rgb")
        dsm_feat = modal_features.get("dsm")
        if rgb_feat is None or dsm_feat is None:
            return self._zero_loss(fallback=fused_feat)

        if self.cca_position == "pre_fusion":
            return self.cca_loss(rgb_feat, dsm_feat)

        if self.cca_position == "post_fusion":
            if fused_feat is None:
                return self._zero_loss(fallback=rgb_feat)
            loss_rgb = self.cca_loss(rgb_feat, fused_feat)
            loss_dsm = self.cca_loss(dsm_feat, fused_feat)
            return 0.5 * (loss_rgb + loss_dsm)

        return self._zero_loss(fallback=fused_feat)

    def encode_decode(self, rgb, modal_x, semantic_prior=None):
        ori_size = rgb.shape
        x_semantic, L_cons, low_L_cons, modal_features = self.backbone(
            rgb,
            modal_x,
            semantic_prior=semantic_prior
        )
        x_semantic = self._apply_stage_modules(x_semantic, semantic_prior=semantic_prior)
        fused_feat = x_semantic[-1] if x_semantic else None
        cca_loss = self._compute_cca_loss(fused_feat, modal_features)

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        return out_semantic, L_cons, low_L_cons, cca_loss

    def forward(self, rgb, modal_x):
        if modal_x.ndim == 3:
            modal_x = torch.unsqueeze(modal_x, dim=1)

        semantic_prior = self.prompt_semantic(rgb) if self.prompt_semantic is not None else None
        if isinstance(semantic_prior, (tuple, list)):
            semantic_prior = semantic_prior[0]

        outputs, L_cons, low_L_cons, cca_loss = self.encode_decode(
            rgb,
            modal_x,
            semantic_prior=semantic_prior
        )

        return outputs, L_cons, low_L_cons, semantic_prior, cca_loss
