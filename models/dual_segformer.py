from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def denormalize_dsm(x):
    mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)
    std = torch.tensor([0.5]).view(1, 1, 1, 1).to(x.device)
    return x * std + mean


def denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return x * std + mean


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for _, m in feature.named_modules():
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


@dataclass
class ModelConfig:
    use_safr: bool = True
    use_caf: bool = True
    use_vlsp: bool = True
    inject_stage2: bool = False
    inject_stage3: bool = False
    inject_stage4: bool = True

    @classmethod
    def from_cfg(cls, cfg):
        if cfg is None:
            return cls()
        defaults = cls()
        use_safr = getattr(cfg, "use_safr", defaults.use_safr)
        use_caf = getattr(cfg, "use_caf", defaults.use_caf)
        use_vlsp = getattr(cfg, "use_vlsp", defaults.use_vlsp)

        vlsp_cfg = getattr(cfg, "vlsp", None)
        inject_stage2 = getattr(vlsp_cfg, "inject_stage2", defaults.inject_stage2) if vlsp_cfg is not None else defaults.inject_stage2
        inject_stage3 = getattr(vlsp_cfg, "inject_stage3", defaults.inject_stage3) if vlsp_cfg is not None else defaults.inject_stage3
        inject_stage4 = getattr(vlsp_cfg, "inject_stage4", defaults.inject_stage4) if vlsp_cfg is not None else defaults.inject_stage4
        return cls(
            use_safr=use_safr,
            use_caf=use_caf,
            use_vlsp=use_vlsp,
            inject_stage2=inject_stage2,
            inject_stage3=inject_stage3,
            inject_stage4=inject_stage4,
        )


class Baseline(nn.Module):
    def __init__(self, cfg=None, num_classes=None, norm_layer=nn.BatchNorm2d, in_chans=None, class_labels=None):
        super(Baseline, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            self.in_chans = [3, 1]

        self.model_cfg = ModelConfig.from_cfg(cfg)

        if cfg.backbone == 'mit_b5':
            from .encoder import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)
        elif cfg.backbone == 'mit_b4':
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)
        elif cfg.backbone == 'mit_b2':
            from .encoder import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)
        elif cfg.backbone == 'mit_b1':
            from .encoder import mit_b1 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)
        elif cfg.backbone == 'mit_b0':
            from .encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)
            self.channels = [32, 64, 160, 256]
        else:
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans, num_classes=num_classes,
                                     fuse_cfg=self.model_cfg)

        from .Seg_head import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)

        self.requires_semantic_prior = self.model_cfg.use_vlsp or self.model_cfg.use_safr
        self.prompt_semantic = None
        prompt_cfg = getattr(cfg, "prompt_semantic", None)
        if prompt_cfg is not None and getattr(prompt_cfg, "enabled", False) and self.requires_semantic_prior:
            if class_labels is None:
                class_labels = [str(idx) for idx in range(num_classes)]
            from .prompt_semantic import PromptSemanticPrior
            self.prompt_semantic = PromptSemanticPrior(
                class_labels=class_labels,
                model_name=prompt_cfg.model_name,
                pretrained=prompt_cfg.pretrained,
                image_size=prompt_cfg.image_size,
            )

        self.init_weights(cfg, pretrained=cfg.pretrained_backbone)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x, semantic_prior=None):
        ori_size = rgb.shape
        x_semantic, L_cons, low_L_cons = self.backbone(
            rgb,
            modal_x,
            semantic_prior=semantic_prior
        )

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        return out_semantic, L_cons, low_L_cons

    def forward(self, rgb, modal_x):
        if modal_x.ndim == 3:
            modal_x = torch.unsqueeze(modal_x, dim=1)

        semantic_prior = None
        if self.prompt_semantic is not None and self.requires_semantic_prior:
            semantic_prior = self.prompt_semantic(rgb)
            if isinstance(semantic_prior, (tuple, list)):
                semantic_prior = semantic_prior[0]

        outputs, L_cons, low_L_cons = self.encode_decode(
            rgb,
            modal_x,
            semantic_prior=semantic_prior
        )

        return outputs, L_cons, low_L_cons, semantic_prior
