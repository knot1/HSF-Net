from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# DEFAULT_PROMPT_TEMPLATES = [
#     "an aerial image of {label}",
#     "{label} in a satellite image",
#     "{label} from above",
# ]

# PROMPT_BANK = {
#     "roads": [
#         "roads in an aerial image",
#         "streets from above",
#         "road networks in a satellite image",
#     ],
#     "buildings": [
#         "an aerial image of buildings",
#         "rooftops in a satellite image",
#         "urban buildings from above",
#     ],
#     "low veg.": [
#         "low vegetation in an aerial image",
#         "grassland from above",
#         "low vegetation in a satellite image",
#     ],
#     "trees": [
#         "trees in an aerial image",
#         "forest canopy from above",
#         "trees from above",
#     ],
#     "cars": [
#         "cars in a parking lot from above",
#         "cars in an aerial image",
#         "vehicles in a satellite image",
#     ],
#     "clutter": [
#         "cluttered urban areas from above",
#         "miscellaneous objects in a satellite image",
#         "mixed clutter from above",
#     ],
# }

# LABEL_ALIASES = {
#     "low veg.": "low vegetation",
#     "low veg": "low vegetation",
# }

DEFAULT_PROMPT_TEMPLATES = [
    "an aerial image of {label}",
    "{label} in a satellite image",
    "{label} from above",
]

# ==================== 专为 Ordos (矿区遥感) 定制的提示词库 ====================
PROMPT_BANK = {
    "road": [
        "dirt roads and paths in an aerial image",
        "mining roads from above",
        "road networks in a satellite image",
    ],
    "slope": [
        "steep mining slopes in an aerial image",
        "open-pit mine terraced slopes from above",
        "excavated soil slopes in a satellite image",
    ],
    "forest": [
        "trees and woods in an aerial image",
        "forest canopy from above",
        "dense vegetation and trees in a satellite image",
    ],
    "building": [
        "industrial buildings in an aerial image",
        "rooftops of mining facilities from above",
        "buildings and structures in a satellite image",
    ],
    "water": [
        "water bodies in an aerial image",
        "pools of water or lakes from above",
        "water surface in a satellite image",
    ],
    "clutter": [
        "cluttered mining areas from above",
        "miscellaneous objects and equipment in a satellite image",
        "mixed clutter from above",
    ],
}

# 添加一些别名容错，防止大小写或单复数没匹配上
LABEL_ALIASES = {
    "roads": "road",
    "buildings": "building",
    "trees": "forest",
    "water body": "water",
}


def _normalize_label(label: str) -> str:
    key = label.strip().lower()
    return LABEL_ALIASES.get(key, key)


def _build_prompts(label: str) -> List[str]:
    if label in PROMPT_BANK:
        return PROMPT_BANK[label]
    normalized = _normalize_label(label)
    if normalized in PROMPT_BANK:
        return PROMPT_BANK[normalized]
    return [template.format(label=normalized) for template in DEFAULT_PROMPT_TEMPLATES]


class PromptSemanticPrior(nn.Module):
    def __init__(
        self,
        class_labels: list[str],
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        image_size: int = 224,
    ):
        super().__init__()
        self.class_labels = list(class_labels)
        self.image_size = image_size

        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="/home/wsj/.cache/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/open_clip_pytorch_model.bin",
        )
        clip_model.eval()
        for param in clip_model.parameters():
            param.requires_grad = False
        self.clip_model = clip_model
        self.tokenizer = open_clip.get_tokenizer(model_name)

        prompt_lists = [_build_prompts(label) for label in self.class_labels]
        max_prompts = max(len(prompts) for prompts in prompt_lists)
        prompt_features = []
        prompt_mask = torch.zeros(len(prompt_lists), max_prompts)

        with torch.no_grad():
            for idx, prompts in enumerate(prompt_lists):
                tokens = self.tokenizer(prompts)
                text_features = self.clip_model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)
                if len(prompts) < max_prompts:
                    pad = torch.zeros(max_prompts - len(prompts), text_features.size(-1))
                    text_features = torch.cat([text_features, pad], dim=0)
                prompt_features.append(text_features)
                prompt_mask[idx, : len(prompts)] = 1.0

        prompt_features = torch.stack(prompt_features, dim=0)
        prompt_counts = prompt_mask.sum(dim=1).clamp(min=1.0)

        self.register_buffer("prompt_features", prompt_features)
        self.register_buffer("prompt_mask", prompt_mask)
        self.register_buffer("prompt_counts", prompt_counts)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.size(1) > 3:
            images = images[:, :3, :, :]
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        images = images.clamp(0.0, 1.0)
        return (images - self.clip_mean) / self.clip_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = self._preprocess_images(images)
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            prompt_features = self.prompt_features
            sim = torch.einsum("bd,kpd->bkp", image_features, prompt_features)
            sim = sim * self.prompt_mask.unsqueeze(0)
            sim_mean = sim.sum(dim=2) / self.prompt_counts.unsqueeze(0)

            logit_scale = self.clip_model.logit_scale.exp()
            logits = sim_mean * logit_scale
            semantic_prior = F.softmax(logits, dim=1)

        return semantic_prior
