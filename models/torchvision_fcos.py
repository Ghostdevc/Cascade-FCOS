"""
models/torchvision_fcos.py

Torchvision-based FCOS wrapper for Cascade FCOS (Tier 1).

Uses torchvision.models.detection.fcos_resnet50_fpn (COCO-pretrained) as the
baseline. The wrapper exposes:
  - FPN features [P3..P7] for the cascade's FCM modules
  - Raw (cls_logits, bbox_regression, bbox_ctrness) per FPN level
    in the FCOS-native format (l,t,r,b stride-normalised regression)

We rebuild the classification subnet's final conv with num_classes=20 for VOC
(the COCO model has 91 classes), keeping all other COCO-pretrained weights.
"""

import torch.nn.functional as F

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.fcos import (
    FCOS,
    FCOSClassificationHead,
)


class TorchvisionFCOSWrapper(nn.Module):
    """Wraps torchvision's FCOS to expose FPN features and raw predictions.

    Torchvision FCOS's head.forward returns a dict with 'cls_logits',
    'bbox_regression', and 'bbox_ctrness' — flattened across FPN levels.
    We instead need per-level (B, C, H_i, W_i) tensors for the cascade.

    The trick: we re-implement head application here so we keep the per-level
    structure. Same conv weights, just no flatten/concat at the end.
    """

    def __init__(self, num_classes: int = 20, pretrained: bool = True):
        super().__init__()

        # ── 1. Load COCO-pretrained FCOS ──────────────────────────────────
        if pretrained:
            from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
            base_fcos = torchvision.models.detection.fcos_resnet50_fpn(
                weights=FCOS_ResNet50_FPN_Weights.COCO_V1
            )
            print(f"[Model] Loaded COCO-pretrained FCOS (91 classes)")
        else:
            base_fcos = torchvision.models.detection.fcos_resnet50_fpn(
                weights=None, num_classes=num_classes + 1
            )

        # ── 2. Extract components ─────────────────────────────────────────
        self.backbone = base_fcos.backbone   # ResNet50+FPN
        self.cls_head = base_fcos.head.classification_head
        self.reg_head = base_fcos.head.regression_head

        # ── 3. Replace classification head's final conv (80 → 20 classes) ─
        # Torchvision's COCO model uses 91 (background + 90 COCO classes)
        # FCOS doesn't use a background class in cls (uses focal loss),
        # so we replace with num_classes=20.
        in_channels = 256
        num_anchors = self.cls_head.num_anchors  # 1 for FCOS (anchor-free)
        self.cls_head.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes,
            kernel_size=3, stride=1, padding=1
        )
        # Focal-loss prior bias init
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_head.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_head.cls_logits.bias, bias_value)
        # Update num_classes attribute
        self.cls_head.num_classes = num_classes
        self.num_classes = num_classes
        print(f"[Model] Replaced cls_logits: 91 → {num_classes} classes")

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W) — already normalised by datasets/pascal_voc.py

        Returns:
            features        : List[Tensor(B, 256, H_i, W_i)] — FPN levels [P3..P7]
            cls_per_level   : List[Tensor(B, num_classes, H_i, W_i)]
            bbox_per_level  : List[Tensor(B, 4, H_i, W_i)]   — raw (l,t,r,b)
            cent_per_level  : List[Tensor(B, 1, H_i, W_i)]
        """
        # ── Backbone + FPN ────────────────────────────────────────────────
        # Torchvision backbone returns OrderedDict {'0':P3, '1':P4, ..., 'pool':P7}
        feat_dict = self.backbone(images)
        features = list(feat_dict.values())  # [P3, P4, P5, P6, P7]

        # ── Apply classification & regression sub-heads per level ─────────
        cls_per_level, bbox_per_level, cent_per_level = [], [], []
        for feat in features:
            # Classification subnet
            cls_feat = self.cls_head.conv(feat)
            cls_logits = self.cls_head.cls_logits(cls_feat)
            cls_per_level.append(cls_logits)

            # Regression subnet (returns bbox + centerness)
            reg_feat = self.reg_head.conv(feat)
            bbox_reg = F.relu(self.reg_head.bbox_reg(reg_feat))
            ctrness  = self.reg_head.bbox_ctrness(reg_feat)
            bbox_per_level.append(bbox_reg)
            cent_per_level.append(ctrness)

        return features, cls_per_level, bbox_per_level, cent_per_level

    # ── Utility: parameter groups for staged training ──────────────────────
    def baseline_head_params(self):
        """Stage 1 trainable params: only cls_head (regression frozen for VOC fine-tune)."""
        return list(self.cls_head.parameters())

    def backbone_fpn_params(self):
        """Backbone + FPN params — frozen in all stages of Tier 1."""
        return list(self.backbone.parameters())