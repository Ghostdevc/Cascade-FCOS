"""
models/cascade_fcos.py — Cascade FCOS with Torchvision FCOS baseline (Tier 1).

Architecture:
  Stage 1 : Torchvision FCOS (COCO-pretrained) — provides FPN features,
            cls_logits, bbox_regression, centerness.
  Stage 2 : FCM_1_to_2 (box-conditioned DCN) → RefinementHead → refined preds.
  Stage 3 : FCM_2_to_3 (box-conditioned DCN) → RefinementHead → refined preds.

The refinement heads (stage 2, 3) are lightweight (1 conv + 3 output convs) to
keep the inference-overhead bonus (<15%).
"""

import math
from typing import Dict, Any, List

import torch
import torch.nn as nn

from .torchvision_fcos import TorchvisionFCOSWrapper
from .refinement_module import LightweightFCM


class RefinementHead(nn.Module):
    """Lightweight head for refinement stages (2 and 3).

    Architecture: shared 3×3 conv + GroupNorm + ReLU → 3 prediction heads.
    Much smaller than torchvision's FCOS head (4 stacked convs per branch)
    to respect the <15% inference overhead bonus.

    Output convention matches torchvision FCOS:
      - bbox_regression: raw (l, t, r, b) in stride-normalised units
        (decoded by multiplying with FPN stride during inference)
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 20,
                 num_fpn_levels: int = 5):
        super().__init__()
        # Shared trunk: one 3×3 conv per branch (cls + reg) for efficiency
        self.cls_trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.reg_trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

        # Prediction heads
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.bbox_pred  = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Per-level learnable scales (FCOS v2 paper: exp(s_i * raw_pred))
        self.scales = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_fpn_levels)
        ])

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Focal-loss prior init for cls head
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, features: List[torch.Tensor]):
        cls_per_level, bbox_per_level, cent_per_level = [], [], []
        for i, feat in enumerate(features):
            cls_feat = self.cls_trunk(feat)
            reg_feat = self.reg_trunk(feat)

            cls_logits = self.cls_logits(cls_feat)
            # exp(scale * raw_pred) — positive bbox distances, smooth gradients
            bbox_pred = torch.exp(
                torch.clamp(self.scales[i] * self.bbox_pred(reg_feat).float(), max=4.6)
            ).to(feat.dtype)
            centerness = self.centerness(reg_feat)

            cls_per_level.append(cls_logits)
            bbox_per_level.append(bbox_pred)
            cent_per_level.append(centerness)
        return cls_per_level, bbox_per_level, cent_per_level


class CascadeFCOS(nn.Module):
    """Cascade FCOS — torchvision-FCOS baseline + 2 box-conditioned refinement stages.

    Tier 1 design (PDF Table 1):
        Pretrained COCO FCOS → frozen backbone + FPN.
        Stage 1: fine-tune the classification head only (VOC has 20 classes).
        Stage 2: train FCM_1_to_2 + RefinementHead 2.
        Stage 3: train FCM_2_to_3 + RefinementHead 3.

    Outputs (per stage) match the format expected by train.py and evaluate.py:
      stage_{1,2,3}     : (cls_per_level, bbox_per_level, cent_per_level)
      features_s1       : FPN features [P3..P7]
      bbox_preds_s1/s2  : detached bbox predictions for next-stage FCM
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Stage 1: torchvision FCOS wrapper (COCO-pretrained)
        self.baseline_fcos = TorchvisionFCOSWrapper(num_classes=num_classes,
                                                     pretrained=True)

        # Stage 2: FCM + RefinementHead
        self.fcm_1_to_2   = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_2 = RefinementHead(in_channels=256,
                                            num_classes=num_classes,
                                            num_fpn_levels=5)

        # Stage 3: FCM + RefinementHead
        self.fcm_2_to_3   = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_3 = RefinementHead(in_channels=256,
                                            num_classes=num_classes,
                                            num_fpn_levels=5)

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _detach(bbox_preds: List[torch.Tensor]) -> List[torch.Tensor]:
        return [bp.detach() for bp in bbox_preds]

    @staticmethod
    def _apply_fcm(fcm, features, bbox_preds_prev):
        return [fcm(f, bp) for f, bp in zip(features, bbox_preds_prev)]

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, max_stage: int = 3) -> Dict[str, Any]:
        """
        Args:
            images   : (B, 3, H, W), pre-normalised
            max_stage: only run up to this stage (1, 2, or 3) — memory saver

        Returns:
            dict (same shape as before) — keys: stage_1, stage_2, stage_3,
            features_s1, bbox_preds_s1, bbox_preds_s2.
            Stages above max_stage are None.
        """
        # ── Stage 1: torchvision FCOS ─────────────────────────────────────
        features_s1, cls_s1, bbox_s1, cent_s1 = self.baseline_fcos(images)

        # Decode torchvision raw bbox_regression to absolute pixel distances:
        # torchvision predicts stride-normalised offsets, we multiply by stride.
        # But for the FCM, RAW values are fine (they encode displacement direction).
        bbox_s1_detached = self._detach(bbox_s1)

        result = {
            "stage_1":       (cls_s1, bbox_s1, cent_s1),
            "features_s1":   features_s1,
            "bbox_preds_s1": bbox_s1_detached,
            "bbox_preds_s2": None,
            "stage_2":       None,
            "stage_3":       None,
        }
        if max_stage == 1:
            return result

        # ── Stage 2 ───────────────────────────────────────────────────────
        features_s2 = self._apply_fcm(self.fcm_1_to_2, features_s1, bbox_s1_detached)
        cls_s2, bbox_s2, cent_s2 = self.head_stage_2(features_s2)
        bbox_s2_detached = self._detach(bbox_s2)

        result["stage_2"]       = (cls_s2, bbox_s2, cent_s2)
        result["bbox_preds_s2"] = bbox_s2_detached
        if max_stage == 2:
            return result

        # ── Stage 3 ───────────────────────────────────────────────────────
        features_s3 = self._apply_fcm(self.fcm_2_to_3, features_s2, bbox_s2_detached)
        cls_s3, bbox_s3, cent_s3 = self.head_stage_3(features_s3)
        result["stage_3"] = (cls_s3, bbox_s3, cent_s3)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Convenience: stage-specific parameter groups
    # ─────────────────────────────────────────────────────────────────────

    def stage1_params(self):
        """Stage 1: fine-tune torchvision FCOS classification head only."""
        return self.baseline_fcos.baseline_head_params()

    def stage2_params(self):
        return list(self.fcm_1_to_2.parameters()) + list(self.head_stage_2.parameters())

    def stage3_params(self):
        return list(self.fcm_2_to_3.parameters()) + list(self.head_stage_3.parameters())