import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from .fcos_baseline import FCOSBackboneFPN, StageSpecificFCOSHead
from .refinement_module import LightweightFCM


class CascadeFCOS(nn.Module):
    """Cascade FCOS — FCOS-faithful iterative refinement via box-conditioned FCMs.

    Fixes vs. original (Bugs 1 & 2)
    ---------------------------------
    Bug 1: FCM offsets were derived only from the feature map; box predictions
           were never fed in.  Fixed in LightweightFCM.forward(x, bbox_pred_prev).

    Bug 2: The forward loop ignored preds_s1 / preds_s2 entirely when building
           features for the next stage.  Fixed here: we extract bbox_preds from
           each stage, DETACH them (PDF Sec. 2.5 — no gradient flows backwards
           through the cascade), reshape to (B, 4, H, W) per level, and pass
           them to the next FCM.

    Information flow
    ----------------
    FPN features  ──►  Head-1  ──►  (cls₁, bbox₁, cent₁)
                          │
                          ▼  bbox₁.detach()
                        FCM₁₋₂(feat, bbox₁) ──►  refined_feat₂
                                                       │
                                              Head-2  ──►  (cls₂, bbox₂, cent₂)
                                                       │
                                                       ▼  bbox₂.detach()
                                                  FCM₂₋₃(refined_feat₂, bbox₂)
                                                          ──►  refined_feat₃
                                                                    │
                                                           Head-3  ──►  (cls₃, bbox₃, cent₃)

    The `.detach()` calls are crucial:
      - They stop gradients from stage t+1 affecting stage t's weights.
      - Together with staged training (train.py --stage flag), they allow
        sequential optimisation: stage 1 is fully trained first, then frozen.

    Forward outputs
    ---------------
    The forward method returns a dict containing:
      "stage_1", "stage_2", "stage_3":
          Each is a tuple (cls_scores, bbox_preds, centerness_scores) where
          each element is a List[Tensor] over the 5 FPN levels.

      "features_s1":
          The raw FPN feature maps [P3..P7] before any refinement.
          Used by DynamicFCOSTargetGenerator.compute_locations() in train.py
          to obtain the base (static) grid locations for stage-1 target
          assignment.

      "bbox_preds_s1", "bbox_preds_s2":
          Per-level bbox predictions from stages 1 and 2, reshaped to
          List[Tensor(B, 4, H_i, W_i)] and ALREADY DETACHED.
          Used by train.py to call target_generator.compute_refined_locations()
          for stages 2 and 3, respectively.
          Returning them here avoids re-running the head in train.py.
    """

    def __init__(self, num_classes: int = 20):
        super().__init__()

        # ── Shared backbone + FPN ─────────────────────────────────────────
        self.backbone_fpn = FCOSBackboneFPN(out_channels=256)

        # ── Stage 1: baseline FCOS head ───────────────────────────────────
        self.head_stage_1 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)

        # ── Stage 2: FCM + head ───────────────────────────────────────────
        self.fcm_1_to_2   = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_2 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)

        # ── Stage 3: FCM + head ───────────────────────────────────────────
        self.fcm_2_to_3   = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_3 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detach_bbox_preds(bbox_preds: List[torch.Tensor]) -> List[torch.Tensor]:
        """Detach per-level bbox predictions before feeding to the next FCM.

        Per PDF Sec. 2.5: gradients must not propagate backwards through the
        cascade chain at inference or joint-training time (staged training
        enforces this implicitly via frozen weights, but detach() is still
        correct and explicit).
        """
        return [bp.detach() for bp in bbox_preds]

    @staticmethod
    def _apply_fcm_per_level(
        fcm: LightweightFCM,
        features: List[torch.Tensor],
        bbox_preds_prev: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Apply a single FCM to all 5 FPN levels.

        Args:
            fcm            : the LightweightFCM module for this transition.
            features       : List[Tensor(B, 256, H_i, W_i)] — current features.
            bbox_preds_prev: List[Tensor(B, 4, H_i, W_i)] — detached box preds.

        Returns:
            List[Tensor(B, 256, H_i, W_i)] — refined features for next stage.
        """
        return [
            fcm(feat, bbox_prev)
            for feat, bbox_prev in zip(features, bbox_preds_prev)
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            images: Tensor(B, 3, H, W) — normalised input batch.

        Returns:
            dict with keys:
              "stage_1"       : (cls₁, bbox₁, cent₁)  — per-level lists
              "stage_2"       : (cls₂, bbox₂, cent₂)
              "stage_3"       : (cls₃, bbox₃, cent₃)
              "features_s1"   : List[Tensor] — raw FPN features (for locations)
              "bbox_preds_s1" : List[Tensor] — bbox predictions from stage 1,
                                detached, shape (B, 4, H_i, W_i) per level.
                                Used to compute refined locations for stage 2.
              "bbox_preds_s2" : List[Tensor] — same for stage 2 → stage 3.
        """
        # ── Backbone + FPN ────────────────────────────────────────────────
        features_s1 = self.backbone_fpn(images)
        # features_s1: [P3, P4, P5, P6, P7], each (B, 256, H_i, W_i)

        # ── Stage 1 ───────────────────────────────────────────────────────
        cls_s1, bbox_s1, cent_s1 = self.head_stage_1(features_s1)
        # Each is a list of 5 tensors over FPN levels.

        # Detach stage-1 boxes before passing to FCM (Fix: Bug 2 + PDF Sec. 2.5)
        bbox_s1_detached = self._detach_bbox_preds(bbox_s1)

        # ── Stage 2 ───────────────────────────────────────────────────────
        # FCM warps features_s1 towards the box centers predicted by stage 1.
        features_s2 = self._apply_fcm_per_level(self.fcm_1_to_2, features_s1, bbox_s1_detached)
        cls_s2, bbox_s2, cent_s2 = self.head_stage_2(features_s2)

        # Detach stage-2 boxes
        bbox_s2_detached = self._detach_bbox_preds(bbox_s2)

        # ── Stage 3 ───────────────────────────────────────────────────────
        features_s3 = self._apply_fcm_per_level(self.fcm_2_to_3, features_s2, bbox_s2_detached)
        cls_s3, bbox_s3, cent_s3 = self.head_stage_3(features_s3)

        return {
            # Predictions per stage (used for loss computation in train.py)
            "stage_1": (cls_s1, bbox_s1, cent_s1),
            "stage_2": (cls_s2, bbox_s2, cent_s2),
            "stage_3": (cls_s3, bbox_s3, cent_s3),

            # Raw FPN features — needed by DynamicFCOSTargetGenerator.compute_locations()
            # to build the static stage-1 grid.
            "features_s1": features_s1,

            # Detached bbox preds — needed by train.py to call
            # target_generator.compute_refined_locations() for stages 2 & 3.
            # Already detached here so train.py does not need to call .detach() again.
            "bbox_preds_s1": bbox_s1_detached,
            "bbox_preds_s2": bbox_s2_detached,
        }

    # ------------------------------------------------------------------
    # Convenience: stage-specific parameter groups (for staged training)
    # ------------------------------------------------------------------

    def stage1_params(self):
        """Parameters trained in Stage 1 (backbone+FPN frozen in Tier 2)."""
        return list(self.head_stage_1.parameters())

    def stage2_params(self):
        """Parameters trained in Stage 2 (stage 1 frozen)."""
        return (
            list(self.fcm_1_to_2.parameters()) +
            list(self.head_stage_2.parameters())
        )

    def stage3_params(self):
        """Parameters trained in Stage 3 (stages 1+2 frozen)."""
        return (
            list(self.fcm_2_to_3.parameters()) +
            list(self.head_stage_3.parameters())
        )
