import torch
import torch.nn as nn
from .fcos_baseline import FCOSBackboneFPN, StageSpecificFCOSHead
from .refinement_module import LightweightFCM

class CascadeFCOS(nn.Module):
    """
    The master module for Cascade FCOS. 
    Wires the FPN, FCMs, and Heads together.
    """
    def __init__(self, num_classes=20):
        super().__init__()
        
        # 1. Base Feature Extractor
        self.backbone_fpn = FCOSBackboneFPN(out_channels=256)
        
        # 2. Stage 1 (Base Detection)
        self.head_stage_1 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)
        
        # 3. Stage 2 (1st Refinement)
        self.fcm_1_to_2 = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_2 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)
        
        # 4. Stage 3 (2nd Refinement)
        self.fcm_2_to_3 = LightweightFCM(in_channels=256, mid_channels=64)
        self.head_stage_3 = StageSpecificFCOSHead(in_channels=256, num_classes=num_classes)

    def forward(self, images):
        """
        images: A batched tensor of shape (B, C, H, W)
        Returns a dictionary of predictions per stage.
        """
        # Extract base FPN features [P3, P4, P5, P6, P7]
        features_s1 = self.backbone_fpn(images)
        
        # Stage 1 Predictions
        preds_s1 = self.head_stage_1(features_s1)
        
        # Warp features for Stage 2 using our LightweighFCM
        features_s2 = [self.fcm_1_to_2(f) for f in features_s1]
        preds_s2 = self.head_stage_2(features_s2)
        
        # Warp features for Stage 3
        features_s3 = [self.fcm_2_to_3(f) for f in features_s2]
        preds_s3 = self.head_stage_3(features_s3)
        
        return {
            "stage_1": preds_s1, # (cls_scores, bbox_preds, cent_scores)
            "stage_2": preds_s2,
            "stage_3": preds_s3,
            "features_s1": features_s1 # Returned so the TargetGenerator can map grid locations
        }