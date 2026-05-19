import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

class FCOSLoss(nn.Module):
    """
    Computes the composite loss for a single Cascade FCOS stage.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def compute_centerness_targets(self, reg_targets):
        """
        Calculates the theoretical centerness targets from ground truth distances.
        reg_targets: (N, 4) tensor containing (l, t, r, b)
        """
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        
        # Add 1e-6 to prevent division by zero
        centerness = (left_right.min(dim=-1)[0] / (left_right.max(dim=-1)[0] + 1e-6)) * \
                     (top_bottom.min(dim=-1)[0] / (top_bottom.max(dim=-1)[0] + 1e-6))
                     
        return torch.sqrt(centerness)

    def forward(self, cls_preds, reg_preds, centerness_preds, labels, reg_targets):
        """
        Args:
            cls_preds: (N, num_classes)
            reg_preds: (N, 4)
            centerness_preds: (N, 1)
            labels: (N,) ground truth labels (0 is background)
            reg_targets: (N, 4) ground truth bounding box distances
        """
        # 1. Prepare Masks
        pos_mask = labels > 0
        num_pos = pos_mask.sum().clamp(min=1.0) # Avoid division by zero
        
        # 2. Classification Loss (Focal Loss over ALL locations)
        # Convert labels to one-hot (excluding background class 0)
        num_classes = cls_preds.shape[1]
        labels_one_hot = F.one_hot(labels, num_classes=num_classes + 1)[:, 1:].float()
        
        # We use torchvision's highly optimized focal loss 
        loss_cls = ops.sigmoid_focal_loss(
            cls_preds, 
            labels_one_hot, 
            alpha=self.alpha, 
            gamma=self.gamma, 
            reduction="sum"
        ) / num_pos

        # If no positive samples, return 0 for reg and centerness
        if num_pos == 1.0 and pos_mask.sum() == 0:
            return loss_cls, cls_preds.sum() * 0, cls_preds.sum() * 0

        # 3. Filter Positive Samples for Regression and Centerness
        pos_reg_preds = reg_preds[pos_mask]
        pos_reg_targets = reg_targets[pos_mask]
        pos_centerness_preds = centerness_preds[pos_mask].view(-1)
        
        # Compute Centerness Targets
        pos_centerness_targets = self.compute_centerness_targets(pos_reg_targets)

        # 4. Centerness Loss (BCE)
        loss_centerness = F.binary_cross_entropy_with_logits(
            pos_centerness_preds, 
            pos_centerness_targets, 
            reduction="sum"
        ) / num_pos

        # 5. Regression Loss (GIoU)
        # Convert (l, t, r, b) distances back to a dummy (x1, y1, x2, y2) format 
        # relative to a 0,0 origin to leverage torchvision's generalized_box_iou_loss
        dummy_pred_boxes = torch.stack([
            -pos_reg_preds[:, 0], -pos_reg_preds[:, 1], 
             pos_reg_preds[:, 2],  pos_reg_preds[:, 3]
        ], dim=1)
        
        dummy_target_boxes = torch.stack([
            -pos_reg_targets[:, 0], -pos_reg_targets[:, 1], 
             pos_reg_targets[:, 2],  pos_reg_targets[:, 3]
        ], dim=1)

        # GIoU loss is weighted by the ground truth centerness 
        # (This forces the network to prioritize well-centered boxes)
        giou_loss_raw = 1 - ops.generalized_box_iou(dummy_pred_boxes, dummy_target_boxes).diag()
        loss_reg = (giou_loss_raw * pos_centerness_targets).sum() / num_pos

        return loss_cls, loss_reg, loss_centerness

# --- Usage Example ---
# loss_fn = FCOSLoss()
# l_cls, l_reg, l_cent = loss_fn(pred_logits, pred_boxes, pred_cent, target_labels, target_boxes)
# total_loss = l_cls + l_reg + l_cent