"""
train.py — Staged training loop for Cascade FCOS (Tier 1, torchvision baseline).

Key changes for Tier 1:
- Stage 1 uses torchvision FCOS predictions. Torchvision regresses
  stride-normalised (l,t,r,b), so we multiply by stride to get pixel units
  before computing loss (target generator produces pixel-unit targets).
- Stages 2/3 use our RefinementHead which already outputs exp(scale*raw)
  in pixel units, no scaling needed.
"""

import os

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from models.loss import FCOSLoss
from models.target_generator import DynamicFCOSTargetGenerator

# Center-sampling radii per stage
STAGE_CENTER_RADIUS = {1: 1.5, 2: 1.0, 3: 0.75}

# FPN strides (P3..P7)
FPN_STRIDES = [8, 16, 32, 64, 128]


def pad_batch_images(images):
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    return torch.stack([
        F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        for img in images
    ])


def _flatten_level_preds(preds_per_level, channels):
    return torch.cat(
        [p.permute(0, 2, 3, 1).reshape(-1, channels) for p in preds_per_level],
        dim=0,
    )


def _flatten_level_preds_with_stride(bbox_per_level, channels=4):
    """Same as _flatten_level_preds but multiplies each level's regression
    output by its FPN stride, converting stride-normalised → pixel units.

    Used ONLY for Stage 1 (torchvision FCOS native output is stride-normalised).
    Stages 2/3 use our RefinementHead which already outputs pixel units.
    """
    flat = []
    for p, stride in zip(bbox_per_level, FPN_STRIDES):
        # p: (B, 4, H, W)
        p_scaled = p * stride
        flat.append(p_scaled.permute(0, 2, 3, 1).reshape(-1, channels))
    return torch.cat(flat, dim=0)


def _extract_prev_bbox_per_image(bbox_preds_prev, img_idx, scale_with_stride=False):
    """Extract per-image, per-level prev bbox preds.

    Args:
        scale_with_stride: True if previous-stage predictions are stride-normalised
                           (i.e. coming from torchvision FCOS stage 1).
    """
    out = []
    for lvl, bp in enumerate(bbox_preds_prev):
        # bp: (B, 4, H, W)
        bp_img = bp[img_idx].permute(1, 2, 0).reshape(-1, 4)
        if scale_with_stride:
            bp_img = bp_img * FPN_STRIDES[lvl]
        out.append(bp_img)
    return out


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    target_gen,
    loss_fn,
    device,
    epoch,
    stage,
    log_file,
):
    """Train one epoch for the specified stage (1, 2, or 3)."""
    assert stage in (1, 2, 3)
    model.train()
    total_loss    = 0.0
    center_radius = STAGE_CENTER_RADIUS[stage]
    stage_key     = f"stage_{stage}"
    prev_bbox_key = {2: "bbox_preds_s1", 3: "bbox_preds_s2"}.get(stage, None)

    # Stage 1: torchvision raw output is stride-normalised, must scale up.
    # Stage 2,3: our RefinementHead outputs pixel units already.
    current_stage_needs_stride_scale = (stage == 1)
    # For prev_bbox in stage 2 (coming from stage 1), need to scale up.
    # For prev_bbox in stage 3 (coming from stage 2), already in pixels.
    prev_stage_needs_stride_scale = (stage == 2)

    for batch_idx, (images_list, targets_list) in enumerate(dataloader):
        images = pad_batch_images(images_list).to(device)
        optimizer.zero_grad()

        # Forward (up to max_stage=stage, saves memory)
        cascade_preds = model(images, max_stage=stage)

        # Static FPN grid (shared across batch — images are padded equally)
        base_locations = target_gen.compute_locations(cascade_preds["features_s1"])

        # Current stage predictions
        cls_preds, reg_preds, cent_preds = cascade_preds[stage_key]
        flat_cls  = _flatten_level_preds(cls_preds, cls_preds[0].shape[1])
        flat_cent = _flatten_level_preds(cent_preds, 1)

        # Regression: scale stride-normalised → pixel units for Stage 1
        if current_stage_needs_stride_scale:
            flat_reg = _flatten_level_preds_with_stride(reg_preds)
        else:
            flat_reg = _flatten_level_preds(reg_preds, 4)

        # Per-image target generation (with re-anchoring for stages 2/3)
        batch_labels, batch_reg_tgts = [], []
        for img_idx, target in enumerate(targets_list):
            gt_boxes  = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)

            prev_bbox = None
            if prev_bbox_key is not None:
                prev_bbox = _extract_prev_bbox_per_image(
                    cascade_preds[prev_bbox_key], img_idx,
                    scale_with_stride=prev_stage_needs_stride_scale,
                )

            labels, reg_tgts = target_gen.generate_targets_for_image(
                base_locations, gt_boxes, gt_labels,
                center_radius=center_radius,
                prev_bbox_preds_per_level=prev_bbox,
            )
            batch_labels.append(labels)
            batch_reg_tgts.append(reg_tgts)

        flat_labels   = torch.cat(batch_labels,   dim=0)
        flat_reg_tgts = torch.cat(batch_reg_tgts, dim=0)

        l_cls, l_reg, l_cent = loss_fn(
            flat_cls, flat_reg, flat_cent, flat_labels, flat_reg_tgts
        )
        batch_loss = l_cls + l_reg + l_cent

        batch_loss.backward()
        nn_utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=3.0,
        )
        optimizer.step()

        total_loss += batch_loss.item()

        if batch_idx % 20 == 0:
            msg = (
                f"[Stage {stage}] Epoch {epoch} | "
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss {batch_loss.item():.4f} "
                f"(cls={l_cls.item():.3f} reg={l_reg.item():.3f} cent={l_cent.item():.3f})"
            )
            print(msg)
            with open(log_file, "a") as fh:
                fh.write(msg + "\n")

    return total_loss / max(len(dataloader), 1)


def save_checkpoint(epoch, model, optimizer, loss, stage, save_dir="logs/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"stage{stage}_epoch{epoch:03d}.pth")
    torch.save({
        "epoch":                epoch,
        "stage":                stage,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"[Checkpoint] Saved -> {path}")
    return path


def load_checkpoint(model, path, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    # strict=False because stage-2 checkpoint may not have stage_3 keys, etc.
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"[Checkpoint] Optimizer state mismatch (expected when changing stages): {e}")
    print(
        f"[Checkpoint] Loaded stage={ckpt.get('stage','?')} "
        f"epoch={ckpt['epoch']} from {path}"
    )
    return ckpt["epoch"], ckpt["loss"]