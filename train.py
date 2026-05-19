"""
train.py — Staged training loop for Cascade FCOS.

Step 4 changes vs. original
----------------------------
- train_one_epoch accepts a `stage` arg (1,2,3) controlling:
    * Which stage's predictions/loss to use.
    * Whether prev_bbox_preds are passed for re-anchored target generation.
    * The center-sampling radius (1.5 -> 1.0 -> 0.75, progressively tighter).
- Only the current stage's loss is backpropagated.
- save_checkpoint / load_checkpoint support chain-loading between stages.
"""

import os

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from models.loss import FCOSLoss
from models.target_generator import DynamicFCOSTargetGenerator

# Center-sampling radii per stage.
# Stage 1: r=1.5 is FCOS paper optimum (Table 6, TPAMI 2022).
# Stages 2/3: progressively tighter — analogous to Cascade R-CNN's increasing
# IoU thresholds (Cai & Vasconcelos, CVPR 2018, Sec. 3.1).
STAGE_CENTER_RADIUS = {1: 1.5, 2: 1.0, 3: 0.75}


def pad_batch_images(images):
    """Pad variable-size tensors to (max_H, max_W) in the batch."""
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    return torch.stack([
        F.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]))
        for img in images
    ])


def _flatten_level_preds(preds_per_level, channels):
    """Flatten List[Tensor(B,C,H_i,W_i)] -> Tensor(B*N_total, channels)."""
    return torch.cat(
        [p.permute(0, 2, 3, 1).reshape(-1, channels) for p in preds_per_level],
        dim=0,
    )


def _extract_prev_bbox_per_image(bbox_preds_prev, img_idx):
    """Extract List[Tensor(N_i, 4)] for one image from batch-level preds."""
    return [
        bp[img_idx].permute(1, 2, 0).reshape(-1, 4)
        for bp in bbox_preds_prev
    ]


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    target_generator,
    loss_fn,
    device,
    epoch,
    stage,
    log_file,
):
    """Train one epoch for cascade stage `stage` (1, 2, or 3).

    Frozen parameters (set by main.py) receive no optimizer updates.
    detach() calls in cascade_fcos.py prevent gradients from propagating
    backward through frozen upstream stages.
    """
    assert stage in (1, 2, 3)
    model.train()
    total_loss    = 0.0
    center_radius = STAGE_CENTER_RADIUS[stage]
    stage_key     = f"stage_{stage}"
    prev_bbox_key = {2: "bbox_preds_s1", 3: "bbox_preds_s2"}.get(stage, None)

    for batch_idx, (images_list, targets_list) in enumerate(dataloader):
        images = pad_batch_images(images_list).to(device)
        optimizer.zero_grad()

        # Full forward (upstream stages are frozen; detach() isolates gradients)
        cascade_preds = model(images)

        # Static FPN grid — shared across all images in the padded batch
        base_locations = target_generator.compute_locations(cascade_preds["features_s1"])

        # Current stage predictions
        cls_preds, reg_preds, cent_preds = cascade_preds[stage_key]
        flat_cls  = _flatten_level_preds(cls_preds,  20)
        flat_reg  = _flatten_level_preds(reg_preds,   4)
        flat_cent = _flatten_level_preds(cent_preds,  1)

        # Per-image target generation (with re-anchoring for stages 2/3)
        batch_labels, batch_reg_tgts = [], []
        for img_idx, target in enumerate(targets_list):
            gt_boxes  = target["boxes"].to(device)
            gt_labels = target["labels"].to(device)

            prev_bbox = None
            if prev_bbox_key is not None:
                prev_bbox = _extract_prev_bbox_per_image(
                    cascade_preds[prev_bbox_key], img_idx
                )

            labels, reg_tgts = target_generator.generate_targets_for_image(
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
    """Save a stage-labelled checkpoint. Filename: stage{N}_epoch{EEE}.pth."""
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
    """Load checkpoint; optionally restore optimizer state. Returns (epoch, loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(
        f"[Checkpoint] Loaded stage={ckpt.get('stage','?')} "
        f"epoch={ckpt['epoch']} from {path}"
    )
    return ckpt["epoch"], ckpt["loss"]
