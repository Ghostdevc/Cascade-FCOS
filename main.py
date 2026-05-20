"""
main.py — Entry point for staged Cascade FCOS training (Tier 1, torchvision baseline).

Tier 1 strategy (PDF Table 1):
    Load COCO-pretrained FCOS → backbone + FPN + regression head are FROZEN.
    Stage 1: fine-tune classification head only (80→20 classes adaptation).
    Stage 2: train FCM_1_to_2 + head_stage_2.
    Stage 3: train FCM_2_to_3 + head_stage_3.

Usage
-----
# Stage 1 — fine-tune torchvision FCOS cls head for VOC (20 classes)
python main.py --config configs/fcos_baseline.yaml --stage 1

# Stage 2 — first refinement stage
python main.py --config configs/refinement_k1.yaml --stage 2 \\
               --resume logs/checkpoints/stage1_epoch003.pth

# Stage 3 — second refinement stage
python main.py --config configs/refinement_k2.yaml --stage 3 \\
               --resume logs/checkpoints/stage2_epoch004.pth
"""

import argparse
import json
import os

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from datasets.pascal_voc import build_trainval_dataset, custom_collate_fn
from models.cascade_fcos import CascadeFCOS
from models.loss import FCOSLoss
from models.target_generator import DynamicFCOSTargetGenerator
from train import load_checkpoint, save_checkpoint, train_one_epoch


# ── Freeze helpers ────────────────────────────────────────────────────────
def _set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad = value


def set_trainable(model: CascadeFCOS, stage: int):
    """Configure requires_grad for Tier-1 staged training.

    The torchvision FCOS baseline contains: backbone (ResNet50), FPN,
    classification_head, regression_head. We freeze everything from the
    pretrained COCO model except what the current stage needs.

    Stage 1: train only baseline_fcos.cls_head (newly initialised for VOC).
             backbone + FPN + reg_head stay frozen with COCO weights.
    Stage 2: freeze ALL stage-1 modules; train fcm_1_to_2 + head_stage_2.
    Stage 3: freeze stages 1+2; train fcm_2_to_3 + head_stage_3.
    """
    # Step 1: freeze everything
    _set_requires_grad(model, False)

    # Step 2: selectively unfreeze the current stage's trainable modules
    if stage == 1:
        # Only the new classification head (20 classes) is trainable
        _set_requires_grad(model.baseline_fcos.cls_head, True)
    elif stage == 2:
        _set_requires_grad(model.fcm_1_to_2,   True)
        _set_requires_grad(model.head_stage_2, True)
    elif stage == 3:
        _set_requires_grad(model.fcm_2_to_3,   True)
        _set_requires_grad(model.head_stage_3, True)

    # Log trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"[Freeze] Stage {stage} (Tier 1) | "
        f"Trainable: {trainable:,} / {total:,} params "
        f"({100*trainable/total:.1f}%)"
    )


# ── Optimizer + Scheduler ─────────────────────────────────────────────────
def build_optimizer(model, cfg):
    """Build AdamW over trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=cfg["lr"],
                       weight_decay=cfg.get("weight_decay", 1e-4))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    milestones = cfg.get("lr_milestones", [8, 11])
    gamma      = cfg.get("lr_gamma", 0.1)
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    stage = args.stage
    assert stage in (1, 2, 3), "--stage must be 1, 2, or 3"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Stage {stage} | Device: {device} | Config: {args.config}")

    # Logging
    os.makedirs("logs", exist_ok=True)
    run_name = cfg.get("run_name", f"stage{stage}")
    log_file = os.path.join("logs", f"{run_name}.log")
    with open(log_file, "w") as fh:
        fh.write(f"--- Cascade FCOS Stage {stage} (Tier 1, torchvision): {run_name} ---\n")
        fh.write(json.dumps(cfg, indent=2) + "\n\n")

    # Dataset — VOC2007+2012 trainval (16,551 images)
    print("[Data] Building combined VOC2007+2012 trainval dataset …")
    train_dataset = build_trainval_dataset(cfg["data_dir"], augment=True)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    print("[Model] Building CascadeFCOS (torchvision baseline) …")
    model = CascadeFCOS(num_classes=20).to(device)

    # Freeze setup per stage
    set_trainable(model, stage)

    # Load previous-stage checkpoint (required for stages 2 and 3)
    start_epoch = 0
    if args.resume:
        load_checkpoint(model, args.resume, device=device)
        # Re-apply freeze after loading (load_state_dict doesn't change requires_grad)
        set_trainable(model, stage)
    elif stage > 1:
        print(
            f"[Warning] Stage {stage} started without --resume checkpoint. "
            "Stages 2/3 should build on a trained stage 1/2."
        )

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    target_gen = DynamicFCOSTargetGenerator()
    loss_fn    = FCOSLoss(alpha=0.25, gamma=2.0)

    epochs = cfg.get("epochs", 3)
    print(f"[Train] Stage {stage} | Epochs: {epochs} | LR: {cfg['lr']}")

    for epoch in range(start_epoch + 1, epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, target_gen, loss_fn,
            device, epoch, stage, log_file,
        )
        scheduler.step()

        log_entry = {
            "epoch": epoch,
            "stage": stage,
            "avg_loss": round(avg_loss, 6),
            "lr": scheduler.get_last_lr()[0],
        }
        summary_msg = f"=== Stage {stage} Epoch {epoch} | avg_loss={avg_loss:.4f} ===\n"
        print(summary_msg.strip())
        with open(log_file, "a") as fh:
            fh.write(summary_msg)
            fh.write(json.dumps(log_entry) + "\n")

        save_checkpoint(
            epoch, model, optimizer, avg_loss,
            stage=stage,
            save_dir=cfg.get("checkpoint_dir", "logs/checkpoints"),
        )

    print(f"[Done] Stage {stage} training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade FCOS — Tier 1 Staged Training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to previous-stage checkpoint")
    args = parser.parse_args()
    main(args)