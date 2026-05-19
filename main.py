"""
main.py — Entry point for staged Cascade FCOS training.

Usage
-----
# Stage 1: train FCOS head (backbone+FPN frozen, Tier 2)
python main.py --config configs/fcos_baseline.yaml --stage 1

# Stage 2: load stage-1 checkpoint, train FCM1 + Head2
python main.py --config configs/refinement_k1.yaml --stage 2 \
               --resume logs/checkpoints/stage1_epoch012.pth

# Stage 3: load stage-2 checkpoint, train FCM2 + Head3
python main.py --config configs/refinement_k2.yaml --stage 3 \
               --resume logs/checkpoints/stage2_epoch008.pth

Step 4 changes vs. original main.py
-------------------------------------
- --stage {1,2,3} CLI argument selects which cascade stage to train.
- --config <yaml> drives all hyperparameters (lr, epochs, batch_size, tier, …).
- --resume <path> loads a previous-stage checkpoint before training starts.
- Freeze logic: set_trainable(model, stage, tier) sets requires_grad correctly.
  Only the current stage's parameters are passed to the optimizer, so frozen
  weights truly receive no gradient updates (verified by optimizer param_groups).
- Dataset: build_trainval_dataset (VOC2007+2012, 16,551) for training;
  build_test_dataset (VOC2007 test, 4,952) created but not used here
  (evaluation is in evaluate.py).
- Structured JSON log entries per epoch for ablation analysis.
"""

import argparse
import json
import os
import sys

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


def set_trainable(model: CascadeFCOS, stage: int, tier: int):
    """Configure requires_grad for staged training.

    Tier 2 (PDF Table 1 — recommended): freeze backbone + FPN always.
    Tier 3: freeze backbone only.

    Stage 1: train head_stage_1 only.
    Stage 2: train fcm_1_to_2 + head_stage_2 only.
    Stage 3: train fcm_2_to_3 + head_stage_3 only.
    """
    # Step 1: freeze everything
    _set_requires_grad(model, False)

    # Step 2: selectively unfreeze based on tier
    if tier == 3:
        # Tier 3: FPN is trainable
        _set_requires_grad(model.backbone_fpn.lat_prj3,  True)
        _set_requires_grad(model.backbone_fpn.lat_prj4,  True)
        _set_requires_grad(model.backbone_fpn.lat_prj5,  True)
        _set_requires_grad(model.backbone_fpn.smooth3,   True)
        _set_requires_grad(model.backbone_fpn.smooth4,   True)
        _set_requires_grad(model.backbone_fpn.smooth5,   True)
        _set_requires_grad(model.backbone_fpn.p6_conv,   True)
        _set_requires_grad(model.backbone_fpn.p7_conv,   True)
    # Tier 2 (default): backbone+FPN fully frozen — nothing extra to unfreeze here.

    # Step 3: unfreeze the current stage's trainable modules
    if stage == 1:
        _set_requires_grad(model.head_stage_1, True)
    elif stage == 2:
        _set_requires_grad(model.fcm_1_to_2,   True)
        _set_requires_grad(model.head_stage_2,  True)
    elif stage == 3:
        _set_requires_grad(model.fcm_2_to_3,   True)
        _set_requires_grad(model.head_stage_3,  True)

    # Log what is trainable
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total       = sum(p.numel() for p in model.parameters())
    print(
        f"[Freeze] Stage {stage} | Tier {tier} | "
        f"Trainable: {trainable:,} / {total:,} params "
        f"({100*trainable/total:.1f}%)"
    )


# ── Optimizer builder ─────────────────────────────────────────────────────
def build_optimizer(model, cfg):
    """Build AdamW over trainable parameters only."""
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Build MultiStepLR with epoch-based milestones from config."""
    milestones = cfg.get("lr_milestones", [8, 11])
    gamma      = cfg.get("lr_gamma", 0.1)
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


# ── Main ──────────────────────────────────────────────────────────────────
def main(args):
    # Load YAML config
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
        fh.write(f"--- Cascade FCOS Stage {stage} Training: {run_name} ---\n")
        fh.write(json.dumps(cfg, indent=2) + "\n\n")

    # Dataset (VOC2007+2012 trainval, 16,551 images)
    print("[Data] Building combined VOC2007+2012 trainval dataset …")
    train_dataset = build_trainval_dataset(cfg["data_dir"], augment=True)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    print("[Model] Building CascadeFCOS …")
    model = CascadeFCOS(num_classes=20).to(device)

    # Freeze / unfreeze parameters per stage + tier
    tier = cfg.get("tier", 2)
    set_trainable(model, stage, tier)

    # Load previous-stage checkpoint (required for stages 2 and 3)
    start_epoch = 0
    if args.resume:
        _, _ = load_checkpoint(model, args.resume, device=device)
        # Re-apply freeze after loading (load_state_dict doesn't change requires_grad)
        set_trainable(model, stage, tier)
    elif stage > 1:
        print(
            f"[Warning] Stage {stage} training started without a --resume checkpoint. "
            "This is unusual; stages 2/3 should build on a trained stage 1/2."
        )

    # Optimizer + Scheduler (only trainable params)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    target_gen = DynamicFCOSTargetGenerator()
    loss_fn    = FCOSLoss(alpha=0.25, gamma=2.0)

    epochs = cfg.get("epochs", 12)
    print(f"[Train] Stage {stage} | Epochs: {epochs} | LR: {cfg['lr']}")

    for epoch in range(start_epoch + 1, epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, target_gen, loss_fn,
            device, epoch, stage, log_file,
        )
        scheduler.step()

        # Structured log entry (JSON line) for ablation analysis
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
    parser = argparse.ArgumentParser(description="Cascade FCOS — Staged Training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config (e.g. configs/fcos_baseline.yaml)"
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3],
        help="Which cascade stage to train (1=baseline, 2=first refinement, 3=second)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a previous-stage checkpoint to load before training"
    )
    args = parser.parse_args()
    main(args)
