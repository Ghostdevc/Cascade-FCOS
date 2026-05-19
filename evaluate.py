"""
evaluate.py — Complete evaluation pipeline for Cascade FCOS.

Implements all metrics required by PDF Section 3:

Performance metrics:
    mAP@[0.5:0.95]  (COCO-style, primary)
    mAP@0.5         (PASCAL metric, primary)
    mAP@0.75
    mAP small / medium / large  (COCO area thresholds: <32², 32²-96², >96²)
    AR@1, AR@10
    All computed on resized-image-space boxes (800px short-edge) for consistency.

Complexity metrics:
    Trainable parameters
    GFLOPs per image  (thop; fallback to fvcore)
    ms per image, FPS  (GPU warm-up + torch.cuda.synchronize())
    Peak GPU memory per image  (torch.cuda.max_memory_allocated())

Inference overhead bonus:
    Stage-1-only latency vs. full-cascade latency → overhead %.
    Target: <15% for the FCM bonus condition.

Per-stage evaluation:
    Evaluates stage 1, 2, 3 separately from a single checkpoint,
    showing the refinement progression (baseline → +1 step → +2 steps).

Usage
-----
# Evaluate all 3 stages from the final (stage-3) checkpoint:
python evaluate.py \\
    --checkpoint logs/checkpoints/stage3_epoch006.pth \\
    --data_dir   data/VOCdevkit \\
    --eval_stages 1 2 3

# Evaluate stage 1 only, no complexity measurement:
python evaluate.py \\
    --checkpoint logs/checkpoints/stage1_epoch012.pth \\
    --data_dir   data/VOCdevkit \\
    --eval_stages 1 \\
    --skip_complexity

Outputs:
    logs/eval_results.json   — full metric dict per stage
    logs/eval_results.csv    — flat table suitable for the report
    Printed results table to stdout.
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.pascal_voc import build_test_dataset, custom_collate_fn
from models.cascade_fcos import CascadeFCOS
from models.target_generator import DynamicFCOSTargetGenerator
from train import load_checkpoint, pad_batch_images

# ── Constants ────────────────────────────────────────────────────────────
FPN_STRIDES      = [8, 16, 32, 64, 128]
PRE_NMS_TOPK     = 500    # pre-NMS candidates per image
NMS_THRESH       = 0.6    # FCOS paper: "NMS threshold 0.6 instead of 0.5"
SCORE_THRESH     = 0.05   # minimum score to keep before NMS
POST_NMS_TOPK    = 100    # maximum boxes after NMS

# COCO area thresholds (in pixels², on resized image coordinates)
AREA_SMALL  = 32 ** 2          # <1024
AREA_MEDIUM = 96 ** 2          # 1024-9216
# large: >9216

WARMUP_ITERS  = 10   # GPU warm-up iterations for timing
TIMING_ITERS  = 50   # timing iterations to average


# ═══════════════════════════════════════════════════════════════════════════
# Inference decoder
# ═══════════════════════════════════════════════════════════════════════════

def _decode_one_level(
    cls_pred:  torch.Tensor,   # (H*W, num_classes)
    bbox_pred: torch.Tensor,   # (H*W, 4)  — (l,t,r,b) in pixels
    cent_pred: torch.Tensor,   # (H*W, 1)
    locations: torch.Tensor,   # (H*W, 2)  — (x, y) query origins
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode predictions for one FPN level into (boxes, scores, labels).

    Score = sigmoid(cls) × sigmoid(centerness).
    Box   = (x-l, y-t, x+r, y+b).

    Returns:
        boxes  : (N, 4) xyxy
        scores : (N,)
        labels : (N,) 1-indexed class labels
    """
    cls_scores = cls_pred.sigmoid()    # (N, C)
    cent       = cent_pred.sigmoid().squeeze(-1)   # (N,)

    # Final score per location: max class prob × centerness
    max_scores, max_labels = cls_scores.max(dim=-1)  # (N,), (N,) 0-indexed
    scores = max_scores * cent                        # centerness weighting

    xs, ys = locations[:, 0], locations[:, 1]
    x1 = xs - bbox_pred[:, 0]
    y1 = ys - bbox_pred[:, 1]
    x2 = xs + bbox_pred[:, 2]
    y2 = ys + bbox_pred[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    labels = max_labels + 1   # convert to 1-indexed to match VOC CLASS_TO_IDX

    return boxes, scores, labels


def decode_predictions(
    cascade_preds: dict,
    stage:         int,
    target_gen:    DynamicFCOSTargetGenerator,
    pre_nms_topk:  int = PRE_NMS_TOPK,
    nms_thresh:    float = NMS_THRESH,
    score_thresh:  float = SCORE_THRESH,
    post_nms_topk: int = POST_NMS_TOPK,
) -> List[Dict[str, torch.Tensor]]:
    """Decode cascade predictions for all images in a batch.

    Handles the re-anchored centers for stages 2 and 3:
      Stage 1 → static FPN grid as query origins.
      Stage 2 → refined from stage-1 bbox_preds  (x' = x + (r-l)/2).
      Stage 3 → refined from stage-2 bbox_preds.

    Returns:
        List[Dict] with keys 'boxes' (M,4), 'scores' (M,), 'labels' (M,)
        — one dict per image in the batch, after NMS.
    """
    from torchvision.ops import nms as torchvision_nms

    features_s1   = cascade_preds["features_s1"]
    base_locs     = target_gen.compute_locations(features_s1)  # static grid
    cls_l, bbox_l, cent_l = cascade_preds[f"stage_{stage}"]

    # Determine query-origin source for stages 2/3
    prev_bbox_key = {2: "bbox_preds_s1", 3: "bbox_preds_s2"}.get(stage, None)

    B = cls_l[0].shape[0]
    results = []

    for b in range(B):
        # Build per-image, per-level query origins
        if prev_bbox_key is not None:
            prev_bp = cascade_preds[prev_bbox_key]
            prev_per_level = [
                prev_bp[l][b].permute(1, 2, 0).reshape(-1, 4)
                for l in range(len(base_locs))
            ]
            query_locs = target_gen.compute_refined_locations(base_locs, prev_per_level)
        else:
            query_locs = base_locs

        all_boxes, all_scores, all_labels = [], [], []

        for lvl in range(len(base_locs)):
            H, W = cls_l[lvl].shape[2:]
            cls_flat  = cls_l[lvl][b].permute(1, 2, 0).reshape(-1, cls_l[lvl].shape[1])
            bbox_flat = bbox_l[lvl][b].permute(1, 2, 0).reshape(-1, 4)
            cent_flat = cent_l[lvl][b].permute(1, 2, 0).reshape(-1, 1)
            locs      = query_locs[lvl]

            boxes, scores, labels = _decode_one_level(
                cls_flat, bbox_flat, cent_flat, locs
            )
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        all_boxes  = torch.cat(all_boxes,  dim=0)   # (N_total, 4)
        all_scores = torch.cat(all_scores, dim=0)   # (N_total,)
        all_labels = torch.cat(all_labels, dim=0)   # (N_total,)

        # Score threshold filter
        keep = all_scores > score_thresh
        all_boxes, all_scores, all_labels = (
            all_boxes[keep], all_scores[keep], all_labels[keep]
        )

        # Pre-NMS: keep top-K by score
        if all_scores.numel() > pre_nms_topk:
            topk_idx = all_scores.topk(pre_nms_topk).indices
            all_boxes, all_scores, all_labels = (
                all_boxes[topk_idx], all_scores[topk_idx], all_labels[topk_idx]
            )

        # Per-class NMS
        if all_scores.numel() > 0:
            keep_idx = torchvision_nms(all_boxes, all_scores, nms_thresh)
            all_boxes, all_scores, all_labels = (
                all_boxes[keep_idx], all_scores[keep_idx], all_labels[keep_idx]
            )

            # Post-NMS: keep top post_nms_topk
            if all_scores.numel() > post_nms_topk:
                topk_idx = all_scores.topk(post_nms_topk).indices
                all_boxes, all_scores, all_labels = (
                    all_boxes[topk_idx], all_scores[topk_idx], all_labels[topk_idx]
                )

        results.append({
            "boxes":  all_boxes.cpu(),
            "scores": all_scores.cpu(),
            "labels": all_labels.cpu(),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Per-stage mAP evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_stage(
    model:      CascadeFCOS,
    dataloader: DataLoader,
    target_gen: DynamicFCOSTargetGenerator,
    device:     torch.device,
    stage:      int,
    verbose:    bool = True,
) -> Dict[str, float]:
    """Run inference on the full test set and compute all PDF-required metrics.

    Uses torchmetrics.detection.MeanAveragePrecision which implements
    COCO-style evaluation internally (via pycocotools-compatible routines).

    Area thresholds (COCO convention, on resized image space):
        small  : area < 32² = 1024 px²
        medium : 1024 ≤ area < 96² = 9216 px²
        large  : area ≥ 9216 px²
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        raise ImportError(
            "torchmetrics not installed. Run: pip install torchmetrics"
        )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,   # COCO default: [0.50, 0.55, …, 0.95]
        area_ranges={
            "all":    (0, float("inf")),
            "small":  (0, AREA_SMALL),
            "medium": (AREA_SMALL, AREA_MEDIUM),
            "large":  (AREA_MEDIUM, float("inf")),
        },
        max_detection_thresholds=[1, 10, 100],
    )

    model.eval()
    n_images = 0

    with torch.no_grad():
        for batch_idx, (images_list, targets_list) in enumerate(dataloader):
            images = pad_batch_images(images_list).to(device)
            cascade_preds = model(images)

            preds = decode_predictions(cascade_preds, stage, target_gen)

            # Build ground-truth list (boxes already in resized image space)
            gt_list = []
            for t in targets_list:
                gt_list.append({
                    "boxes":  t["boxes"].float(),
                    "labels": t["labels"],
                })

            metric.update(preds, gt_list)
            n_images += len(images_list)

            if verbose and batch_idx % 200 == 0:
                print(f"  [Stage {stage}] {n_images} images processed …")

    result = metric.compute()

    # Extract and rename to the PDF's required metric names
    metrics = {
        "mAP_coco":   float(result.get("map",        0.0)),
        "mAP_50":     float(result.get("map_50",     0.0)),
        "mAP_75":     float(result.get("map_75",     0.0)),
        "mAP_small":  float(result.get("map_small",  0.0)),
        "mAP_medium": float(result.get("map_medium", 0.0)),
        "mAP_large":  float(result.get("map_large",  0.0)),
        "AR_1":       float(result.get("mar_1",      0.0)),
        "AR_10":      float(result.get("mar_10",     0.0)),
    }
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Stage-1-only wrapper (for overhead measurement)
# ═══════════════════════════════════════════════════════════════════════════

class _Stage1OnlyWrapper(nn.Module):
    """Runs only backbone+FPN + head_stage_1. Used to measure overhead."""
    def __init__(self, cascade_model: CascadeFCOS):
        super().__init__()
        self.backbone_fpn = cascade_model.backbone_fpn
        self.head_stage_1 = cascade_model.head_stage_1

    def forward(self, x):
        feats = self.backbone_fpn(x)
        cls1, bbox1, cent1 = self.head_stage_1(feats)
        return feats, cls1, bbox1, cent1


# ═══════════════════════════════════════════════════════════════════════════
# Complexity measurement
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> Optional[float]:
    """Estimate GFLOPs using thop (with fvcore fallback).

    Note: thop may under-count DeformConv2d FLOPs; the reported value is a
    lower bound.  We flag this in the output.
    """
    # Try thop first
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return flops / 1e9  # GFLOPs
    except Exception:
        pass

    # Fallback: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        flop_counter = FlopCountAnalysis(model, input_tensor)
        flop_counter.unsupported_ops_warnings(False)
        flop_counter.uncalled_modules_warnings(False)
        return flop_counter.total() / 1e9
    except Exception:
        return None


def measure_latency(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int] = (800, 1066),
    warmup: int = WARMUP_ITERS,
    iters:  int = TIMING_ITERS,
) -> Tuple[float, float]:
    """Measure inference latency in ms/image and FPS.

    Uses GPU warm-up + torch.cuda.synchronize() for accurate timing.
    Returns (ms_per_image, fps).
    """
    model.eval()
    dummy = torch.randn(1, 3, *input_size, device=device)
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms_per_img = (t1 - t0) / iters * 1000.0
    fps        = 1000.0 / ms_per_img
    return ms_per_img, fps


def measure_memory(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int] = (800, 1066),
) -> Optional[float]:
    """Measure peak GPU memory (MB) for a single-image forward pass."""
    if device.type != "cuda":
        return None

    model.eval()
    dummy = torch.randn(1, 3, *input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / 1024 ** 2  # MB


def measure_complexity(
    model: CascadeFCOS,
    device: torch.device,
    input_size: Tuple[int, int] = (800, 1066),
) -> Dict:
    """Collect all PDF-required complexity metrics.

    Also measures stage-1-only vs. full-cascade latency for the overhead bonus.
    """
    print("[Complexity] Counting parameters …")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)

    # GFLOPs
    print("[Complexity] Estimating GFLOPs …")
    model_cpu  = model.cpu().eval()
    dummy_cpu  = torch.randn(1, 3, *input_size)
    gflops     = measure_flops(model_cpu, dummy_cpu)
    model      = model.to(device)

    # Full-cascade latency
    print("[Complexity] Measuring full-cascade latency …")
    full_ms, full_fps = measure_latency(model, device, input_size)

    # Stage-1-only latency (overhead bonus check)
    print("[Complexity] Measuring stage-1-only latency …")
    s1_wrapper = _Stage1OnlyWrapper(model).to(device).eval()
    s1_ms, _   = measure_latency(s1_wrapper, device, input_size)

    overhead_pct = (full_ms - s1_ms) / s1_ms * 100.0

    # Peak GPU memory
    print("[Complexity] Measuring peak GPU memory …")
    peak_mem_mb = measure_memory(model, device, input_size)

    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "gflops":           round(gflops, 2) if gflops is not None else "N/A",
        "gflops_note":      "lower bound (DeformConv2d may be under-counted)",
        "full_cascade_ms":  round(full_ms, 2),
        "full_cascade_fps": round(full_fps, 1),
        "stage1_only_ms":   round(s1_ms, 2),
        "overhead_pct":     round(overhead_pct, 1),
        "overhead_target":  "<15% (bonus condition)",
        "overhead_pass":    overhead_pct < 15.0,
        "peak_gpu_mem_mb":  round(peak_mem_mb, 1) if peak_mem_mb is not None else "N/A (CPU)",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Results formatting
# ═══════════════════════════════════════════════════════════════════════════

def _print_results_table(all_metrics: Dict[str, Dict], complexity: Optional[Dict]):
    """Print a formatted results table to stdout."""
    col_w = 14
    header_cols = ["Metric"] + [f"Stage {s}" for s in sorted(all_metrics.keys())]
    print("\n" + "=" * (col_w * len(header_cols) + 2))
    print("  CASCADE FCOS — EVALUATION RESULTS (VOC2007 test)")
    print("=" * (col_w * len(header_cols) + 2))

    # Header row
    row = "".join(f"{c:<{col_w}}" for c in header_cols)
    print(row)
    print("-" * (col_w * len(header_cols)))

    # Metric rows
    metric_labels = {
        "mAP_coco":   "mAP@[.5:.95]",
        "mAP_50":     "mAP@0.5",
        "mAP_75":     "mAP@0.75",
        "mAP_small":  "mAP_small",
        "mAP_medium": "mAP_medium",
        "mAP_large":  "mAP_large",
        "AR_1":       "AR@1",
        "AR_10":      "AR@10",
    }
    stages = sorted(all_metrics.keys())
    for key, label in metric_labels.items():
        row = f"{label:<{col_w}}"
        for s in stages:
            val = all_metrics[s].get(key, float("nan"))
            row += f"{val*100:>{col_w-1}.2f}%"
            row += " "
        print(row)

    # Complexity section
    if complexity:
        print("-" * (col_w * len(header_cols)))
        print("COMPLEXITY")
        print("-" * (col_w * len(header_cols)))
        crows = [
            ("Params (total)",   f"{complexity['total_params']:,}"),
            ("Params (train.)",  f"{complexity['trainable_params']:,}"),
            ("GFLOPs",           str(complexity["gflops"])),
            ("ms / image",       f"{complexity['full_cascade_ms']} ms"),
            ("FPS",              f"{complexity['full_cascade_fps']}"),
            ("Stage-1 ms",       f"{complexity['stage1_only_ms']} ms"),
            ("FCM overhead",     f"{complexity['overhead_pct']}% ({'PASS ✓' if complexity['overhead_pass'] else 'FAIL ✗'})"),
            ("Peak GPU mem",     f"{complexity['peak_gpu_mem_mb']} MB"),
        ]
        for label, val in crows:
            print(f"  {label:<20} {val}")

    print("=" * (col_w * len(header_cols) + 2))
    print()


def _save_results(all_metrics: Dict, complexity: Optional[Dict], out_dir: str = "logs"):
    """Save results as JSON and CSV."""
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    payload = {"performance": all_metrics, "complexity": complexity}
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] JSON → {json_path}")

    # CSV (flat, one row per stage × metric)
    csv_path = os.path.join(out_dir, "eval_results.csv")
    rows = []
    for stage, metrics in all_metrics.items():
        for metric_key, val in metrics.items():
            rows.append({
                "stage": stage,
                "metric": metric_key,
                "value": round(val * 100, 4),   # percentage
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Save] CSV  → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Build test dataset
    print("[Eval] Loading VOC2007 test set …")
    test_dataset = build_test_dataset(args.data_dir)
    # batch_size=1 ensures consistent timing and avoids padding artifacts
    test_loader  = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Load model + checkpoint
    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model = CascadeFCOS(num_classes=20).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    target_gen = DynamicFCOSTargetGenerator()
    all_metrics = {}

    # Per-stage evaluation
    for stage in sorted(args.eval_stages):
        print(f"\n[Eval] ── Stage {stage} ──────────────────────────────────────")
        stage_metrics = evaluate_stage(model, test_loader, target_gen, device, stage)
        all_metrics[stage] = stage_metrics

        # Quick summary
        print(
            f"  mAP@[.5:.95]={stage_metrics['mAP_coco']*100:.2f}%  "
            f"mAP@0.5={stage_metrics['mAP_50']*100:.2f}%  "
            f"mAP@0.75={stage_metrics['mAP_75']*100:.2f}%"
        )

    # Complexity (optional)
    complexity = None
    if not args.skip_complexity:
        print("\n[Eval] ── Complexity ─────────────────────────────────────────")
        complexity = measure_complexity(model, device)

    # Print table + save
    _print_results_table(all_metrics, complexity)
    _save_results(all_metrics, complexity, out_dir="logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade FCOS — Evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pth). A stage-3 checkpoint contains all weights."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/VOCdevkit",
        help="Path to VOCdevkit/ (must contain VOC2007/ImageSets/Main/test.txt)"
    )
    parser.add_argument(
        "--eval_stages", type=int, nargs="+", default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which cascade stages to evaluate (default: 1 2 3)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker count"
    )
    parser.add_argument(
        "--skip_complexity", action="store_true",
        help="Skip GFLOPs / latency / memory measurement (for quick AP-only runs)"
    )
    args = parser.parse_args()
    run_evaluation(args)