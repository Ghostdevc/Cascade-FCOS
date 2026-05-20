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

Ablation B support:
    --no_centerness_stages 2 3  disables centerness weighting in the decoder
    for the specified stages (no retraining needed — same checkpoint).
    score = sigmoid(cls)  instead of  sigmoid(cls) × sigmoid(centerness).

Usage
-----
# Standard evaluation of all 3 stages:
python evaluate.py \\
    --checkpoint  logs/checkpoints/stage3_epoch006.pth \\
    --data_dir    data/VOCdevkit \\
    --eval_stages 1 2 3

# Ablation B — no centerness at refinement stages:
python evaluate.py \\
    --checkpoint         logs/checkpoints/stage3_epoch006.pth \\
    --data_dir           data/VOCdevkit \\
    --eval_stages        1 2 3 \\
    --no_centerness_stages 2 3 \\
    --run_name           abl_B_no_cent

# Ablation A — T=1 only, skip complexity:
python evaluate.py \\
    --checkpoint  logs/checkpoints/stage1_epoch012.pth \\
    --eval_stages 1 \\
    --skip_complexity

Outputs:
    logs/<run_name>/eval_results.json   — full metric dict per stage
    logs/<run_name>/eval_results.csv    — flat table for the report
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import sys
try:
    import lzma
except ImportError:
    from backports import lzma
    sys.modules['lzma'] = lzma

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.pascal_voc import build_test_dataset, custom_collate_fn
from models.cascade_fcos import CascadeFCOS
from models.target_generator import DynamicFCOSTargetGenerator
from train import load_checkpoint, pad_batch_images

# ── Constants ─────────────────────────────────────────────────────────────
FPN_STRIDES   = [8, 16, 32, 64, 128]
PRE_NMS_TOPK  = 500
NMS_THRESH    = 0.6    # FCOS paper: "NMS threshold 0.6 instead of 0.5"
SCORE_THRESH  = 0.05
POST_NMS_TOPK = 100

# COCO area thresholds (px², on resized image coordinates)
AREA_SMALL  = 32 ** 2   # <  1024
AREA_MEDIUM = 96 ** 2   # < 9216 (≥ 1024)
# large: ≥ 9216

WARMUP_ITERS = 10
TIMING_ITERS = 50


# ═══════════════════════════════════════════════════════════════════════════
# Inference decoder
# ═══════════════════════════════════════════════════════════════════════════

def _decode_one_level(
    cls_pred:        torch.Tensor,   # (N, num_classes)
    bbox_pred:       torch.Tensor,   # (N, 4)  — (l,t,r,b)
    cent_pred:       torch.Tensor,   # (N, 1)
    locations:       torch.Tensor,   # (N, 2)  — (x, y) query origins
    use_centerness:  bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one FPN level → (boxes_xyxy, scores, labels).

    score = sigmoid(cls) × sigmoid(centerness)  [if use_centerness]
    score = sigmoid(cls)                         [Ablation B: no centerness]
    """
    cls_scores = cls_pred.sigmoid()                         # (N, C)
    cent       = cent_pred.sigmoid().squeeze(-1)            # (N,)

    max_scores, max_labels = cls_scores.max(dim=-1)         # (N,), (N,) 0-indexed
    scores = max_scores * cent if use_centerness else max_scores

    xs, ys = locations[:, 0], locations[:, 1]
    boxes  = torch.stack([
        xs - bbox_pred[:, 0],
        ys - bbox_pred[:, 1],
        xs + bbox_pred[:, 2],
        ys + bbox_pred[:, 3],
    ], dim=-1)

    labels = max_labels + 1   # 1-indexed to match VOC CLASS_TO_IDX

    return boxes, scores, labels


def decode_predictions(
    cascade_preds:        dict,
    stage:                int,
    target_gen:           DynamicFCOSTargetGenerator,
    no_centerness_stages: Set[int] = frozenset(),
    pre_nms_topk:         int   = PRE_NMS_TOPK,
    nms_thresh:           float = NMS_THRESH,
    score_thresh:         float = SCORE_THRESH,
    post_nms_topk:        int   = POST_NMS_TOPK,
) -> List[Dict[str, torch.Tensor]]:
    """Decode cascade predictions for all images in the batch.

    Handles re-anchored centers for stages 2/3 (Fix 3 / target_generator).
    Supports Ablation B via no_centerness_stages set.

    Returns:
        List[Dict] with 'boxes' (M,4), 'scores' (M,), 'labels' (M,)
        per image — after NMS.
    """
    from torchvision.ops import nms as tv_nms

    features_s1   = cascade_preds["features_s1"]
    base_locs     = target_gen.compute_locations(features_s1)
    cls_l, bbox_l, cent_l = cascade_preds[f"stage_{stage}"]
    prev_bbox_key = {2: "bbox_preds_s1", 3: "bbox_preds_s2"}.get(stage, None)
    use_centerness = stage not in no_centerness_stages

    B       = cls_l[0].shape[0]
    results = []

    for b in range(B):
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
            N    = query_locs[lvl].shape[0]
            cls_ = cls_l[lvl][b].permute(1, 2, 0).reshape(N, -1)
            bb_  = bbox_l[lvl][b].permute(1, 2, 0).reshape(N, 4)
            ct_  = cent_l[lvl][b].permute(1, 2, 0).reshape(N, 1)

            bx, sc, lb = _decode_one_level(cls_, bb_, ct_, query_locs[lvl], use_centerness)
            all_boxes.append(bx)
            all_scores.append(sc)
            all_labels.append(lb)

        boxes  = torch.cat(all_boxes,  0)
        scores = torch.cat(all_scores, 0)
        labels = torch.cat(all_labels, 0)

        # Score threshold
        keep = scores > score_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Pre-NMS top-K
        if scores.numel() > pre_nms_topk:
            idx    = scores.topk(pre_nms_topk).indices
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        # Per-class NMS
        if scores.numel() > 0:
            keep_idx = tv_nms(boxes, scores, nms_thresh)
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
            if scores.numel() > post_nms_topk:
                idx    = scores.topk(post_nms_topk).indices
                boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        results.append({
            "boxes":  boxes.cpu(),
            "scores": scores.cpu(),
            "labels": labels.cpu(),
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Per-stage mAP evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_stage(
    model:                CascadeFCOS,
    dataloader:           DataLoader,
    target_gen:           DynamicFCOSTargetGenerator,
    device:               torch.device,
    stage:                int,
    no_centerness_stages: Set[int] = frozenset(),
    verbose:              bool = True,
) -> Dict[str, float]:
    """Run full-dataset inference and compute all PDF-required metrics.

    Uses torchmetrics MeanAveragePrecision (COCO-style).
    Area thresholds on resized image space (800px short-edge).
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        raise ImportError("pip install torchmetrics")

    
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,         # COCO default [0.50:0.05:0.95]
        extended_summary=True,       # map_small/medium/large
        max_detection_thresholds=[1, 10, 100],
    )

    model.eval()
    n = 0
    with torch.no_grad():
        for batch_idx, (imgs_list, tgts_list) in enumerate(dataloader):
            images        = pad_batch_images(imgs_list).to(device)
            cascade_preds = model(images)
            preds         = decode_predictions(
                cascade_preds, stage, target_gen,
                no_centerness_stages=no_centerness_stages,
            )
            gts = [{"boxes": t["boxes"].float(), "labels": t["labels"]} for t in tgts_list]
            metric.update(preds, gts)
            n += len(imgs_list)
            if verbose and batch_idx % 200 == 0:
                print(f"    [{n} / {len(dataloader.dataset)}] …")

    r = metric.compute()
    return {
        "mAP_coco":   float(r.get("map",        0.0)),
        "mAP_50":     float(r.get("map_50",     0.0)),
        "mAP_75":     float(r.get("map_75",     0.0)),
        "mAP_small":  float(r.get("map_small",  0.0)),
        "mAP_medium": float(r.get("map_medium", 0.0)),
        "mAP_large":  float(r.get("map_large",  0.0)),
        "AR_1":       float(r.get("mar_1",      0.0)),
        "AR_10":      float(r.get("mar_10",     0.0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage-1-only wrapper (overhead measurement)
# ═══════════════════════════════════════════════════════════════════════════

class _Stage1OnlyWrapper(nn.Module):
    def __init__(self, m: CascadeFCOS):
        super().__init__()
        self.backbone_fpn = m.backbone_fpn
        self.head_stage_1 = m.head_stage_1

    def forward(self, x):
        feats           = self.backbone_fpn(x)
        cls1, bb1, ct1  = self.head_stage_1(feats)
        return feats, cls1, bb1, ct1


# ═══════════════════════════════════════════════════════════════════════════
# Complexity measurement
# ═══════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model: nn.Module, dummy: torch.Tensor) -> Optional[float]:
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, dummy)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 1e9
    except Exception:
        return None


def measure_latency(
    model:      nn.Module,
    device:     torch.device,
    input_size: Tuple[int, int] = (800, 1066),
    warmup:     int = WARMUP_ITERS,
    iters:      int = TIMING_ITERS,
) -> Tuple[float, float]:
    model.eval()
    dummy    = torch.randn(1, 3, *input_size, device=device)
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms  = (t1 - t0) / iters * 1000.0
    fps = 1000.0 / ms
    return ms, fps


def measure_memory(
    model:      nn.Module,
    device:     torch.device,
    input_size: Tuple[int, int] = (800, 1066),
) -> Optional[float]:
    if device.type != "cuda":
        return None
    model.eval()
    dummy = torch.randn(1, 3, *input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_complexity(
    model:      CascadeFCOS,
    device:     torch.device,
    input_size: Tuple[int, int] = (800, 1066),
) -> Dict:
    print("[Complexity] Counting parameters …")
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)

    print("[Complexity] Estimating GFLOPs …")
    model_cpu = model.cpu().eval()
    gflops    = measure_flops(model_cpu, torch.randn(1, 3, *input_size))
    model.to(device)

    print("[Complexity] Measuring full-cascade latency …")
    full_ms, full_fps = measure_latency(model, device, input_size)

    print("[Complexity] Measuring stage-1-only latency …")
    s1w    = _Stage1OnlyWrapper(model).to(device).eval()
    s1_ms, _ = measure_latency(s1w, device, input_size)

    overhead_pct = (full_ms - s1_ms) / s1_ms * 100.0

    print("[Complexity] Measuring peak GPU memory …")
    peak_mem = measure_memory(model, device, input_size)

    return {
        "total_params":     total_params,
        "trainable_params": trainable_params,
        "gflops":           round(gflops, 2) if gflops is not None else "N/A",
        "gflops_note":      "lower bound (DeformConv2d may be under-counted)",
        "full_cascade_ms":  round(full_ms, 2),
        "full_cascade_fps": round(full_fps, 1),
        "stage1_only_ms":   round(s1_ms, 2),
        "overhead_pct":     round(overhead_pct, 1),
        "overhead_target":  "<15%",
        "overhead_pass":    overhead_pct < 15.0,
        "peak_gpu_mem_mb":  round(peak_mem, 1) if peak_mem is not None else "N/A",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════

def _print_table(all_metrics: Dict, complexity: Optional[Dict]):
    col_w   = 14
    stages  = sorted(all_metrics.keys())
    headers = ["Metric"] + [f"Stage {s}" for s in stages]
    sep     = "=" * (col_w * len(headers) + 2)
    print("\n" + sep)
    print("  CASCADE FCOS — EVALUATION RESULTS (VOC2007 test)")
    print(sep)
    print("".join(f"{h:<{col_w}}" for h in headers))
    print("-" * (col_w * len(headers)))
    rows = [
        ("mAP_coco",   "mAP@[.5:.95]"),
        ("mAP_50",     "mAP@0.5"),
        ("mAP_75",     "mAP@0.75"),
        ("mAP_small",  "mAP_small"),
        ("mAP_medium", "mAP_medium"),
        ("mAP_large",  "mAP_large"),
        ("AR_1",       "AR@1"),
        ("AR_10",      "AR@10"),
    ]
    for key, label in rows:
        row = f"{label:<{col_w}}"
        for s in stages:
            v    = all_metrics[s].get(key, float("nan"))
            row += f"{v*100:>{col_w-2}.2f}%  "
        print(row)
    if complexity:
        print("-" * (col_w * len(headers)))
        print("COMPLEXITY")
        print("-" * (col_w * len(headers)))
        for label, val in [
            ("Params (total)",  f"{complexity['total_params']:,}"),
            ("Params (train.)", f"{complexity['trainable_params']:,}"),
            ("GFLOPs",          str(complexity["gflops"])),
            ("ms / image",      f"{complexity['full_cascade_ms']} ms"),
            ("FPS",             str(complexity["full_cascade_fps"])),
            ("Stage-1 ms",      f"{complexity['stage1_only_ms']} ms"),
            ("FCM overhead",    f"{complexity['overhead_pct']}%  "
                                f"({'PASS ✓' if complexity['overhead_pass'] else 'FAIL ✗'})"),
            ("Peak GPU mem",    f"{complexity['peak_gpu_mem_mb']} MB"),
        ]:
            print(f"  {label:<22}{val}")
    print(sep + "\n")


def _save_results(all_metrics: Dict, complexity: Optional[Dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    payload = {"performance": all_metrics, "complexity": complexity}
    json_path = os.path.join(out_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] JSON → {json_path}")

    csv_path = os.path.join(out_dir, "eval_results.csv")
    rows = [
        {"stage": s, "metric": k, "value_pct": round(v * 100, 4)}
        for s, mets in all_metrics.items()
        for k, v in mets.items()
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "metric", "value_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"[Save] CSV  → {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    print("[Eval] Loading VOC2007 test set …")
    test_ds  = build_test_dataset(args.data_dir)
    test_ldr = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Eval] Loading checkpoint: {args.checkpoint}")
    model = CascadeFCOS(num_classes=20).to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    target_gen           = DynamicFCOSTargetGenerator()
    no_cent              = set(args.no_centerness_stages)
    all_metrics: Dict    = {}

    for stage in sorted(args.eval_stages):
        cent_note = " [no centerness]" if stage in no_cent else ""
        print(f"\n[Eval] ── Stage {stage}{cent_note} ──────────────────────────────")
        m = evaluate_stage(model, test_ldr, target_gen, device, stage,
                           no_centerness_stages=no_cent)
        all_metrics[stage] = m
        print(
            f"  mAP@[.5:.95]={m['mAP_coco']*100:.2f}%  "
            f"mAP@0.5={m['mAP_50']*100:.2f}%  "
            f"mAP@0.75={m['mAP_75']*100:.2f}%"
        )

    complexity = None
    if not args.skip_complexity:
        print("\n[Eval] ── Complexity ─────────────────────────────────────────")
        complexity = measure_complexity(model, device)

    _print_table(all_metrics, complexity)

    # Output directory — one subfolder per run_name so ablation runs don't clobber each other
    out_dir = os.path.join("logs", args.run_name)
    _save_results(all_metrics, complexity, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade FCOS — Evaluation")
    parser.add_argument("--checkpoint",    type=str,  required=True)
    parser.add_argument("--data_dir",      type=str,  default="data/VOCdevkit")
    parser.add_argument("--eval_stages",   type=int,  nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3])
    parser.add_argument("--no_centerness_stages", type=int, nargs="*", default=[],
                        help="Ablation B: stages for which centerness is NOT used in score")
    parser.add_argument("--run_name",      type=str,  default="default",
                        help="Subfolder under logs/ for this run's outputs")
    parser.add_argument("--num_workers",   type=int,  default=4)
    parser.add_argument("--skip_complexity", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)
