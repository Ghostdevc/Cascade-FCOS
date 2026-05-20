"""
evaluate.py — Cascade FCOS evaluation pipeline (Tier 1, torchvision baseline).

Key changes:
- Decoder handles stride-normalised bbox from stage 1 (torchvision FCOS native)
  vs. pixel-unit bbox from stages 2/3 (RefinementHead exp(scale*raw)).
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.pascal_voc import build_test_dataset, custom_collate_fn
from models.cascade_fcos import CascadeFCOS
from models.target_generator import DynamicFCOSTargetGenerator
from train import load_checkpoint, pad_batch_images, FPN_STRIDES

# ── Constants ─────────────────────────────────────────────────────────────
PRE_NMS_TOPK  = 500
NMS_THRESH    = 0.6
SCORE_THRESH  = 0.05
POST_NMS_TOPK = 100

AREA_SMALL    = 32 ** 2
AREA_MEDIUM   = 96 ** 2

WARMUP_ITERS  = 10
TIMING_ITERS  = 50


# ═══════════════════════════════════════════════════════════════════════════
# Inference decoder
# ═══════════════════════════════════════════════════════════════════════════

def _decode_one_level(
    cls_pred:        torch.Tensor,
    bbox_pred:       torch.Tensor,
    cent_pred:       torch.Tensor,
    locations:       torch.Tensor,
    stride:          int,
    is_stage_1:      bool,
    use_centerness:  bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode one FPN level into (boxes_xyxy, scores, labels).

    Stage 1 (torchvision FCOS): bbox is stride-normalised, multiply by stride.
    Stages 2/3 (RefinementHead): bbox is already in pixel units.
    """
    cls_scores = cls_pred.sigmoid()
    cent       = cent_pred.sigmoid().squeeze(-1)

    max_scores, max_labels = cls_scores.max(dim=-1)
    scores = max_scores * cent if use_centerness else max_scores

    if is_stage_1:
        bbox_pred = bbox_pred * stride

    xs, ys = locations[:, 0], locations[:, 1]
    boxes  = torch.stack([
        xs - bbox_pred[:, 0],
        ys - bbox_pred[:, 1],
        xs + bbox_pred[:, 2],
        ys + bbox_pred[:, 3],
    ], dim=-1)
    labels = max_labels + 1
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
    """Decode cascade predictions for all images in the batch."""
    from torchvision.ops import nms as tv_nms

    features_s1   = cascade_preds["features_s1"]
    base_locs     = target_gen.compute_locations(features_s1)
    cls_l, bbox_l, cent_l = cascade_preds[f"stage_{stage}"]
    prev_bbox_key  = {2: "bbox_preds_s1", 3: "bbox_preds_s2"}.get(stage, None)
    use_centerness = stage not in no_centerness_stages
    is_stage_1     = (stage == 1)
    # For stage 2, prev (stage 1) is stride-normalised; for stage 3, prev is pixel
    prev_is_stride_normalised = (stage == 2)

    B = cls_l[0].shape[0]
    results = []

    for b in range(B):
        # Refined query origins for stages 2/3
        if prev_bbox_key is not None:
            prev_bp = cascade_preds[prev_bbox_key]
            prev_per_level = []
            for lvl in range(len(base_locs)):
                bp = prev_bp[lvl][b].permute(1, 2, 0).reshape(-1, 4)
                if prev_is_stride_normalised:
                    bp = bp * FPN_STRIDES[lvl]
                prev_per_level.append(bp)
            query_locs = target_gen.compute_refined_locations(base_locs, prev_per_level)
        else:
            query_locs = base_locs

        all_boxes, all_scores, all_labels = [], [], []
        for lvl in range(len(base_locs)):
            N    = query_locs[lvl].shape[0]
            cls_ = cls_l[lvl][b].permute(1, 2, 0).reshape(N, -1)
            bb_  = bbox_l[lvl][b].permute(1, 2, 0).reshape(N, 4)
            ct_  = cent_l[lvl][b].permute(1, 2, 0).reshape(N, 1)

            bx, sc, lb = _decode_one_level(
                cls_, bb_, ct_, query_locs[lvl],
                stride=FPN_STRIDES[lvl],
                is_stage_1=is_stage_1,
                use_centerness=use_centerness,
            )
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
            idx = scores.topk(pre_nms_topk).indices
            boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        # NMS
        if scores.numel() > 0:
            keep_idx = tv_nms(boxes, scores, nms_thresh)
            boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]
            if scores.numel() > post_nms_topk:
                idx = scores.topk(post_nms_topk).indices
                boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

        results.append({"boxes": boxes.cpu(), "scores": scores.cpu(), "labels": labels.cpu()})
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Per-stage mAP evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_stage(model, dataloader, target_gen, device, stage,
                   no_centerness_stages: Set[int] = frozenset(), verbose: bool = True):
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=None,
        extended_summary=True,
        max_detection_thresholds=[1, 10, 100],
    )

    model.eval()
    n = 0
    with torch.no_grad():
        for batch_idx, (imgs_list, tgts_list) in enumerate(dataloader):
            images        = pad_batch_images(imgs_list).to(device)
            cascade_preds = model(images, max_stage=stage)
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
        "mAP_coco":   float(r.get("map", 0.0)),
        "mAP_50":     float(r.get("map_50", 0.0)),
        "mAP_75":     float(r.get("map_75", 0.0)),
        "mAP_small":  float(r.get("map_small", 0.0)),
        "mAP_medium": float(r.get("map_medium", 0.0)),
        "mAP_large":  float(r.get("map_large", 0.0)),
        "AR_1":       float(r.get("mar_1", 0.0)),
        "AR_10":      float(r.get("mar_10", 0.0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Complexity
# ═══════════════════════════════════════════════════════════════════════════

class _Stage1OnlyWrapper(nn.Module):
    def __init__(self, m: CascadeFCOS):
        super().__init__()
        self.model = m
    def forward(self, x):
        return self.model(x, max_stage=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops(model, dummy):
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis
            fa = FlopCountAnalysis(model, dummy)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            return fa.total() / 1e9
        except Exception:
            return None


def measure_latency(model, device, input_size=(800, 1066),
                    warmup=WARMUP_ITERS, iters=TIMING_ITERS):
    model.eval()
    dummy = torch.randn(1, 3, *input_size, device=device)
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
    ms = (t1 - t0) / iters * 1000.0
    return ms, 1000.0 / ms


def measure_memory(model, device, input_size=(800, 1066)):
    if device.type != "cuda":
        return None
    model.eval()
    dummy = torch.randn(1, 3, *input_size, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 1024 ** 2


def measure_complexity(model, device, input_size=(800, 1066)):
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
    s1w   = _Stage1OnlyWrapper(model).to(device).eval()
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

def _print_table(all_metrics, complexity):
    col_w   = 14
    stages  = sorted(all_metrics.keys())
    headers = ["Metric"] + [f"Stage {s}" for s in stages]
    sep     = "=" * (col_w * len(headers) + 2)
    print("\n" + sep)
    print("  CASCADE FCOS — EVALUATION RESULTS (VOC2007 test)")
    print(sep)
    print("".join(f"{h:<{col_w}}" for h in headers))
    print("-" * (col_w * len(headers)))
    for key, label in [
        ("mAP_coco",   "mAP@[.5:.95]"),
        ("mAP_50",     "mAP@0.5"),
        ("mAP_75",     "mAP@0.75"),
        ("mAP_small",  "mAP_small"),
        ("mAP_medium", "mAP_medium"),
        ("mAP_large",  "mAP_large"),
        ("AR_1",       "AR@1"),
        ("AR_10",      "AR@10"),
    ]:
        row = f"{label:<{col_w}}"
        for s in stages:
            v = all_metrics[s].get(key, float("nan"))
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


def _save_results(all_metrics, complexity, out_dir):
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

    target_gen = DynamicFCOSTargetGenerator()
    no_cent    = set(args.no_centerness_stages)
    all_metrics = {}

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
    _save_results(all_metrics, complexity, out_dir=os.path.join("logs", args.run_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascade FCOS — Evaluation")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--data_dir",    type=str, default="data/VOCdevkit")
    parser.add_argument("--eval_stages", type=int, nargs="+", default=[1, 2, 3],
                        choices=[1, 2, 3])
    parser.add_argument("--no_centerness_stages", type=int, nargs="*", default=[])
    parser.add_argument("--run_name",    type=str, default="default")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_complexity", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)