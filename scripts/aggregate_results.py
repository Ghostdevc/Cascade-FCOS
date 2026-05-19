"""
scripts/aggregate_results.py

Scans all subdirectories of --logs_dir for eval_results.json files,
combines them into a unified comparison DataFrame, and outputs:
    logs/ablation_summary.csv   — one row per (experiment, stage, metric)
    logs/ablation_pivot.csv     — pivot table (experiments × metrics) for the report
    Prints a formatted comparison table to stdout.

Usage
-----
python scripts/aggregate_results.py --logs_dir logs/

Expected directory layout (produced by evaluate.py --run_name <name>):
    logs/
    ├── default/eval_results.json         # standard T=3 run
    ├── abl_B_no_cent/eval_results.json
    ├── abl_C_uniform/eval_results.json
    └── abl_C_aggressive/eval_results.json
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List


METRIC_COLS = [
    "mAP_coco", "mAP_50", "mAP_75",
    "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10",
]
METRIC_LABELS = {
    "mAP_coco":   "mAP@[.5:.95]",
    "mAP_50":     "mAP@0.5",
    "mAP_75":     "mAP@0.75",
    "mAP_small":  "mAP_small",
    "mAP_medium": "mAP_medium",
    "mAP_large":  "mAP_large",
    "AR_1":       "AR@1",
    "AR_10":      "AR@10",
}


def find_result_files(logs_dir: str) -> List[Path]:
    """Recursively find all eval_results.json files under logs_dir."""
    return sorted(Path(logs_dir).rglob("eval_results.json"))


def load_results(json_path: Path) -> Dict:
    with open(json_path) as f:
        data = json.load(f)
    return data


def flatten_results(json_path: Path, data: Dict) -> List[Dict]:
    """Flatten a single eval_results.json into list of row dicts."""
    run_name = json_path.parent.name
    rows     = []
    perf     = data.get("performance", {})
    cplx     = data.get("complexity", {})

    for stage_key, metrics in perf.items():
        stage = int(stage_key)
        row   = {
            "run":   run_name,
            "stage": stage,
        }
        for m in METRIC_COLS:
            row[m] = round(metrics.get(m, float("nan")) * 100, 3)

        # Attach complexity only for the max stage row
        if cplx and stage == max(int(k) for k in perf.keys()):
            row["gflops"]           = cplx.get("gflops", "N/A")
            row["ms_per_img"]       = cplx.get("full_cascade_ms", "N/A")
            row["fps"]              = cplx.get("full_cascade_fps", "N/A")
            row["overhead_pct"]     = cplx.get("overhead_pct", "N/A")
            row["overhead_pass"]    = cplx.get("overhead_pass", "N/A")
            row["trainable_params"] = cplx.get("trainable_params", "N/A")
            row["peak_gpu_mem_mb"]  = cplx.get("peak_gpu_mem_mb", "N/A")
        rows.append(row)
    return rows


def print_comparison_table(all_rows: List[Dict]):
    """Print a compact comparison table grouped by run."""
    # Group by run
    runs = {}
    for r in all_rows:
        runs.setdefault(r["run"], []).append(r)

    col_w   = 13
    metrics = ["mAP_coco", "mAP_50", "mAP_75", "AR_10"]
    hdr     = f"{'Experiment':<28} {'Stage':<7}" + "".join(
        f"{METRIC_LABELS[m]:>{col_w}}" for m in metrics
    )
    sep = "-" * len(hdr)
    print("\n" + "=" * len(hdr))
    print(" ABLATION COMPARISON TABLE (VOC2007 test, values in %)")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for run_name, rows in sorted(runs.items()):
        for row in sorted(rows, key=lambda r: r["stage"]):
            line = f"{run_name:<28} {row['stage']:<7}"
            for m in metrics:
                v = row.get(m, float("nan"))
                line += f"{v:>{col_w}.2f}"
            print(line)
        print(sep)
    print()


def build_pivot(all_rows: List[Dict]) -> List[Dict]:
    """Build pivot rows: one row per (run, stage), all metric columns."""
    return sorted(all_rows, key=lambda r: (r["run"], r["stage"]))


def main(args):
    result_files = find_result_files(args.logs_dir)
    if not result_files:
        print(f"[Aggregate] No eval_results.json files found under '{args.logs_dir}'")
        print("  Run evaluate.py first to generate results.")
        return

    print(f"[Aggregate] Found {len(result_files)} result file(s):")
    for f in result_files:
        print(f"  {f}")

    all_rows = []
    for f in result_files:
        data = load_results(f)
        rows = flatten_results(f, data)
        all_rows.extend(rows)

    # Print comparison table
    print_comparison_table(all_rows)

    # Save summary CSV (long format)
    out_dir  = args.logs_dir
    long_csv = os.path.join(out_dir, "ablation_summary.csv")
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(long_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"[Save] Long-format CSV → {long_csv}")

    # Save pivot CSV (wide format: one row per run×stage)
    pivot_csv = os.path.join(out_dir, "ablation_pivot.csv")
    pivot     = build_pivot(all_rows)
    if pivot:
        fieldnames = ["run", "stage"] + METRIC_COLS + [
            "gflops", "ms_per_img", "fps", "overhead_pct",
            "overhead_pass", "trainable_params", "peak_gpu_mem_mb",
        ]
        with open(pivot_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(pivot)
        print(f"[Save] Pivot CSV       → {pivot_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Cascade FCOS ablation results")
    parser.add_argument(
        "--logs_dir", type=str, default="logs",
        help="Root directory containing per-run eval_results.json files"
    )
    args = parser.parse_args()
    main(args)
