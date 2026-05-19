# Ablation Experiment Configs

Three ablations satisfy the PDF's "at least 3" requirement.
All ablations build on the standard stage-1 checkpoint unless noted.

---

## Ablation A — Number of refinement stages (T = 1, 2, 3)

No new training required. The standard checkpoints already cover T=1,2,3.
Evaluate each checkpoint and compare with evaluate.py:

```bash
# T=1 (baseline only)
python evaluate.py --checkpoint logs/checkpoints/stage1_epoch012.pth \
                   --eval_stages 1 --skip_complexity

# T=2 (baseline + 1 refinement)
python evaluate.py --checkpoint logs/checkpoints/stage2_epoch008.pth \
                   --eval_stages 1 2 --skip_complexity

# T=3 (baseline + 2 refinements) — includes complexity/overhead check
python evaluate.py --checkpoint logs/checkpoints/stage3_epoch006.pth \
                   --eval_stages 1 2 3
```

Hypothesis: mAP increases with T; latency increases by <15% per FCM stage.

---

## Ablation B — With vs. Without centerness re-prediction at refinement stages

No retraining. Controlled via --no_centerness_stages flag in evaluate.py.
Uses the standard stage-3 checkpoint.

```bash
# With centerness (default)
python evaluate.py --checkpoint logs/checkpoints/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --run_name abl_B_with_cent

# Without centerness at stages 2 and 3
python evaluate.py --checkpoint logs/checkpoints/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --no_centerness_stages 2 3 \
                   --run_name abl_B_no_cent
```

Hypothesis (per FCOS paper Table 4): centerness improves AP by suppressing
off-center high-confidence false positives.

---

## Ablation C — Center-sampling radius schedule

Requires retraining stages 2 and 3 with each radius variant.
Stage 1 is identical in all variants (center_radius=1.5).

Three schedules:
  Progressive (default): 1.5 → 1.0 → 0.75
  Uniform:               1.5 → 1.5 → 1.5
  Aggressive:            1.5 → 0.75 → 0.5

```bash
# --- Uniform schedule ---
# Stage 2
python main.py --config configs/ablations/abl_C_uniform_s2.yaml --stage 2 \
               --resume logs/checkpoints/stage1_epoch012.pth
# Stage 3
python main.py --config configs/ablations/abl_C_uniform_s3.yaml --stage 3 \
               --resume logs/checkpoints/abl_C_uniform/stage2_epoch008.pth

# Evaluate uniform
python evaluate.py --checkpoint logs/checkpoints/abl_C_uniform/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --run_name abl_C_uniform \
                   --skip_complexity

# --- Aggressive schedule ---
# Stage 2
python main.py --config configs/ablations/abl_C_aggressive_s2.yaml --stage 2 \
               --resume logs/checkpoints/stage1_epoch012.pth
# Stage 3
python main.py --config configs/ablations/abl_C_aggressive_s3.yaml --stage 3 \
               --resume logs/checkpoints/abl_C_aggressive/stage2_epoch008.pth

# Evaluate aggressive
python evaluate.py --checkpoint logs/checkpoints/abl_C_aggressive/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --run_name abl_C_aggressive \
                   --skip_complexity
```

Hypothesis: Progressive tightening > Uniform > Aggressive (too few positives at
stage 3 with radius=0.5 will starve the refinement head).

---

## Running all ablations + aggregation

After completing all evaluation runs:

```bash
python scripts/aggregate_results.py --logs_dir logs/
```

This reads every eval_results.json file under logs/ and prints the full
comparison table, saving logs/ablation_summary.csv for the report.
