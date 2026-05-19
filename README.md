# Cascade-FCOS

# 1. Standard training + evaluation
python main.py --config configs/fcos_baseline.yaml   --stage 1
python main.py --config configs/refinement_k1.yaml  --stage 2 --resume logs/checkpoints/stage1_epoch012.pth
python main.py --config configs/refinement_k2.yaml  --stage 3 --resume logs/checkpoints/stage2_epoch008.pth
python evaluate.py --checkpoint logs/checkpoints/stage3_epoch006.pth --eval_stages 1 2 3 --run_name default

# 2. Ablation B (no retraining)
python evaluate.py --checkpoint logs/checkpoints/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --no_centerness_stages 2 3 --run_name abl_B_no_cent

# 3. Ablation C (retraining stages 2+3)
python main.py --config configs/ablations/abl_C_uniform_s2.yaml --stage 2 --resume logs/checkpoints/stage1_epoch012.pth
python main.py --config configs/ablations/abl_C_uniform_s3.yaml --stage 3 --resume logs/checkpoints/abl_C_uniform/stage2_epoch008.pth
python evaluate.py --checkpoint logs/checkpoints/abl_C_uniform/stage3_epoch006.pth \
                   --eval_stages 1 2 3 --run_name abl_C_uniform --skip_complexity

# 4. Aggregate all results
python scripts/aggregate_results.py --logs_dir logs/

# 5. Open notebook
jupyter notebook notebooks/results_visualization.ipynb