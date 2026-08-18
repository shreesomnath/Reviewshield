#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for training to finish..."
while pgrep -f 'train_dpo.py.*dpo_qwen14b_v1' > /dev/null; do sleep 60; done
echo "[$(date)] Training finished. Starting D0 (baseline) eval, 50 papers..."
apptainer exec --nv --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/evaluation/run_eval.py --condition D0 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 50
echo "[$(date)] D0 done. Starting D2 (trained model) eval, 50 papers..."
apptainer exec --nv --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/evaluation/run_eval.py --condition D2 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 50   --adapter-path /workspace/outputs/checkpoints/dpo_qwen14b_v1/final
echo "[$(date)] Minimum-scope evaluation (D0 vs D2) complete."
