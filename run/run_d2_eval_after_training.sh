#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for training to finish..."
while pgrep -f 'train_dpo.py.*dpo_qwen14b_v2' > /dev/null; do sleep 60; done
echo "[$(date)] Training finished. Checking adapter exists before eval..."
if [ ! -d /workspace/outputs/checkpoints/dpo_qwen14b_v2/final ] && [ ! -d /media/airlab/ROCSTOR/revguard/outputs/checkpoints/dpo_qwen14b_v2/final ]; then
  echo "[$(date)] ERROR: no final adapter found at dpo_qwen14b_v2/final - training likely crashed. Not running eval against a missing checkpoint." >&2
  exit 1
fi
echo "[$(date)] Adapter found. Starting D2 (trained model) eval, 50 papers..."
apptainer exec --nv --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/evaluation/run_eval.py --condition D2 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 50   --adapter-path /workspace/outputs/checkpoints/dpo_qwen14b_v2/final
echo "[$(date)] D2 eval complete. Minimum-scope evaluation (D0 vs D2) done."
