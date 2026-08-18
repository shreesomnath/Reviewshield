#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for D0 local eval to finish..."
while pgrep -f 'condition D0.*local' > /dev/null; do sleep 60; done
echo "[$(date)] D0 finished. Starting D1 local eval, alone on GPU1..."
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python /workspace/scripts/evaluation/run_eval.py --condition D1 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 50
echo "[$(date)] D1 local eval complete."
