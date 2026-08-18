#!/bin/bash
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for D3 SFT training (pid 77351) to finish on GPU0..."
while kill -0 77351 2>/dev/null; do sleep 60; done
echo "[$(date)] Training process exited. Checking adapter exists before eval..."
if [ ! -d /workspace/outputs/checkpoints/sft_qwen14b_v1/final ] && [ ! -d /media/airlab/ROCSTOR/revguard/outputs/checkpoints/sft_qwen14b_v1/final ]; then
  echo "[$(date)] ERROR: no final adapter found at sft_qwen14b_v1/final - training likely crashed. Not running eval." >&2
  exit 1
fi
echo "[$(date)] Adapter found. Starting D3 eval, 50 papers..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python /workspace/scripts/evaluation/run_eval.py --condition D3 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 50 --adapter-path /workspace/outputs/checkpoints/sft_qwen14b_v1/final
echo "[$(date)] D3 eval complete."
