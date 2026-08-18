#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for longform validation (pid 35738) to finish on GPU1..."
while kill -0 35738 2>/dev/null; do sleep 60; done
echo "[$(date)] GPU1 free. Starting Qwen content-capture D0..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D0 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 10
echo "[$(date)] D0 done. Starting Qwen content-capture D1..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D1 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 10
echo "[$(date)] D1 done. Starting Qwen content-capture D2..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D2 --model-id Qwen/Qwen2.5-14B-Instruct --backend local --limit 10 --adapter-path /workspace/outputs/checkpoints/dpo_qwen14b_v2/final
echo "[$(date)] QWEN_CONTENT_CAPTURE_ALL_DONE"
