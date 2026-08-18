#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for Llama D1 eval (pid 52616) to finish on GPU0..."
while kill -0 52616 2>/dev/null; do sleep 60; done
echo "[$(date)] GPU0 free. Starting Llama content-capture D0..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D0 --model-id meta-llama/Llama-3.1-8B-Instruct --backend local --limit 10
echo "[$(date)] D0 done. Starting Llama content-capture D1..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D1 --model-id meta-llama/Llama-3.1-8B-Instruct --backend local --limit 10
echo "[$(date)] D1 done. Starting Llama content-capture D2..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/content_capture.py --condition D2 --model-id meta-llama/Llama-3.1-8B-Instruct --backend local --limit 10 --adapter-path /workspace/outputs/checkpoints/dpo_llama8b_v1/final
echo "[$(date)] LLAMA_CONTENT_CAPTURE_ALL_DONE"
