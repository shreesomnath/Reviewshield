#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Waiting for D0-Llama eval to finish (file-existence check, not pgrep)..."
while [ ! -f outputs/eval/D0_meta-llama_Llama-3.1-8B-Instruct.json ]; do sleep 30; done
echo "[$(date)] D0-Llama done. Starting long-form validation on GPU1..."
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=1 --env TRITON_CACHE_DIR=/workspace/scratch/.triton --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/longform_validation.py
echo "[$(date)] Long-form validation complete."
