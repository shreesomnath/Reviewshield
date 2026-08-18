#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Starting combined Gemini D0/D1 regeneration (main eval fix + long-form addition)..."
apptainer exec --env GEMINI_API_KEY="$GEMINI_API_KEY" --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif python -u /workspace/scripts/evaluation/run_eval_gemini_combined.py
echo "[$(date)] Combined Gemini run complete."
