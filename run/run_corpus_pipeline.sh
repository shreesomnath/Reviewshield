#!/bin/bash
set -e
cd /media/airlab/ROCSTOR/revguard
echo "[$(date)] Starting corpus pull (n=800)"
apptainer exec --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/download_arxiv_corpus.py --n 800
echo "[$(date)] Corpus pull done. Running audit."
apptainer exec --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/audit_corpus.py --in /workspace/data/interim/arxiv_corpus.jsonl
echo "[$(date)] Audit done. Running benchmark assembly."
apptainer exec --bind /media/airlab/ROCSTOR/revguard:/workspace revguard.sif   python /workspace/scripts/benchmark_assembly.py --n-papers 800
echo "[$(date)] Corpus pipeline complete."
