"""Download the HF weights this project needs, straight to the
project-local HF cache (HF_HOME=/workspace/models per revguard.def, bound
from ROCSTOR - plenty of space for the full bf16 72B checkpoint, ~145GB).

Order: smallest/fastest first, so an auth or config problem surfaces early
rather than after hours of downloading the 72B model. Llama-3.1-8B-Instruct
is gated on Hugging Face (requires accepting Meta's license and an approved
token); it is attempted but wrapped so a missing/unapproved HF_TOKEN does
not abort the Qwen downloads, which are fully open.

Usage (inside the container):
    apptainer exec --env HF_TOKEN=xxxx --bind ... revguard.sif \
        python /workspace/scripts/download_models.py
"""
import sys
import time

from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen2.5-14B-Instruct", "primary reviewer base (defended model)"),
    ("meta-llama/Llama-3.1-8B-Instruct", "secondary reviewer base (robustness check, GATED)"),
    ("Qwen/Qwen2.5-72B-Instruct", "local generator/judge for preference-pair data"),
]


def main():
    results = {}
    for repo_id, role in MODELS:
        print(f"\n=== Downloading {repo_id} ({role}) ===", flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(
                repo_id=repo_id,
                ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx"],
                # keep safetensors only; skip redundant/legacy weight formats
            )
            dt = time.time() - t0
            print(f"OK: {repo_id} -> {path}  ({dt/60:.1f} min)", flush=True)
            results[repo_id] = {"status": "ok", "path": path, "minutes": dt / 60}
        except Exception as e:
            dt = time.time() - t0
            print(f"FAILED: {repo_id} after {dt/60:.1f} min: {e}", flush=True)
            results[repo_id] = {"status": "failed", "error": str(e)}
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print(f"  -> looks like a gating/auth issue; needs an approved "
                      f"HF_TOKEN with access to {repo_id}. Skipping, continuing "
                      f"with remaining models.", flush=True)
            else:
                raise  # unexpected failure type: don't silently continue

    print("\n=== Summary ===")
    for repo_id, r in results.items():
        print(f"  {repo_id}: {r['status']}")
    n_failed = sum(1 for r in results.values() if r["status"] == "failed")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
