#!/usr/bin/env bash
# CPU-compatibility patches for running MedSAM3 / SAM3 inference on a CPU-only box.
#
# The vendored sam3/ code is written CUDA-only: it hardcodes device="cuda",
# calls .cuda()/.pin_memory(), and the LoRA checkpoint is CUDA-serialized.
# This script rewrites those to CPU so `infer_sam.py` runs without a GPU.
#
# ⚠️ This makes the checkout CPU-only. On a real GPU box, do NOT run this.
# Run from the repo root AFTER `pip install -e .`:
#   bash deploy/cpu_compat.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] install triton (imported at module load by sam3)..."
pip install -q triton || true

echo "[2/4] redirect hardcoded CUDA -> CPU in sam3/ ..."
find sam3 -name "*.py" -print0 | xargs -0 sed -i \
  -e 's/device="cuda"/device="cpu"/g' \
  -e 's/\.cuda(non_blocking=True)/.cpu()/g' \
  -e 's/\.cuda()/.cpu()/g'

echo "[3/4] remove GPU-only .pin_memory() calls ..."
find sam3 -name "*.py" -print0 | xargs -0 sed -i 's/\.pin_memory()//g'

echo "[4/4] load LoRA checkpoint with map_location=cpu ..."
sed -i 's/torch.load(load_path)/torch.load(load_path, map_location="cpu")/' lora_layers.py

echo "CPU-compat patches applied. Inference will run on CPU (slow but functional)."
