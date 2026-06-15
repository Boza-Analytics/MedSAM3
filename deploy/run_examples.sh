#!/usr/bin/env bash
# Run MedSAM3 zero-shot inference over the TNT + PLA test images with several
# candidate text prompts, saving annotated visualizations to deploy/results/.
#
# Prereqs: setup_ec2.sh done, `huggingface-cli login` done, LoRA weights downloaded:
#   huggingface-cli download lal-Joey/MedSAM3_v1 --local-dir weights/medsam3_v1
#
# Usage:  bash deploy/run_examples.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG="configs/full_lora_config.yaml"
THRESHOLD="${THRESHOLD:-0.4}"     # lower than default 0.5 — niche targets score low
NMS_IOU="${NMS_IOU:-0.5}"
OUTDIR="deploy/results"
mkdir -p "$OUTDIR"

# --- locate LoRA weights -----------------------------------------------------
WEIGHTS="${WEIGHTS:-}"
if [[ -z "$WEIGHTS" ]]; then
  WEIGHTS="$(find weights/medsam3_v1 outputs -type f \( -name '*.pt' -o -name '*.safetensors' \) 2>/dev/null | head -1 || true)"
fi
if [[ -z "$WEIGHTS" || ! -f "$WEIGHTS" ]]; then
  echo "!! Could not find LoRA weights. Download them first:"
  echo "   huggingface-cli download lal-Joey/MedSAM3_v1 --local-dir weights/medsam3_v1"
  echo "   (or set WEIGHTS=/path/to/weights.pt)"
  exit 1
fi
echo "==> Using LoRA weights: $WEIGHTS"
echo "==> threshold=$THRESHOLD nms-iou=$NMS_IOU"

run() {  # run <image> <output_name> <prompt...>
  local img="$1"; local name="$2"; shift 2
  if [[ ! -f "$img" ]]; then echo "   (skip, missing: $img)"; return; fi
  echo "----> $img  prompts: $*"
  python3 infer_sam.py \
    --config "$CONFIG" \
    --weights "$WEIGHTS" \
    --image "$img" \
    --prompt "$@" \
    --threshold "$THRESHOLD" \
    --nms-iou "$NMS_IOU" \
    --boundingbox True \
    --output "$OUTDIR/$name.png"
}

echo "############### TNT (tunneling nanotubes) ###############"
for img in test_images/tnt/*; do
  base="$(basename "${img%.*}")"
  run "$img" "tnt__${base}" "tunneling nanotube" "cell" "cell membrane protrusion"
done

echo "############### PLA (proximity ligation assay) ###############"
for img in test_images/pla/*; do
  base="$(basename "${img%.*}")"
  run "$img" "pla__${base}" "fluorescent spot" "nucleus" "cell"
done

echo
echo "==> Done. Annotated results in $OUTDIR/"
echo "    Pull them back with:"
echo "    scp -i your-key.pem -r ubuntu@<EC2_IP>:~/MedSAM3/$OUTDIR ./results"
