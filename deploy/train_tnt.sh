#!/usr/bin/env bash
# Fine-tune MedSAM3 (LoRA) on the corrected TNT dataset. Run on a GPU instance.
#   1) put the Roboflow COCO export at  ~/MedSAM3/data/{train,valid,test}/
#   2) bash deploy/train_tnt.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source ~/venv/bin/activate 2>/dev/null || true

# sanity: GPU + data present
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU — training needs a GPU instance'; print('GPU:', torch.cuda.get_device_name(0))"
for s in train valid; do
  [ -f "data/$s/_annotations.coco.json" ] || { echo "Missing data/$s/_annotations.coco.json"; exit 1; }
done

# point the config at ./data and a TNT output dir (edit config once instead if preferred)
python3 - <<'PY'
import yaml, pathlib
p = pathlib.Path("configs/full_lora_config.yaml")
c = yaml.safe_load(p.read_text())
c["training"]["data_dir"] = str(pathlib.Path("data").resolve())
c["output"]["output_dir"] = "outputs/sam3_lora_tnt"
p.write_text(yaml.safe_dump(c, sort_keys=False))
print("config -> data_dir=./data, output_dir=outputs/sam3_lora_tnt")
PY

echo "Starting training…"
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml

echo "Done. New weights: outputs/sam3_lora_tnt/best_lora_weights.pt"
echo "Deploy: set WEIGHTS in gradio_app.py to that path (or copy over weights/medsam3_v1/),"
echo "then: sudo systemctl restart medsam3fe"
