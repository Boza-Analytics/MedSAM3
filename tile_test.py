#!/usr/bin/env python3
"""Compare whole-image vs tiled inference on a large image."""
import time
import tempfile
from PIL import Image
from infer_sam import SAM3LoRAInference
from tiling import tiled_predict

eng = SAM3LoRAInference(config_path="configs/full_lora_config.yaml",
                        weights_path="weights/medsam3_v1/best_lora_weights.pt",
                        detection_threshold=0.4, nms_iou_threshold=0.5)

IMG = "test_images/pla/pla_plisa_atg_interactions_plos_ccby_g001.png"
prompts = ["fluorescent spot", "nucleus", "cell"]
pil = Image.open(IMG).convert("RGB")


def summ(r):
    return " | ".join(f"{r[i]['prompt']}={r[i]['num_detections']}"
                      f"({(float(r[i]['scores'].max()) if r[i]['num_detections'] else 0):.2f})"
                      for i in range(len(prompts)))


print("TILETEST_START  image=%dx%d" % pil.size, flush=True)
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
pil.save(tmp)
t0 = time.time(); rw = eng.predict(tmp, prompts); tw = time.time() - t0
print(f"WHOLE time={tw:.0f}s :: {summ(rw)}", flush=True)

t0 = time.time(); rt = tiled_predict(eng, pil, prompts, tile=1008, overlap=0.25); tt = time.time() - t0
print(f"TILED time={tt:.0f}s :: {summ(rt)}", flush=True)
print("TILETEST_DONE", flush=True)
