#!/usr/bin/env python3
"""TNT prompt bake-off: run each prompt variant on several TNT images, rank by
detections x confidence so we can pick the best default prompts."""
import tempfile
from collections import defaultdict
from PIL import Image
from infer_sam import SAM3LoRAInference

PROMPTS = [
    "tunneling nanotube",
    "thin tube connecting cells",
    "membrane bridge between cells",
    "intercellular bridge",
    "thin membrane protrusion",
    "filament between cells",
    "actin filament",
]
IMAGES = [
    "test_images/tnt/tnt_rpe_cells_factin_plos_ccby_g003.png",
    "test_images/tnt/tnt_hiv_macrophages_wikimedia_ccby4.png",
    "test_images/tnt/tnt_mesothelioma_actin_plos_ccby_g003.png",
    "test_images/tnt/tnt_monocytes_membrane_frontiers_ccby_g002.png",
]

eng = SAM3LoRAInference(config_path="configs/full_lora_config.yaml",
                        weights_path="weights/medsam3_v1/best_lora_weights.pt",
                        detection_threshold=0.35, nms_iou_threshold=0.5)

print("BAKE_START", flush=True)
agg = defaultdict(lambda: {"hits": 0, "dets": 0, "score_sum": 0.0})
for img in IMAGES:
    pil = Image.open(img).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    pil.save(tmp)
    name = img.split("/")[-1][:28]
    for p in PROMPTS:
        r = eng.predict(tmp, [p])[0]
        n = r["num_detections"]
        mx = float(r["scores"].max()) if n else 0.0
        if n:
            agg[p]["hits"] += 1
            agg[p]["dets"] += n
            agg[p]["score_sum"] += mx
        print(f"RESULT img={name} prompt='{p}' dets={n} maxscore={mx:.3f}", flush=True)

print("RANKING (by images-with-hit, then avg max-score):", flush=True)
ranked = sorted(PROMPTS, key=lambda p: (agg[p]["hits"],
                agg[p]["score_sum"] / max(agg[p]["hits"], 1)), reverse=True)
for p in ranked:
    a = agg[p]
    avg = a["score_sum"] / max(a["hits"], 1)
    print(f"RANK '{p}': hits={a['hits']}/{len(IMAGES)} totaldets={a['dets']} "
          f"avgmax={avg:.3f}", flush=True)
print("BAKE_DONE", flush=True)
