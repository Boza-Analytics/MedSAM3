#!/usr/bin/env python3
"""Test the user's cascade idea: Frangi-highlight the tubes, then feed the
highlighted image to SAM3. Compare SAM3 detections on ORIGINAL vs HIGHLIGHTED."""
import tempfile
import numpy as np
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import (skeletonize, remove_small_objects,
                                binary_closing, dilation, disk)
from infer_sam import SAM3LoRAInference


def frangi_mask(pil, threshold=0.12, min_len=25):
    rgb = np.asarray(pil.convert("RGB")).astype(float) / 255.0
    gray = rgb.max(axis=2)
    tub = frangi(gray, sigmas=range(1, 4), black_ridges=False)
    if tub.max() > 0:
        tub = tub / tub.max()
    m = tub > threshold
    m = binary_closing(m, np.ones((3, 3), bool))
    m = remove_small_objects(m, min_size=max(8, min_len // 2))
    return skeletonize(m)


def highlight(pil, mask, color=(0, 255, 255)):
    arr = np.asarray(pil.convert("RGB")).copy()
    arr[dilation(mask, disk(2))] = color
    return Image.fromarray(arr)


eng = SAM3LoRAInference(config_path="configs/full_lora_config.yaml",
                        weights_path="weights/medsam3_v1/best_lora_weights.pt",
                        detection_threshold=0.35, nms_iou_threshold=0.5)

PROMPTS = ["membrane bridge between cells", "thin membrane protrusion"]
IMAGES = [
    "test_images/tnt/tnt_rpe_cells_factin_plos_ccby_g003.png",
    "test_images/tnt/tnt_hiv_macrophages_wikimedia_ccby4.png",
]

print("CASCADE_START", flush=True)
for img in IMAGES:
    pil = Image.open(img).convert("RGB")
    hl = highlight(pil, frangi_mask(pil))
    hl.save("/home/ubuntu/hl_" + img.split("/")[-1])
    name = img.split("/")[-1][:24]
    for label, im in [("ORIG", pil), ("HILITE", hl)]:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        im.save(tmp)
        for p in PROMPTS:
            r = eng.predict(tmp, [p])[0]
            n = r["num_detections"]
            mx = float(r["scores"].max()) if n else 0.0
            print(f"RESULT {label} img={name} prompt='{p}' dets={n} maxscore={mx:.3f}",
                  flush=True)
print("CASCADE_DONE", flush=True)
