#!/usr/bin/env python3
"""Quick experiment: does raising SAM3 input resolution improve detections?
Loads the model once, varies the resize resolution, prints one RESULT line per run."""
import time
from infer_sam import SAM3LoRAInference
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
)

CONFIG = "configs/full_lora_config.yaml"
WEIGHTS = "weights/medsam3_v1/best_lora_weights.pt"

eng = SAM3LoRAInference(config_path=CONFIG, weights_path=WEIGHTS,
                        detection_threshold=0.4, nms_iou_threshold=0.5)


def set_res(res):
    eng.resolution = res
    eng.transform = ComposeAPI(transforms=[
        RandomResizeAPI(sizes=res, max_size=res, square=True, consistent_transform=False),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


TESTS = [
    ("PLA", "test_images/pla/pla_plisa_atg_interactions_plos_ccby_g001.png",
     ["fluorescent spot", "nucleus", "cell"]),
    ("TNT", "test_images/tnt/tnt_rpe_cells_factin_plos_ccby_g003.png",
     ["tunneling nanotube", "cell membrane protrusion", "cell"]),
]
RESOLUTIONS = [1008, 1536, 2048]

print("SWEEP_START", flush=True)
for label, img, prompts in TESTS:
    for res in RESOLUTIONS:
        set_res(res)
        t0 = time.time()
        try:
            r = eng.predict(img, prompts)
            dt = time.time() - t0
            parts = []
            for idx in sorted([k for k in r if k != "_image"]):
                d = r[idx]
                n = d["num_detections"]
                mx = float(d["scores"].max()) if n else 0.0
                parts.append(f"{d['prompt']}={n}({mx:.2f})")
            print(f"RESULT {label} res={res} time={dt:.0f}s :: " + " | ".join(parts), flush=True)
        except Exception as e:
            print(f"RESULT {label} res={res} ERROR {type(e).__name__}: {e}", flush=True)
print("SWEEP_DONE", flush=True)
