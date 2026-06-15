#!/usr/bin/env python3
"""Smoke-test PLA per-ROI counting (single pass, fast)."""
from PIL import Image
from infer_sam import SAM3LoRAInference
from pla_count import pla_quantify

eng = SAM3LoRAInference(config_path="configs/full_lora_config.yaml",
                        weights_path="weights/medsam3_v1/best_lora_weights.pt",
                        detection_threshold=0.4, nms_iou_threshold=0.5)

pil = Image.open("test_images/pla/pla_plisa_atg_interactions_plos_ccby_g001.png")
print("PLACOUNT_START", flush=True)
img, summary = pla_quantify(eng, pil, "fluorescent spot", "nucleus", use_tiling=False)
img.save("/home/ubuntu/pla_count_result.png")
print(summary, flush=True)
print("PLACOUNT_DONE", flush=True)
