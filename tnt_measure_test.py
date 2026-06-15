#!/usr/bin/env python3
"""Quick test of the Frangi TNT measurement on the test images."""
from PIL import Image
from tnt_measure import measure_tnt

IMAGES = [
    "tnt_rpe_cells_factin_plos_ccby_g003.png",
    "tnt_hiv_macrophages_wikimedia_ccby4.png",
    "tnt_mesothelioma_actin_plos_ccby_g003.png",
]
for name in IMAGES:
    pil = Image.open(f"test_images/tnt/{name}")
    for thr in (0.10, 0.18):
        img, summary = measure_tnt(pil, "Automaticky (max)", threshold=thr, min_len_px=25)
        img.save(f"/home/ubuntu/tntres_{thr}_{name}")
        head = summary.split("\n")[2]  # the count line
        print(f"{name}  thr={thr}  ->  {head}", flush=True)
print("TNTTEST_DONE", flush=True)
