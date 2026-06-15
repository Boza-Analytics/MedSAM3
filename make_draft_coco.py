#!/usr/bin/env python3
"""Generate DRAFT COCO annotations for TNT images using the Frangi tubularity
filter, so a human only has to *correct* candidates in Roboflow rather than draw
from scratch. Each candidate tube -> one polygon labelled 'tunneling nanotube'.

Usage:  python3 make_draft_coco.py annotate/images
Writes annotate/images/_annotations.coco.json
"""
import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
from skimage.filters import frangi
from skimage.morphology import (skeletonize, remove_small_objects,
                                binary_closing, dilation, disk)
from skimage.measure import label, regionprops


def frangi_components(pil, threshold=0.22, min_len=45, straightness=0.72,
                      margin=3, max_per_image=15):
    """Return up to `max_per_image` of the longest, straightest tube candidates.
    Stricter defaults than the live tool: drafts should be high-precision so a human
    mostly *confirms* rather than *deletes*."""
    rgb = np.asarray(pil.convert("RGB")).astype(float) / 255.0
    gray = rgb.max(axis=2)
    tub = frangi(gray, sigmas=range(1, 4), black_ridges=False)
    if tub.max() > 0:
        tub = tub / tub.max()
    m = tub > threshold
    m = binary_closing(m, np.ones((3, 3), bool))
    m = remove_small_objects(m, min_size=max(8, min_len // 2))
    skel = skeletonize(m)
    H, W = skel.shape
    lbl = label(skel)
    scored = []
    for r in regionprops(lbl):
        if r.area < min_len:
            continue
        minr, minc, maxr, maxc = r.bbox
        if minr <= margin or minc <= margin or maxr >= H - margin or maxc >= W - margin:
            continue
        if (r.major_axis_length / max(r.area, 1)) < straightness:
            continue
        scored.append((r.area, r.label))
    scored.sort(reverse=True)
    comps = [dilation(lbl == lab, disk(2)) for _, lab in scored[:max_per_image]]
    return comps, (W, H)


def comp_to_polys(comp):
    cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1).tolist() for c in cnts if len(c) >= 3]


def main(d):
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    images, anns, aid = [], [], 1
    for iid, f in enumerate(files, 1):
        pil = Image.open(os.path.join(d, f)).convert("RGB")
        comps, (W, H) = frangi_components(pil)
        images.append({"id": iid, "file_name": f, "width": W, "height": H})
        for comp in comps:
            polys = comp_to_polys(comp)
            if not polys:
                continue
            ys, xs = np.where(comp)
            anns.append({
                "id": aid, "image_id": iid, "category_id": 1, "iscrowd": 0,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "area": int(comp.sum()), "segmentation": polys})
            aid += 1
        print(f"{f}: {sum(a['image_id'] == iid for a in anns)} draft candidates", flush=True)
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "tunneling nanotube", "supercategory": "cell"}]}
    out = os.path.join(d, "_annotations.coco.json")
    with open(out, "w") as fh:
        json.dump(coco, fh, indent=1)
    print(f"WROTE {out}: {len(images)} images, {len(anns)} draft annotations", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "annotate/images")
