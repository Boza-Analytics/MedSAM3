"""PLA quantification: count fluorescent spots per ROI (nucleus/cell).

Runs SAM3 for the spot prompt and the ROI prompt, assigns each spot to the ROI
whose mask contains the spot centroid, and renders nuclei (outlined + numbered
with their spot count) plus the spots (green = inside an ROI, red = outside).
Returns (PIL image, markdown summary).
"""
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tiling import tiled_predict


def _detect(eng, pil, prompts, use_tiling):
    if use_tiling:
        return tiled_predict(eng, pil, prompts)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    pil.save(tmp)
    return eng.predict(tmp, prompts)


def pla_quantify(eng, pil, spot_prompt="fluorescent spot", roi_prompt="nucleus",
                 use_tiling=True):
    pil = pil.convert("RGB")
    W, H = pil.size
    res = _detect(eng, pil, [spot_prompt, roi_prompt], use_tiling)
    spots, rois = res[0], res[1]

    boxes = spots["boxes"] if spots["num_detections"] else np.zeros((0, 4))
    centroids = (np.stack([(boxes[:, 0] + boxes[:, 2]) / 2,
                           (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
                 if len(boxes) else np.zeros((0, 2)))
    roi_masks = (rois["masks"] if rois["num_detections"] and rois["masks"] is not None
                 else [])
    nroi = len(roi_masks)

    roi_centroids = []
    roi_radii = []
    for m in roi_masks:
        ys, xs = np.where(m)
        if len(xs):
            roi_centroids.append((xs.mean(), ys.mean()))
            roi_radii.append((m.sum() / np.pi) ** 0.5)  # equiv. radius
        else:
            roi_centroids.append((0, 0))
            roi_radii.append(0)
    rc = np.array(roi_centroids) if nroi else np.zeros((0, 2))
    # spots within ~2 nucleus radii of a centre are attributed to that cell
    reach = (np.median([r for r in roi_radii if r > 0]) * 2.5) if any(roi_radii) else 0

    counts = [0] * nroi
    assign = []
    outside = 0
    for cx, cy in centroids:
        ix = min(max(int(round(cx)), 0), W - 1)
        iy = min(max(int(round(cy)), 0), H - 1)
        hit = -1
        # 1) inside a mask?
        for j, m in enumerate(roi_masks):
            if m[iy, ix]:
                hit = j
                break
        # 2) else nearest centre within reach
        if hit < 0 and nroi:
            d = np.hypot(rc[:, 0] - cx, rc[:, 1] - cy)
            j = int(d.argmin())
            if d[j] <= reach:
                hit = j
        assign.append(hit)
        if hit >= 0:
            counts[hit] += 1
        else:
            outside += 1

    # ---- render ----
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(pil)
    ax.axis("off")
    for j, m in enumerate(roi_masks):
        ax.contour(m, levels=[0.5], colors="cyan", linewidths=1.2)
        cxr, cyr = roi_centroids[j]
        ax.text(cxr, cyr, f"{j+1}:{counts[j]}", color="white", fontsize=10,
                ha="center", va="center",
                bbox=dict(facecolor="navy", alpha=0.7, pad=1))
    for (cx, cy), a in zip(centroids, assign):
        ax.plot(cx, cy, "o", markersize=4,
                color="lime" if a >= 0 else "red")

    total = len(centroids)
    assigned = total - outside
    ax.set_title(f"PLA: {total} teček · {assigned} v ROI · {outside} mimo · {nroi} ROI",
                 fontsize=12)
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()

    lines = [f"### Počítání PLA teček na ROI ({roi_prompt})", "",
             f"- ROI (oblastí): **{nroi}**",
             f"- PLA teček celkem: **{total}**",
             f"- v ROI: **{assigned}** · mimo ROI: **{outside}**"]
    if nroi:
        lines.append(f"- průměr teček na ROI: **{assigned / nroi:.1f}**")
        lines += ["", "| ROI | tečky |", "|---|---|"]
        for j in range(nroi):
            lines.append(f"| {j+1} | {counts[j]} |")
    return Image.open(out), "\n".join(lines)
