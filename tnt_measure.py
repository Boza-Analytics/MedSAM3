"""TNT detection + measurement via a tubularity (Frangi) filter.

SAM3 is poor at hairline tubes; ridge/vesselness filters are the classic tool for
thin curvilinear structures. We enhance tube-like ridges, threshold, skeletonise,
then measure each TNT's length. Returns (PIL overlay, markdown summary with lengths).

No model / GPU / training needed — pure scikit-image.
"""
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import frangi
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, remove_small_objects, binary_closing
from skimage.measure import label, regionprops


def _pick_channel(rgb, channel):
    if channel.startswith("Červený"):
        return rgb[:, :, 0]
    if channel.startswith("Zelený"):
        return rgb[:, :, 1]
    if channel.startswith("Modrý"):
        return rgb[:, :, 2]
    if channel.startswith("Šedá"):
        return rgb2gray(rgb)
    return rgb.max(axis=2)  # Automaticky: max projection (jas v jakémkoli kanálu)


def measure_tnt(pil, channel="Automaticky (max)", threshold=0.12,
                min_len_px=25, pixel_size_um=None, straightness=0.6, max_len_px=1200):
    rgb = np.asarray(pil.convert("RGB")).astype(float) / 255.0
    gray = _pick_channel(rgb, channel)

    tub = frangi(gray, sigmas=range(1, 4), black_ridges=False)
    if tub.max() > 0:
        tub = tub / tub.max()

    mask = tub > float(threshold)
    mask = binary_closing(mask, np.ones((3, 3), bool))
    mask = remove_small_objects(mask, min_size=max(8, int(min_len_px) // 2))
    skel = skeletonize(mask)

    H, W = skel.shape
    margin = 3
    lbl = label(skel)
    keep = np.zeros_like(skel, dtype=bool)
    lengths_px = []
    for r in regionprops(lbl):
        if not (int(min_len_px) <= r.area <= int(max_len_px)):
            continue
        minr, minc, maxr, maxc = r.bbox
        if minr <= margin or minc <= margin or maxr >= H - margin or maxc >= W - margin:
            continue  # drop border/panel-edge artifacts
        # straightness: straight line -> major_axis ≈ pixel count; squiggly/branched -> much less
        if (r.major_axis_length / max(r.area, 1)) < float(straightness):
            continue
        lengths_px.append(int(r.area))
        keep[lbl == r.label] = True

    unit, conv = ("µm", float(pixel_size_um)) if pixel_size_um else ("px", 1.0)
    lengths = sorted([L * conv for L in lengths_px], reverse=True)
    n = len(lengths)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(pil)
    ax.axis("off")
    ys, xs = np.where(keep)
    if len(xs):
        ax.plot(xs, ys, ".", color="magenta", markersize=1.0)
    ax.set_title(
        (f"TNT: {n} struktur · medián délky {np.median(lengths):.1f} {unit}"
         if n else "TNT: 0 nalezeno (zkuste nižší práh / jiný kanál)"),
        fontsize=12)
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()

    lines = ["### Detekce + měření TNT (filtr trubic – Frangi)", "",
             f"- nalezeno struktur (≥{int(min_len_px)} px): **{n}**"]
    if n:
        lines += [
            f"- délka — medián **{np.median(lengths):.1f} {unit}**, "
            f"průměr {np.mean(lengths):.1f}, max {max(lengths):.1f}",
            f"- celková délka: {sum(lengths):.0f} {unit}",
            "", f"| TNT | délka ({unit}) |", "|---|---|"]
        for i, L in enumerate(lengths, 1):
            lines.append(f"| {i} | {L:.1f} |")
        if not pixel_size_um:
            lines.append("\n_Tip: zadejte velikost pixelu (µm/px) pro délky v µm._")
    else:
        lines.append("_Zkuste snížit práh citlivosti nebo vybrat jiný kanál._")
    return Image.open(out), "\n".join(lines)
