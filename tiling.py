"""Tiled (sliding-window) inference for SAM3.

SAM3 is locked to 1008x1008 input, so a big image gets squashed and small
structures vanish. Instead we split the image into overlapping crops, run SAM3
on each at native detail, offset detections back to global coordinates, and
merge overlaps with NMS. Returns the same dict shape as SAM3LoRAInference.predict
so it can be fed straight to .visualize().
"""
import tempfile
import numpy as np
import torch
from torchvision.ops import nms


def _axis_starts(length, tile, step):
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile + 1, step))
    if xs[-1] != length - tile:
        xs.append(length - tile)
    return xs


def tiled_predict(eng, pil_image, prompts, tile=1008, overlap=0.25):
    pil_image = pil_image.convert("RGB")
    W, H = pil_image.size

    # small image -> just a normal single pass
    if W <= tile and H <= tile:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        pil_image.save(tmp)
        return eng.predict(tmp, prompts)

    step = max(1, int(tile * (1 - overlap)))
    xs = _axis_starts(W, tile, step)
    ys = _axis_starts(H, tile, step)

    agg = {i: {"boxes": [], "scores": [], "masks": []} for i in range(len(prompts))}
    for y in ys:
        for x in xs:
            x2, y2 = min(x + tile, W), min(y + tile, H)
            crop = pil_image.crop((x, y, x2, y2))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            crop.save(tmp)
            res = eng.predict(tmp, prompts)
            for i in range(len(prompts)):
                d = res[i]
                if not d["num_detections"]:
                    continue
                b = d["boxes"].copy()
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                agg[i]["boxes"].append(b)
                agg[i]["scores"].append(d["scores"])
                if d["masks"] is not None:
                    for m in d["masks"]:
                        full = np.zeros((H, W), dtype=bool)
                        ch, cw = m.shape
                        full[y:y + ch, x:x + cw] = m[:H - y, :W - x]
                        agg[i]["masks"].append(full)

    results = {"_image": pil_image}
    for i, p in enumerate(prompts):
        if not agg[i]["boxes"]:
            results[i] = {"prompt": p, "boxes": None, "scores": None,
                          "masks": None, "num_detections": 0}
            continue
        B = np.concatenate(agg[i]["boxes"]).astype(np.float32)
        S = np.concatenate(agg[i]["scores"]).astype(np.float32)
        keep = nms(torch.from_numpy(B), torch.from_numpy(S),
                   eng.nms_iou_threshold).numpy()
        masks = np.stack(agg[i]["masks"])[keep] if agg[i]["masks"] else None
        results[i] = {"prompt": p, "boxes": B[keep], "scores": S[keep],
                      "masks": masks, "num_detections": int(len(keep))}
    return results
