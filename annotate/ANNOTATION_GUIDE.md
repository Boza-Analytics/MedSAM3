# TNT annotation guide (≈1–2 hours, the one step only you can do)

We have **50 open-access TNT images** in `annotate/images/` and auto-generated **draft**
outlines (`_annotations.coco.json`) from the Frangi filter. Your job: **correct the drafts**
in Roboflow (much faster than drawing from scratch). You understand TNTs — you are the
"intelligence" that decides what's a real TNT.

## 1. Create a Roboflow project (free)
- Sign up at https://roboflow.com → **Create New Project**
- Type: **Instance Segmentation** · Class name: **tunneling nanotube**

## 2. Upload images + the draft labels
- Drag the whole `annotate/images/` folder (the 50 images **and** `_annotations.coco.json`)
  into the upload box. Roboflow reads the COCO file and shows the draft TNT outlines.

## 3. Correct (the actual work)
For each image:
- **Delete** outlines that are NOT TNTs (stress fibers inside cells, cell edges, debris).
- **Fix** outlines that are close but off.
- **Add** any real TNTs the filter missed (draw along the tube).
- Rule of thumb: a TNT is a **thin tube spanning between two cells, hovering over background**
  — not the dense actin mesh inside a cell body.
- It's fine if some images end up with **0** TNTs — delete all and move on.

## 4. Split + export
- Roboflow → **Generate** a version. Train/Valid/Test split **70/20/10**.
- (Optional) add light augmentation: horizontal/vertical flip, 90° rotation. Microscopy is
  orientation-invariant, so this safely multiplies the data.
- **Export** → format **COCO** → "download zip to computer".
- The zip contains `train/ valid/ test/`, each with images + `_annotations.coco.json` —
  **exactly** the layout the training script needs.

## 5. Hand back
- Unzip into the repo as `data/` (so you have `data/train`, `data/valid`, `data/test`).
- Tell me it's ready — I'll run `deploy/train_tnt.sh` on a GPU instance and deploy the
  fine-tuned model into the web app.

## Notes
- More corrected images = better model. 50 is a starter set; even 30 well-labelled images
  will show whether fine-tuning helps. You can add your own TNT images to the Roboflow
  project too.
- Keep the class name exactly **tunneling nanotube** — it becomes the model's text prompt.
