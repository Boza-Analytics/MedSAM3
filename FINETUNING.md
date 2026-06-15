# Fine-tuning MedSAM3 for TNT / PLA

Zero-shot, the model only knows its ~330 trained medical concepts — **TNT isn't one of
them**, and PLA spot detection, while decent, can be sharpened. Fine-tuning (LoRA) is how
you *teach the model your concept* from your own labelled images. This is the single
biggest lever for TNT.

## The honest prerequisites

1. **A GPU.** LoRA training on CPU is impractical (days per run). You need the **g4dn/g5
   GPU instance** — which means the AWS **GPU quota** has to be raised first (currently 0).
   Inference runs on CPU; *training really doesn't*.
2. **Labelled data.** This is the actual work. Quality beats quantity, but plan for:
   - **TNT:** ~50–200 images with the nanotubes annotated (polygons or masks).
   - **PLA:** ~30–100 images; spots are tiny and numerous — annotating every dot is
     painful, so consider labelling a representative subset or using point/box labels.
3. **Diversity.** Different cell types, magnifications, stains, microscopes → generalises.

## Step 1 — Annotate (COCO format)

Use any of: **Roboflow** (easiest, exports COCO + the repo has `convert_roboflow_to_coco.py`),
**CVAT**, **Label Studio**, or **QuPath**.

- Draw a mask/polygon around each TNT (or each PLA spot / nucleus).
- **Category name matters:** it becomes the *text concept* the model learns. Name it
  exactly what you'll prompt with, e.g. `tunneling nanotube`, `PLA spot`, `nucleus`.

## Step 2 — Lay out the data

```
data/
├── train/  (images + _annotations.coco.json)
├── valid/  (images + _annotations.coco.json)
└── test/   (images + _annotations.coco.json)
```
~70/20/10 split. This is the standard SAM3/COCO layout the training script expects.

## Step 3 — Configure

In `configs/full_lora_config.yaml`:
```yaml
training:
  data_dir: "/home/ubuntu/MedSAM3/data"   # your data root
  batch_size: 4            # lower if GPU OOM
  learning_rate: 5e-5
  num_epochs: 100          # small dataset -> fewer; watch validation
lora:
  rank: 16                 # 8–32; higher = more capacity, more VRAM
output:
  output_dir: "outputs/sam3_lora_tnt"
```

## Step 4 — Train (on the GPU box)

```bash
source ~/venv/bin/activate
python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
```
Checkpoints land in `output_dir`; `best_lora_weights.pt` is the one to use.

## Step 5 — Use your new weights

Point inference / the frontend at the new checkpoint:
```bash
# CLI
python3 infer_sam.py --config configs/full_lora_config.yaml \
  --weights outputs/sam3_lora_tnt/best_lora_weights.pt \
  --image your_tnt.png --prompt "tunneling nanotube"
```
For the web UI, set `WEIGHTS` in `gradio_app.py` to the new path and restart the service.

## Practical tips

- **Start small** to validate the loop (20–30 images, few epochs), then scale up.
- **Small data overfits fast** — use the validation split, early-stop, augment (flips,
  rotations, intensity jitter; microscopy is rotation-invariant).
- **TNT specifically:** because tubes are thin, annotate generously and consider also
  training a `cell` class so the model learns figure-ground context.
- **Don't expect miracles from 10 images** — the jump from zero-shot usually needs a few
  dozen good annotations.

## Alternative / complement for TNT

Thin tubular structures are also the classic domain of **tubularity (Frangi/Sato vesselness)
filters** + skeletonisation for length. That needs *no* training and may beat zero-shot
SAM3 for TNT today — worth running alongside a fine-tuned model.
