# Next steps: from MedSAM3 segmentation → your two research goals

MedSAM3 gives you **masks + boxes** from a text prompt. Your goals need a thin
**analysis layer** on top of those masks, and (very likely) **fine-tuning** so the
masks are actually good on TNT/PLA imagery.

## Goal 1 — Automate TNT detection and measurement
1. **Detect**: segment the nanotubes (prompt or, better, a fine-tuned model).
2. **Measure** (post-processing on each mask — `scikit-image`, already a dependency):
   - *Length*: skeletonize the mask (`skimage.morphology.skeletonize`) and sum the
     skeleton path length; or use `regionprops` `major_axis_length`.
   - *Width/diameter*: distance transform along the skeleton.
   - *Count*: number of distinct TNT instances connecting cell pairs.
   - Convert px → µm using the image's scale bar / known pixel size.

## Goal 2 — Automate PLA spot counting and ROI definition
1. **Define ROI**: segment cells or nuclei (prompt `"nucleus"` / `"cell"`, or fine-tune).
   Each cell/nucleus mask = one ROI.
2. **Count spots** inside each ROI (post-processing):
   - PLA puncta are small bright blobs → `skimage.feature.blob_log` /
     `blob_dog`, or threshold + `scipy.ndimage.label` + count connected components.
   - Report spots-per-cell, intensity, size distribution.
   - (Segmenting tiny puncta directly with SAM3 is hard; blob detection inside the
     ROI mask is the standard, robust approach.)

## The likely-necessary step: LoRA fine-tuning
MedSAM3-v1 is medical-concept trained; TNT/PLA are out of distribution. To get usable masks:
1. **Annotate** a small set (start ~30–100 images) in **COCO format** (e.g. Roboflow,
   CVAT, or Label Studio). The repo even ships `convert_roboflow_to_coco.py`.
2. Lay out data as `data/train`, `data/valid`, `data/test` each with
   `_annotations.coco.json` (standard SAM3/COCO layout).
3. Point `configs/full_lora_config.yaml` → `training.data_dir` at it.
4. Train on the GPU instance:
   ```bash
   python3 train_sam3_lora_native.py --config configs/full_lora_config.yaml
   ```
5. Re-run `deploy/run_examples.sh` pointing `WEIGHTS=` at your new
   `outputs/sam3_lora_full/best_lora_weights.pt`.

## Suggested workflow
1. Get the AWS box running (see `deploy/README_AWS.md`).
2. Run the **zero-shot baseline** on `test_images/` → see how far off the base model is.
3. Decide per task whether prompts alone are good enough or fine-tuning is needed
   (PLA ROI via "cell"/"nucleus" might be okay zero-shot; TNT almost certainly needs FT).
4. Annotate → fine-tune → add the measurement/counting post-processing scripts.
