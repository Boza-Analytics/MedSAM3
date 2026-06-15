# Deploying MedSAM3 on AWS (GPU)

This is the recommended way to run MedSAM3 for your TNT / PLA experiments. SAM3 is a large
model and the code path is CUDA-only, so a single NVIDIA GPU instance is the sweet spot.

> ⚠️ **Cost discipline:** GPU instances are expensive (~$0.5–1.3 / hour). **STOP the instance
> when you're not using it.** You still pay for the EBS disk while stopped (cheap), but not for
> the GPU. Don't leave it running overnight.

---

## 1. Pick an instance type

| Use case | Instance | GPU | On-demand (eu-central-1, approx) | Notes |
|---|---|---|---|---|
| **Inference / experimenting (start here)** | `g5.xlarge` | 1× A10G 24 GB | ~$1.30/hr | Comfortable headroom at 1008px. **Recommended.** |
| Cheapest that works | `g4dn.xlarge` | 1× T4 16 GB | ~$0.75/hr | Fine for inference, tighter on memory. |
| Fine-tuning later | `g5.2xlarge`+ | 1× A10G 24 GB | ~$1.55/hr | More CPU/RAM for data loading + training. |

You're already in **eu-central-1 (Frankfurt)** — stay there.

## 2. Check your GPU quota FIRST (this bites people)

New-ish accounts often have a **0 vCPU quota for GPU instances**, which blocks launch.
- Service Quotas → search **"Running On-Demand G and VT instances"**.
- You need at least **4 vCPUs** (g5.xlarge / g4dn.xlarge are 4 vCPU each).
- If it's 0, request an increase to e.g. 8. Approval can take a few hours to a couple of days,
  so do this before anything else.

## 3. Launch the instance

- **AMI:** search the AWS console for
  **"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"**.
  It ships with the NVIDIA driver + CUDA already installed (saves a lot of pain).
- **Instance type:** `g5.xlarge` (from step 1).
- **Key pair:** use an existing one or create a new `.pem`.
- **Storage:** bump the root EBS volume to **100 GB gp3** (default 8 GB is nowhere near
  enough — torch + SAM3 weights + LoRA are several GB).
- **Security group:** inbound **SSH (22) from My IP** only. Nothing else needed.

## 4. Connect and run setup

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# sanity check the GPU is visible
nvidia-smi

# get your fork (includes the deploy scripts + test images)
git clone https://github.com/Boza-Analytics/MedSAM3.git
cd MedSAM3

# one-shot environment setup (venv, CUDA torch, deps, LoRA download)
bash deploy/setup_ec2.sh
```

`setup_ec2.sh` will pause and tell you to run `huggingface-cli login` — follow
[`deploy/HF_SETUP.md`](HF_SETUP.md) to get your token first.

## 5. Run inference

```bash
source ~/medsam3-venv/bin/activate
cd ~/MedSAM3
bash deploy/run_examples.sh        # runs TNT + PLA prompts over the test images
```

Outputs (annotated PNGs) land in `deploy/results/`. Pull them back to your laptop with:
```bash
scp -i your-key.pem -r ubuntu@<EC2_PUBLIC_IP>:~/MedSAM3/deploy/results ./results
```

---

## Important expectations (read this)

MedSAM3-v1 was trained on **medical concepts** (CT/MRI/histopathology/etc.). Your tasks —
**Tunneling Nanotubes** and **Proximity Ligation Assay spots** — are niche fluorescence
microscopy targets that are almost certainly **not in its concept vocabulary**.

So treat the zero-shot run as a **baseline experiment**, not a finished tool:
- It may segment whole cells / nuclei reasonably ("cell", "nucleus" prompts).
- It will probably **struggle** with thin TNT tubes and tiny PLA puncta out of the box.
- For real "automated TNT measurement" and "PLA spot counting + ROI", you'll most likely need
  to **annotate a small dataset (COCO format) and LoRA-fine-tune** using
  `train_sam3_lora_native.py`. The zero-shot results tell you how far off the base is and
  which prompts are worth fine-tuning.

See [`../NEXT_STEPS.md`](../NEXT_STEPS.md) for the fine-tuning path.
