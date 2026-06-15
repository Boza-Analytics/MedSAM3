#!/usr/bin/env bash
# One-shot environment setup for MedSAM3 on an AWS GPU instance.
# Target: Ubuntu 22.04 "Deep Learning Base OSS Nvidia Driver GPU AMI"
# (NVIDIA driver + CUDA already present). Run from inside the cloned MedSAM3 repo:
#   bash deploy/setup_ec2.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/medsam3-venv"

echo "==> Repo: $REPO_DIR"
echo "==> Checking GPU is visible..."
nvidia-smi || { echo "!! nvidia-smi failed — are you on a GPU instance with the driver AMI?"; exit 1; }

echo "==> Installing system packages..."
sudo apt-get update -y
sudo apt-get install -y \
  python3.11 python3.11-venv python3.11-dev \
  build-essential git \
  ffmpeg libsm6 libxext6   # OpenCV runtime deps

echo "==> Creating virtualenv at $VENV ..."
python3.11 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip wheel setuptools

echo "==> Installing CUDA-enabled PyTorch (cu124) first so deps see it satisfied..."
pip install "torch==2.7.*" torchvision --index-url https://download.pytorch.org/whl/cu124

echo "==> Installing MedSAM3 + remaining dependencies..."
cd "$REPO_DIR"
pip install -e .

echo "==> Verifying torch sees CUDA..."
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cat <<'EOF'

============================================================
NEXT: log in to Hugging Face (see deploy/HF_SETUP.md)
------------------------------------------------------------
  source ~/medsam3-venv/bin/activate
  huggingface-cli login        # paste your hf_... token
  huggingface-cli whoami       # confirm

Then download the MedSAM3 LoRA weights:
  huggingface-cli download lal-Joey/MedSAM3_v1 --local-dir weights/medsam3_v1

Then run the experiments:
  bash deploy/run_examples.sh
============================================================
EOF
