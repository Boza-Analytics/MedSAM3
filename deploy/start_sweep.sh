#!/bin/bash
cd /home/ubuntu/MedSAM3
source /home/ubuntu/venv/bin/activate
export HOME=/home/ubuntu
export HF_HOME=/home/ubuntu/.cache/huggingface
export HF_TOKEN="$(cat /home/ubuntu/.hf_token 2>/dev/null)"
exec python3 -u /home/ubuntu/MedSAM3/sweep_resolution.py > /home/ubuntu/sweep.log 2>&1
