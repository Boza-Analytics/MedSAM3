#!/bin/bash
# Launch the Gradio frontend (intended to be started via systemd-run).
cd /home/ubuntu/MedSAM3
source /home/ubuntu/venv/bin/activate
export GRADIO_ANALYTICS_ENABLED=False
# service runs as root via systemd-run -> point it at the ubuntu user's HF cache + token
export HOME=/home/ubuntu
export HF_HOME=/home/ubuntu/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:-$(cat /home/ubuntu/.hf_token 2>/dev/null)}"
exec python3 -u /home/ubuntu/MedSAM3/gradio_app.py > /home/ubuntu/fe.log 2>&1
